from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import paired_bootstrap as pb
import run_maxent_phoenix as phoenix
import run_maxent_anneal as anneal


def _save_bundle(
    path,
    numerator,
    sign,
    *,
    observable="g",
    tau=None,
    beta=None,
):
    numerator = np.asarray(numerator)
    sign = np.asarray(sign)
    if tau is None:
        tau = np.arange(numerator.shape[1], dtype=float) * 0.1
    if beta is None:
        beta = 0.1 * numerator.shape[1]
    metadata = {
        "format_version": pb.FORMAT_VERSION,
        "observable": observable,
        "beta": beta,
        "dt": 0.1,
        "L": numerator.shape[1],
        "normalization": "test",
    }
    bundle = pb.validate_paired_bundle(
        numerator=numerator,
        sign=sign,
        n_sample=np.full(sign.shape, 10.0),
        tau=tau,
        mean=pb.ratio_of_sums(numerator, sign),
        stderr=np.zeros(numerator.shape[1]),
        metadata=metadata,
        source_files=np.asarray(
            [f"bin_{index}.h5" for index in range(sign.size)]
        ),
    )
    pb.save_paired_bundle(path, bundle)
    return bundle


def test_prepare_rows_preserves_ratio_center_and_bootstrap_covariance(tmp_path):
    sign = np.array([3.0, -0.5, 2.0, 4.0])
    numerator = np.array(
        [
            [6.0, 3.0, 9.0, 12.0],
            [1.0, 2.0, 1.0, 3.0],
            [5.0, 2.0, 4.0, 7.0],
            [8.0, 7.0, 5.0, 9.0],
        ]
    )
    bundle_path = tmp_path / "g_paired.npz"
    bundle = _save_bundle(bundle_path, numerator, sign)

    prepared = phoenix.prepare_paired_maxent_input(
        bundle_path,
        64,
        rng=np.random.default_rng(7),
        bootstrap_block_size=1,
    )
    rows = prepared["chi"]
    estimates = pb.bootstrap_ratio_of_sums(
        bundle.numerator,
        bundle.sign,
        prepared["indices"],
    )

    np.testing.assert_allclose(rows.mean(axis=0), bundle.mean)
    np.testing.assert_allclose(
        np.cov(rows, rowvar=False) / rows.shape[0],
        np.cov(estimates, rowvar=False),
    )
    assert prepared["metadata"]["source_nbin"] == 4
    assert prepared["metadata"]["maxent_nrow"] == 64
    assert prepared["metadata"]["covariance_bootstrap_samples"] == 64
    assert prepared["metadata"]["sign_reweighting"] == (
        "paired_bootstrap_ratio_of_sums"
    )


def test_append_uses_same_paired_bootstrap_indices(tmp_path):
    sign = np.array([2.0, 1.0, 3.0, 2.0])
    numerator = np.arange(16.0).reshape(4, 4) + 1.0
    append_numerator = np.array([[2.0], [5.0], [9.0], [4.0]])
    bundle_path = tmp_path / "corr_paired.npz"
    append_path = tmp_path / "append_paired.npz"
    _save_bundle(bundle_path, numerator, sign, observable="JNJN")
    append_bundle = _save_bundle(
        append_path,
        append_numerator,
        sign,
        observable="JNJN_beta",
        tau=np.array([0.4]),
        beta=0.4,
    )

    prepared = phoenix.prepare_paired_maxent_input(
        bundle_path,
        40,
        rng=np.random.default_rng(11),
        append_bundle_path=append_path,
    )
    append_estimates = pb.bootstrap_ratio_of_sums(
        append_bundle.numerator,
        append_bundle.sign,
        prepared["indices"],
    )

    np.testing.assert_allclose(
        prepared["append"].mean(axis=0),
        append_bundle.mean,
    )
    np.testing.assert_allclose(
        np.cov(prepared["append"], rowvar=False)
        / prepared["append"].shape[0],
        np.cov(append_estimates, rowvar=False),
    )


def test_prepare_rejects_non_negligible_complex_correlator(tmp_path):
    sign = np.array([2.0 + 0.2j, 1.0 + 0.1j, 3.0 - 0.1j])
    numerator = np.array(
        [
            [2.0 + 1.0j, 3.0, 4.0, 5.0],
            [1.0, 2.0 + 0.5j, 3.0, 4.0],
            [4.0, 3.0, 2.0 + 0.3j, 1.0],
        ]
    )
    bundle_path = tmp_path / "complex_paired.npz"
    _save_bundle(bundle_path, numerator, sign)

    with pytest.raises(ValueError, match="non-negligible imaginary part"):
        phoenix.prepare_paired_maxent_input(
            bundle_path,
            20,
            rng=np.random.default_rng(3),
        )


def test_outer_spectrum_bootstrap_resamples_append_with_correlator(monkeypatch):
    chi = np.arange(24.0).reshape(6, 4)
    append = (10.0 * chi[:, :1]) + 1.0
    seen = []

    def fake_preprocess(
        sampled_chi,
        dt,
        beta,
        *,
        grid_info,
        op_type,
        sym,
        model_arr,
        append,
    ):
        seen.append((sampled_chi.copy(), append.copy()))
        return {
            "tau": np.arange(sampled_chi.shape[1], dtype=float) * dt,
            "lhs": sampled_chi,
            "K": np.zeros((sampled_chi.shape[1], 2)),
            "norm": 1.0,
            "m": np.array([0.5, 0.5]),
        }

    monkeypatch.setattr(phoenix.maxent, "Preprocess", fake_preprocess)
    monkeypatch.setattr(
        phoenix.maxent,
        "MaxEnt",
        lambda pre, **kwargs: np.array([0.4, 0.6]),
    )

    results = phoenix.perform_maxent(
        chi,
        (np.array([-1.0, 1.0]), np.ones(2)),
        {"dt": 0.1, "beta": 0.4, "L": 4},
        append=append,
        alpha_arr=np.array([1.0, 10.0]),
        bs=3,
        op_type="boson",
        sym=False,
        rng=np.random.default_rng(19),
    )

    assert len(seen) == 3
    for sampled_chi, sampled_append in seen:
        np.testing.assert_array_equal(
            sampled_append,
            10.0 * sampled_chi[:, :1] + 1.0,
        )
    assert results["s"].shape == (3, 2)


def test_phoenix_main_loads_paired_bundle_without_beta_or_dt(
    tmp_path,
    monkeypatch,
):
    data_path = tmp_path / "data"
    output_path = tmp_path / "output"
    data_path.mkdir()
    bundle_path = data_path / "g_paired.npz"
    _save_bundle(
        bundle_path,
        np.arange(20.0).reshape(5, 4) + 1.0,
        np.array([2.0, 1.0, 3.0, 2.0, 4.0]),
    )
    captured = {}

    def fake_perform_maxent(chi, omega_grid, metadata, **kwargs):
        captured["chi"] = chi.copy()
        captured["metadata"] = dict(metadata)
        return {
            "A": np.array([0.4, 0.6]),
            "s": np.array([[1.0, 2.0]]),
        }

    monkeypatch.setattr(phoenix, "perform_maxent", fake_perform_maxent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_maxent_phoenix.py",
            "--data_path",
            str(data_path),
            "--data_file",
            bundle_path.name,
            "--omega_max",
            "4",
            "--n_omega",
            "2",
            "--bs",
            "24",
            "--sym",
            "--rnd_seed",
            "17",
            "--output_path",
            str(output_path),
        ],
    )

    phoenix.main()

    assert captured["chi"].shape == (24, 4)
    assert captured["metadata"]["dt"] == pytest.approx(0.1)
    assert captured["metadata"]["beta"] == pytest.approx(0.4)
    assert captured["metadata"]["source_nbin"] == 5
    assert captured["metadata"]["maxent_nrow"] == 24
    assert captured["metadata"]["covariance_bootstrap_samples"] == 24
    assert captured["metadata"]["spectrum_bootstrap_samples"] == 24
    saved_metadata = np.load(
        output_path / "metadata.npy",
        allow_pickle=True,
    ).item()
    assert saved_metadata["sign_reweighting"] == (
        "paired_bootstrap_ratio_of_sums"
    )
    assert saved_metadata["bootstrap_seed"] == 17


def test_anneal_reuses_phoenix_paired_preparation_for_each_temperature(
    tmp_path,
    monkeypatch,
):
    base = tmp_path / "runs"
    high = base / "T_high"
    low = base / "T_low"
    high.mkdir(parents=True)
    low.mkdir()
    sign = np.array([2.0, 1.0, 3.0, 2.0])
    _save_bundle(
        high / "corr_paired.npz",
        np.arange(16.0).reshape(4, 4) + 1.0,
        sign,
    )
    _save_bundle(
        low / "corr_paired.npz",
        np.arange(20.0).reshape(4, 5) + 2.0,
        sign,
    )
    calls = []

    def fake_perform_maxent(chi, omega_grid, metadata, **kwargs):
        calls.append(
            {
                "shape": chi.shape,
                "metadata": dict(metadata),
                "model": kwargs["anneal_arr"],
            }
        )
        return {
            "A": np.array([0.35, 0.65]),
            "s": np.array([[1.0, 2.0]]),
        }

    monkeypatch.setattr(
        anneal.run_maxent_phoenix,
        "perform_maxent",
        fake_perform_maxent,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_maxent_anneal.py",
            "--base",
            str(base),
            "--items",
            "T_high,2.5",
            "T_low,2.0",
            "--data_file",
            "corr_paired.npz",
            "--omega_max",
            "4",
            "--n_omega",
            "2",
            "--bs",
            "18",
            "--sym",
            "--rnd_seed",
            "23",
        ],
    )

    anneal.main()

    assert [call["shape"] for call in calls] == [(18, 4), (18, 5)]
    assert calls[0]["metadata"]["beta"] == pytest.approx(0.4)
    assert calls[1]["metadata"]["beta"] == pytest.approx(0.5)
    assert calls[0]["metadata"]["anneal_index"] == 0
    assert calls[1]["metadata"]["anneal_index"] == 1
    assert calls[0]["metadata"]["covariance_bootstrap_samples"] == 18
    assert calls[1]["metadata"]["covariance_bootstrap_samples"] == 18
    assert calls[0]["metadata"]["spectrum_bootstrap_samples"] == 18
    assert calls[1]["metadata"]["spectrum_bootstrap_samples"] == 18
    assert calls[0]["model"] is None
    np.testing.assert_array_equal(calls[1]["model"], np.array([0.35, 0.65]))
