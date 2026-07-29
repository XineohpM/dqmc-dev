from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_sum_rule
import dos_proxy
import extract_local_moment
import paired_bootstrap as pb
import plot_JNJN
import plot_compressibility_from_n_mu
import resistivity_proxy


def _save_bundle(
    path,
    numerator,
    sign,
    *,
    observable,
    component=None,
    dt=0.1,
    stderr=None,
):
    numerator = np.asarray(numerator)
    sign = np.asarray(sign)
    nbin, length = numerator.shape
    if stderr is None:
        stderr = np.full(length, 0.05)
    metadata = {
        "format_version": pb.FORMAT_VERSION,
        "observable": observable,
        "beta": dt * length,
        "dt": dt,
        "L": length,
        "normalization": "test",
    }
    if component is not None:
        metadata["component"] = component
    bundle = pb.validate_paired_bundle(
        numerator=numerator,
        sign=sign,
        n_sample=np.full(nbin, 10.0),
        tau=np.arange(length, dtype=float) * dt,
        mean=pb.ratio_of_sums(numerator, sign),
        stderr=np.asarray(stderr),
        metadata=metadata,
        source_files=np.asarray(
            [f"bin_{index}.h5" for index in range(nbin)]
        ),
    )
    pb.save_paired_bundle(path, bundle)
    return bundle


def test_dos_proxy_uses_paired_ratio_bootstrap(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    sign = np.array([1.0, 3.0, 2.0, 4.0])
    numerator = np.array(
        [
            [2.0, 3.0, 4.0, 5.0, 4.0, 3.0],
            [9.0, 8.0, 7.0, 6.0, 7.0, 8.0],
            [3.0, 5.0, 8.0, 9.0, 8.0, 5.0],
            [8.0, 9.0, 10.0, 12.0, 10.0, 9.0],
        ]
    )
    bundle = _save_bundle(
        run / "1_particle_local_g_paired.npz",
        numerator,
        sign,
        observable="g",
    )
    output_prefix = tmp_path / "output" / "proxy"
    temperature = 1.0 / float(bundle.metadata["beta"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dos_proxy.py",
            "--path",
            str(tmp_path),
            "--items",
            f"run,,{temperature}",
            "--output_path",
            str(output_prefix),
            "--nboot",
            "30",
            "--seed",
            "19",
        ],
    )

    dos_proxy.main()

    with np.load(
        str(output_prefix) + "_dos_0freq.npz",
        allow_pickle=True,
    ) as result:
        expected_mid = dos_proxy.G_beta_over_2(bundle.mean)[0]
        assert result["G_mid"][0] == pytest.approx(expected_mid)
        indices = pb.bootstrap_indices(bundle.nbin, 30, seed=19)
        estimates = pb.bootstrap_ratio_of_sums(
            bundle.numerator,
            bundle.sign,
            indices,
        )
        values = np.asarray(
            [
                dos_proxy.dos(
                    temperature,
                    dos_proxy.G_beta_over_2(row)[0],
                )
                for row in estimates
            ]
        )
        assert result["dos_stderr"][0] == pytest.approx(
            values.std(ddof=1)
        )
        assert "dos_p16" not in result
        assert "dos_p84" not in result
        assert result["nbin"][0] == bundle.nbin


def test_resistivity_proxy_and_plot_consume_jj_bundle(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    sign = np.array([1.0, 2.0, 3.0, 4.0])
    base_curve = np.array([9.0, 7.0, 5.5, 4.5, 4.0, 4.5, 5.5, 7.0])
    numerator = np.stack(
        [
            sign[0] * base_curve + 0.2,
            sign[1] * base_curve - 0.4,
            sign[2] * base_curve + 0.6,
            sign[3] * base_curve - 0.8,
        ]
    )
    bundle = _save_bundle(
        run / "JNJN_xx_paired.npz",
        numerator,
        sign,
        observable="JNJN",
        component="xx",
        stderr=np.linspace(0.01, 0.08, 8),
    )
    output_prefix = tmp_path / "output" / "rho"
    temperature = 1.0 / float(bundle.metadata["beta"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resistivity_proxy.py",
            "--path",
            str(tmp_path),
            "--items",
            f"run,,{temperature}",
            "--output_path",
            str(output_prefix),
            "--sym",
            "--nboot",
            "24",
            "--seed",
            "31",
            "--deriv_window",
            "2",
        ],
    )

    resistivity_proxy.main()

    center = resistivity_proxy._symmetrize_about_beta_over_2(bundle.mean)
    lambda_mid = resistivity_proxy.Lambda_xx_beta_over_2(center)[0]
    with np.load(str(output_prefix) + "_rho1.npz") as rho1_result:
        assert rho1_result["Lambda_mid"][0] == pytest.approx(lambda_mid)
        assert rho1_result["rho_mean"][0] == pytest.approx(
            resistivity_proxy.rho1(temperature, lambda_mid)
        )
        assert np.isfinite(rho1_result["rho_stderr"][0])
        assert "rho_p16" not in rho1_result
        assert "rho_p84" not in rho1_result
    with np.load(str(output_prefix) + "_rho2.npz") as rho2_result:
        assert np.isfinite(rho2_result["rho_mean"][0])
        assert np.isfinite(rho2_result["rho_stderr"][0])
        assert "rho_p16" not in rho2_result
        assert "rho_p84" not in rho2_result

    figure = tmp_path / "JNJN.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_JNJN.py",
            "--path",
            str(run),
            "--out",
            str(figure),
        ],
    )
    plot_JNJN.main()
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_sum_rule_uses_synchronized_source_file_bootstrap(
    tmp_path,
    monkeypatch,
    capsys,
):
    run = tmp_path / "run"
    run.mkdir()
    sign = np.array([2.0, 1.0, 3.0, 2.0])
    curve = np.array([5.0, 4.0, 3.5, 3.0, 3.5, 4.0])
    numerator = np.stack(
        [
            sign[0] * curve + 0.2,
            sign[1] * curve - 0.1,
            sign[2] * curve + 0.3,
            sign[3] * curve - 0.4,
        ]
    )
    bundle = _save_bundle(
        run / "JNJN_xx_paired.npz",
        numerator,
        sign,
        observable="JNJN",
        component="xx",
    )
    norm4 = 4 * check_sum_rule.half_interval_norm(
        bundle.mean,
        float(bundle.metadata["dt"]),
        float(bundle.metadata["beta"]),
    )

    eqlt_sign = np.array([4.0, 2.0, 3.0, 1.0])
    for index, source_file in enumerate(bundle.source_files):
        kinetic_numerator = -norm4 * eqlt_sign[index]
        g00 = np.zeros((2, 2), dtype=float)
        g00[0, 1] = kinetic_numerator / 8.0
        g00[1, 0] = kinetic_numerator / 8.0
        with h5py.File(run / source_file, "w") as handle:
            metadata = handle.create_group("metadata")
            metadata.create_dataset("Nx", data=2)
            metadata.create_dataset("Ny", data=2)
            metadata.create_dataset("t'", data=0.0)
            metadata.create_dataset(
                "beta",
                data=float(bundle.metadata["beta"]),
            )
            eqlt = handle.create_group("meas_eqlt")
            eqlt.create_dataset("sign", data=eqlt_sign[index])
            eqlt.create_dataset("n_sample", data=10)
            eqlt.create_dataset(
                "g00",
                data=g00.reshape(-1, order="F"),
            )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_sum_rule.py",
            "--path",
            str(tmp_path),
            "--relpath_list",
            "run",
            "--bootstrap",
            "20",
            "--seed",
            "7",
        ],
    )

    check_sum_rule.main()

    output = capsys.readouterr().out
    assert "bootstrap alignment =  paired-by-source_files" in output
    assert "norm of correlator = kinetic energy" in output
    assert "unequal-time mean sign" in output
    assert "equal-time mean sign" in output


def test_local_moment_center_is_ratio_of_sums_with_fluctuating_sign():
    n_sample = np.full(4, 10.0)
    sign = np.array([3.0, -0.5, 2.0, 4.0])
    numerator = np.array([1.2, -0.4, 0.7, 2.1])

    mean, stderr = extract_local_moment._jackknife_stats(
        n_sample,
        sign,
        numerator,
    )

    assert mean == pytest.approx(numerator.sum() / sign.sum())
    assert np.isfinite(stderr)
    assert stderr >= 0


def test_n_mu_center_uses_independent_paired_data(
    tmp_path,
    monkeypatch,
):
    t_dir = tmp_path / "T0.5_beta2_U4"
    mu_values = np.array([-0.1, 0.0, 0.1])
    sign = np.array([3.0, -0.5, 2.0, 4.0])
    perturbation = np.array([0.12, -0.08, 0.05, -0.03])
    data_by_name = {}

    for index, mu in enumerate(mu_values):
        mu_dir = t_dir / f"mu{mu:g}"
        mu_dir.mkdir(parents=True)
        physical_n = 0.8 + 2.5 * mu
        numerator = (
            sign * physical_n
            + (1.0 + index) * perturbation
        )[:, None]
        n_mean = float(numerator.sum() / sign.sum())
        data_by_name[mu_dir.name] = {
            "mu": float(mu),
            "n_mean": n_mean,
            "n_err": 0.01,
            "density_numerator": numerator,
            "sign": sign,
            "n_sample": np.full(4, 10.0),
            "nsite": 1,
            "mu_dir": mu_dir,
        }

    monkeypatch.setattr(
        plot_compressibility_from_n_mu,
        "load_mu_density_bins",
        lambda path: data_by_name[path.name],
    )
    result = plot_compressibility_from_n_mu.compute_kappa_for_T_dir(
        t_dir=t_dir,
        filling=0.8,
        window=3,
        min_points=3,
        range_tol=1e-3,
        nboot=128,
        seed=37,
        bootstrap_block_size=1,
    )

    expected_density = np.array(
        [data_by_name[f"mu{mu:g}"]["n_mean"] for mu in mu_values]
    )
    expected_kappa = np.polyfit(mu_values, expected_density, 1)[0]
    assert result["status"] == "ok"
    assert result["kappa"] == pytest.approx(expected_kappa)
    assert np.isfinite(result["kappa_err"])
