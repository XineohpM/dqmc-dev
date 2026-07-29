from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_warm
import get_n_from_best_mu
import paired_bootstrap


def _write_density_bin(path, n_sample, sign, density, mu=-0.5):
    with h5py.File(path, "w") as h5:
        meas = h5.create_group("meas_eqlt")
        meas.create_dataset("n_sample", data=n_sample)
        meas.create_dataset("sign", data=sign)
        meas.create_dataset("density", data=np.asarray([density]))
        metadata = h5.create_group("metadata")
        metadata.create_dataset("mu", data=mu)


def test_get_n_keeps_zero_sign_bin_and_uses_ratio_of_sums(tmp_path):
    signs = np.array([2.0, 0.0, 3.0, 1.0])
    density_numerators = np.array([4.0, 5.0, 9.0, 2.0])
    for index, (sign, numerator) in enumerate(
        zip(signs, density_numerators)
    ):
        _write_density_bin(
            tmp_path / f"bin_{index}.h5",
            n_sample=10,
            sign=sign,
            density=numerator,
        )

    mu, mean, stderr = get_n_from_best_mu.get_mu_n(
        str(tmp_path) + "/"
    )

    assert mu == pytest.approx(-0.5)
    assert mean == pytest.approx(
        density_numerators.sum() / signs.sum()
    )
    assert np.isfinite(stderr)


def test_get_n_rejects_zero_leave_one_out_denominator(tmp_path):
    for index, sign in enumerate([1.0, -1.0, 1.0]):
        _write_density_bin(
            tmp_path / f"bin_{index}.h5",
            n_sample=10,
            sign=sign,
            density=2.0 + index,
        )

    with pytest.raises(
        ValueError,
        match="leave-one-out accumulated sign/phase",
    ):
        get_n_from_best_mu.get_mu_n(str(tmp_path) + "/")


def test_check_warm_loads_paired_raw_and_prefix_diagnostics(tmp_path):
    numerator = np.array(
        [
            [2.0, 4.0],
            [6.0, 10.0],
            [1.0, 3.0],
            [8.0, 12.0],
        ]
    )
    sign = np.array([0.0, 2.0, -1.0, 3.0])
    n_sample = np.full(4, 10.0)
    metadata = {
        "format_version": paired_bootstrap.FORMAT_VERSION,
        "observable": "JNJN",
        "component": "xx",
        "beta": 0.2,
        "dt": 0.1,
        "L": 2,
        "normalization": "test",
    }
    bundle = paired_bootstrap.validate_paired_bundle(
        numerator=numerator,
        sign=sign,
        n_sample=n_sample,
        tau=np.array([0.0, 0.1]),
        mean=paired_bootstrap.ratio_of_sums(numerator, sign),
        stderr=np.zeros(2),
        metadata=metadata,
        source_files=np.asarray(
            [f"bin_{index}.h5" for index in range(4)]
        ),
    )
    paired_bootstrap.save_paired_bundle(
        tmp_path / "JNJN_xx_paired.npz",
        bundle,
    )

    diagnostics = check_warm.load_optional_paired_diagnostics(
        tmp_path,
        "JNJN_xx_paired.npz",
    )

    np.testing.assert_allclose(
        diagnostics[
            "JNJN_xx_paired__raw_numerator_per_sample"
        ],
        numerator.mean(axis=1) / n_sample,
    )
    np.testing.assert_allclose(
        diagnostics["JNJN_xx_paired__sign_phase_per_sample"],
        sign / n_sample,
    )

    cumulative_numerator = np.cumsum(numerator, axis=0)
    cumulative_sign = np.cumsum(sign)
    expected_prefix = np.full(4, np.nan)
    expected_prefix[1:] = (
        cumulative_numerator[1:].mean(axis=1)
        / cumulative_sign[1:]
    )
    np.testing.assert_allclose(
        diagnostics["JNJN_xx_paired__prefix_ratio_of_sums"],
        expected_prefix,
        equal_nan=True,
    )
