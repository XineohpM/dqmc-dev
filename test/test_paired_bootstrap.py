from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import paired_bootstrap as pb


def _valid_inputs():
    sign = np.array([4.0, 2.0, 3.0])
    numerator = np.array(
        [
            [8.0, 4.0],
            [6.0, 2.0],
            [3.0, 9.0],
        ]
    )
    mean = numerator.sum(axis=0) / sign.sum()
    return {
        "numerator": numerator,
        "sign": sign,
        "n_sample": np.array([10, 10, 10]),
        "tau": np.array([0.0, 0.1]),
        "mean": mean,
        "stderr": np.array([0.2, 0.3]),
        "metadata": {
            "format_version": 1,
            "observable": "JNJN",
            "component": "xx",
            "beta": 0.2,
            "dt": 0.1,
            "normalization": "per_site_q0",
        },
        "source_files": np.array(["bin0.h5", "bin1.h5", "bin2.h5"]),
    }


def test_validate_save_load_roundtrip(tmp_path):
    bundle = pb.validate_paired_bundle(**_valid_inputs())
    path = tmp_path / "JNJN_xx_paired.npz"

    pb.save_paired_bundle(path, bundle)
    loaded = pb.load_paired_bundle(path)

    assert loaded.nbin == 3
    assert loaded.ntau == 2
    np.testing.assert_array_equal(loaded.numerator, bundle.numerator)
    np.testing.assert_array_equal(loaded.sign, bundle.sign)
    np.testing.assert_array_equal(loaded.source_files, bundle.source_files)
    assert loaded.metadata["observable"] == "JNJN"
    assert loaded.metadata["component"] == "xx"


def test_validator_rejects_mean_of_ratios():
    data = _valid_inputs()
    data["mean"] = (data["numerator"] / data["sign"][:, None]).mean(axis=0)

    with pytest.raises(pb.PairedBundleError, match="mean is inconsistent"):
        pb.validate_paired_bundle(**data)


def test_validator_rejects_incomplete_or_misaligned_bins():
    data = _valid_inputs()
    data["n_sample"] = np.array([10, 9, 10])
    with pytest.raises(pb.PairedBundleError, match="completed-bin mask"):
        pb.validate_paired_bundle(**data)

    data = _valid_inputs()
    data["sign"] = data["sign"][:-1]
    with pytest.raises(pb.PairedBundleError, match="sign must have shape"):
        pb.validate_paired_bundle(**data)


def test_ratio_rejects_near_zero_total_sign():
    numerator = np.ones((2, 3))
    sign = np.array([1.0, -1.0])
    with pytest.raises(pb.PairedBundleError, match="too close to zero"):
        pb.ratio_of_sums(numerator, sign)


def test_bootstrap_indices_are_reproducible():
    first = pb.bootstrap_indices(4, 6, seed=17)
    second = pb.bootstrap_indices(4, 6, seed=17)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (6, 4)


def test_bootstrap_ratio_matches_direct_paired_sums():
    numerator = np.array(
        [
            [2.0, 4.0],
            [3.0, 9.0],
            [-1.0, 2.0],
        ]
    )
    sign = np.array([2.0, 1.0, -0.5])
    indices = np.array([[0, 0, 1], [1, 2, 0]], dtype=np.int64)

    result = pb.bootstrap_ratio_of_sums(
        numerator, sign, indices, chunk_size=1
    )
    expected = np.array(
        [
            numerator[[0, 0, 1]].sum(axis=0)
            / sign[[0, 0, 1]].sum(),
            numerator[[1, 2, 0]].sum(axis=0)
            / sign[[1, 2, 0]].sum(),
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_bootstrap_rejects_zero_sign_replicate():
    numerator = np.ones((2, 3))
    sign = np.array([1.0, -1.0])
    indices = np.array([[0, 1]], dtype=np.int64)

    with pytest.raises(pb.PairedBundleError, match="replicate"):
        pb.bootstrap_ratio_of_sums(numerator, sign, indices)


def test_shared_indices_preserve_observable_relationship():
    sign = np.array([3.0, 2.0, 4.0, 1.0])
    numerator = np.arange(12.0).reshape(4, 3)
    indices = pb.bootstrap_indices(4, 20, seed=4)

    first = pb.bootstrap_ratio_of_sums(numerator, sign, indices)
    second = pb.bootstrap_ratio_of_sums(2.0 * numerator, sign, indices)

    np.testing.assert_allclose(second, 2.0 * first)


def test_circular_block_bootstrap_indices_keep_rows_in_blocks():
    indices = pb.bootstrap_indices(
        7,
        8,
        sample_size=7,
        block_size=3,
        seed=12,
    )

    assert indices.shape == (8, 7)
    assert np.all((indices >= 0) & (indices < 7))
    for row in indices:
        np.testing.assert_array_equal(row[1:3], (row[0] + np.arange(1, 3)) % 7)
        np.testing.assert_array_equal(row[4:6], (row[3] + np.arange(1, 3)) % 7)


def test_complex_phase_bootstrap_is_supported():
    sign = np.array([2.0 + 0.2j, 1.5 - 0.1j, 1.0 + 0.3j])
    numerator = np.array(
        [
            [2.0 + 1.0j, 3.0],
            [1.0 - 0.5j, 4.0],
            [0.5 + 0.2j, 2.0],
        ]
    )
    indices = np.array([[0, 1, 2], [2, 2, 0]], dtype=np.int64)

    result = pb.bootstrap_ratio_of_sums(numerator, sign, indices)
    expected = np.array(
        [
            numerator[row].sum(axis=0) / sign[row].sum()
            for row in indices
        ]
    )
    np.testing.assert_allclose(result, expected)
    assert np.iscomplexobj(result)


def test_bootstrap_covariance_rows_match_required_interface():
    estimates = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.2, 1.7, 3.1],
            [0.8, 2.4, 2.9],
            [1.1, 2.2, 3.2],
        ]
    )
    center = np.array([0.95, 2.05, 3.05])

    rows = pb.bootstrap_covariance_rows(estimates, center)

    np.testing.assert_allclose(rows.mean(axis=0), center)
    np.testing.assert_allclose(
        np.cov(rows, rowvar=False) / rows.shape[0],
        np.cov(estimates, rowvar=False),
    )
