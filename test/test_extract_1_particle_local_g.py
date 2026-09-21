from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_1_particle_local_g.py"
sys.path.insert(0, str(ROOT / "scripts"))

import paired_bootstrap as pb


def _write_bin(
    path,
    *,
    gu_local,
    gd_local,
    sign,
    n_sample,
    beta=0.3,
    dt=0.1,
    nx=2,
    ny=1,
    interaction=4.0,
    mu=-1.0,
):
    gu_local = np.asarray(gu_local)
    gd_local = np.asarray(gd_local)
    length = int(round(beta / dt))
    nsite = nx * ny
    assert gu_local.shape == (length,)
    assert gd_local.shape == (length,)

    dtype = np.result_type(gu_local.dtype, gd_local.dtype)
    gu = np.zeros((length, nsite), dtype=dtype)
    gd = np.zeros((length, nsite), dtype=dtype)
    gu[:, 0] = gu_local
    gd[:, 0] = gd_local
    if nsite > 1:
        gu[:, 1:] = -7.0
        gd[:, 1:] = 11.0
    g = 0.5 * (gu + gd)

    with h5py.File(str(path), "w") as handle:
        uneqlt = handle.create_group("meas_uneqlt")
        uneqlt.create_dataset("gt0_u", data=gu.reshape(-1))
        uneqlt.create_dataset("gt0_d", data=gd.reshape(-1))
        uneqlt.create_dataset("gt0", data=g.reshape(-1))
        uneqlt.create_dataset("sign", data=sign)
        uneqlt.create_dataset("n_sample", data=int(n_sample))

        metadata = handle.create_group("metadata")
        metadata.create_dataset("beta", data=float(beta))
        metadata.create_dataset("Nx", data=int(nx))
        metadata.create_dataset("Ny", data=int(ny))
        metadata.create_dataset("U", data=float(interaction))
        metadata.create_dataset("mu", data=float(mu))

        params = handle.create_group("params")
        params.create_dataset("dt", data=float(dt))


def _run_extractor(input_path, output_path):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--path",
            str(input_path),
            "--output_path",
            str(output_path),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_outputs_only_validated_paired_bundles_and_drops_incomplete_bin(
    tmp_path,
):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()

    gu_complete = (
        np.array([8.0, 4.0, 2.0]),
        np.array([3.0, 9.0, 6.0]),
    )
    gd_complete = (
        np.array([4.0, 8.0, 6.0]),
        np.array([9.0, 3.0, 12.0]),
    )
    _write_bin(
        input_path / "bin_0.h5",
        gu_local=gu_complete[0],
        gd_local=gd_complete[0],
        sign=4.0,
        n_sample=10,
    )
    _write_bin(
        input_path / "bin_1.h5",
        gu_local=gu_complete[1],
        gd_local=gd_complete[1],
        sign=2.0,
        n_sample=10,
    )
    _write_bin(
        input_path / "bin_2_incomplete.h5",
        gu_local=np.full(3, 1000.0),
        gd_local=np.full(3, -1000.0),
        sign=1.0,
        n_sample=5,
    )

    result = _run_extractor(input_path, output_path)
    assert result.returncode == 0, result.stderr
    assert "Dropped 1 incomplete bins: bin_2_incomplete.h5" in result.stdout

    output_files = sorted(path.name for path in output_path.iterdir())
    assert output_files == [
        "1_particle_local_g_paired.npz",
        "1_particle_local_gd_paired.npz",
        "1_particle_local_gu_paired.npz",
    ]

    bundles = {
        observable: pb.load_paired_bundle(
            output_path / "1_particle_local_{}_paired.npz".format(observable)
        )
        for observable in ("g", "gu", "gd")
    }
    expected_gu = np.stack(gu_complete)
    expected_gd = np.stack(gd_complete)
    expected_g = 0.5 * (expected_gu + expected_gd)
    expected = {"g": expected_g, "gu": expected_gu, "gd": expected_gd}

    for observable, bundle in bundles.items():
        np.testing.assert_array_equal(bundle.numerator, expected[observable])
        np.testing.assert_array_equal(bundle.sign, np.array([4.0, 2.0]))
        np.testing.assert_array_equal(bundle.n_sample, np.array([10.0, 10.0]))
        np.testing.assert_array_equal(
            bundle.source_files,
            np.array(["bin_0.h5", "bin_1.h5"]),
        )
        np.testing.assert_allclose(
            bundle.mean,
            expected[observable].sum(axis=0) / 6.0,
        )
        np.testing.assert_allclose(bundle.tau, np.array([0.0, 0.1, 0.2]))
        assert bundle.metadata["observable"] == observable
        assert bundle.metadata["normalization"] == "local_displacement_zero"
        assert bundle.metadata["Nx"] == 2
        assert bundle.metadata["Ny"] == 1


def test_rejects_dt_mismatch_between_bins(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    _write_bin(
        input_path / "bin_0.h5",
        gu_local=np.array([1.0, 2.0, 3.0]),
        gd_local=np.array([3.0, 2.0, 1.0]),
        sign=5.0,
        n_sample=10,
    )
    _write_bin(
        input_path / "bin_1.h5",
        gu_local=np.array([1.0, 3.0]),
        gd_local=np.array([3.0, 1.0]),
        sign=4.0,
        n_sample=10,
        dt=0.15,
    )

    result = _run_extractor(input_path, output_path)

    assert result.returncode != 0
    assert "Inconsistent dt" in result.stderr
    assert not list(output_path.glob("*"))


def test_rejects_metadata_mismatch_between_bins(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    common = {
        "gu_local": np.array([1.0, 2.0, 3.0]),
        "gd_local": np.array([3.0, 2.0, 1.0]),
        "sign": 5.0,
        "n_sample": 10,
    }
    _write_bin(input_path / "bin_0.h5", mu=-1.0, **common)
    _write_bin(input_path / "bin_1.h5", mu=-0.5, **common)

    result = _run_extractor(input_path, output_path)

    assert result.returncode != 0
    assert "Inconsistent mu" in result.stderr
    assert not list(output_path.glob("*"))


def test_preserves_complex_phase_and_green_function(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()

    gu = np.array(
        [
            [1.0 + 2.0j, 2.0 - 1.0j, 3.0 + 0.5j],
            [2.0 - 0.5j, 1.0 + 1.5j, 4.0 - 2.0j],
            [0.5 + 1.0j, 3.0 + 0.25j, 2.0 + 2.0j],
        ]
    )
    gd = np.array(
        [
            [3.0 - 1.0j, 1.0 + 0.5j, 2.0 + 1.0j],
            [1.0 + 2.0j, 4.0 - 1.0j, 3.0 + 0.25j],
            [2.0 + 0.5j, 2.0 - 2.0j, 1.0 + 1.5j],
        ]
    )
    sign = np.array([1.0 + 0.2j, 0.5 + 0.7j, -0.2 + 0.9j])
    for index in range(3):
        _write_bin(
            input_path / f"bin_{index}.h5",
            gu_local=gu[index],
            gd_local=gd[index],
            sign=sign[index],
            n_sample=10,
        )

    result = _run_extractor(input_path, output_path)
    assert result.returncode == 0, result.stderr

    expected = {"gu": gu, "gd": gd, "g": 0.5 * (gu + gd)}
    for observable, numerator in expected.items():
        bundle = pb.load_paired_bundle(
            output_path / f"1_particle_local_{observable}_paired.npz"
        )
        assert np.iscomplexobj(bundle.numerator)
        assert np.iscomplexobj(bundle.sign)
        assert np.iscomplexobj(bundle.mean)
        np.testing.assert_array_equal(bundle.numerator, numerator)
        np.testing.assert_array_equal(bundle.sign, sign)
        np.testing.assert_allclose(
            bundle.mean,
            numerator.sum(axis=0) / sign.sum(),
        )
        assert not np.iscomplexobj(bundle.stderr)
        assert np.all(bundle.stderr >= 0)
        assert bundle.metadata["dt"] == pytest.approx(0.1)


def test_allows_zero_sign_bin_but_bootstrap_rejects_zero_denominator(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()

    gu = np.array(
        [
            [2.0, 4.0, 6.0],
            [1.0, 3.0, 5.0],
            [4.0, 2.0, 1.0],
        ]
    )
    gd = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [2.0, 4.0, 6.0],
        ]
    )
    sign = np.array([0.0, 2.0, 3.0])
    for index in range(3):
        _write_bin(
            input_path / f"bin_{index}.h5",
            gu_local=gu[index],
            gd_local=gd[index],
            sign=sign[index],
            n_sample=10,
        )

    result = _run_extractor(input_path, output_path)
    assert result.returncode == 0, result.stderr

    bundle = pb.load_paired_bundle(
        output_path / "1_particle_local_g_paired.npz"
    )
    np.testing.assert_array_equal(bundle.sign, sign)
    np.testing.assert_allclose(
        bundle.mean,
        (0.5 * (gu + gd)).sum(axis=0) / sign.sum(),
    )

    zero_denominator_indices = np.zeros((1, 3), dtype=np.int64)
    with pytest.raises(
        pb.PairedBundleError,
        match="bootstrap accumulated sign is too close to zero",
    ):
        pb.bootstrap_ratio_of_sums(
            bundle.numerator,
            bundle.sign,
            zero_denominator_indices,
        )


def test_rejects_zero_jackknife_leave_one_out_denominator(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()

    for index, sign in enumerate((0.0, 2.0)):
        _write_bin(
            input_path / f"bin_{index}.h5",
            gu_local=np.array([1.0, 2.0, 3.0]),
            gd_local=np.array([3.0, 2.0, 1.0]),
            sign=sign,
            n_sample=10,
        )

    result = _run_extractor(input_path, output_path)

    assert result.returncode != 0
    assert "Jackknife leave-one-out accumulated sign/phase" in result.stderr
    assert not list(output_path.glob("*"))
