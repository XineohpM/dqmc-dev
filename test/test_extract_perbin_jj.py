from pathlib import Path
import os
import subprocess
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extract_perbin_jj.py"
sys.path.insert(0, str(ROOT / "scripts"))

import paired_bootstrap as pb


RUNNER = r"""
from pathlib import Path
import runpy
import sys
import types

root = Path.cwd()
sys.path.insert(0, str(root / "util"))
sys.path.insert(0, str(root / "scripts"))

# util/jqjq.py uses Python 3.9 built-in generic annotations while the test
# environment is Python 3.8. Enable postponed annotations in this test process
# without modifying the production module.
source_path = root / "util" / "jqjq.py"
module = types.ModuleType("jqjq")
module.__file__ = str(source_path)
source = "from __future__ import annotations\n" + source_path.read_text()
exec(compile(source, str(source_path), "exec"), module.__dict__)
sys.modules["jqjq"] = module

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _thermal_shapes(length, nx, ny, bps, b2ps):
    return {
        "j2j2": (length, b2ps, b2ps, ny, nx),
        "jj2": (length, b2ps, bps, ny, nx),
        "j2j": (length, bps, b2ps, ny, nx),
        "jnj2": (length, b2ps, bps, ny, nx),
        "j2jn": (length, bps, b2ps, ny, nx),
        "jjn": (length, bps, bps, ny, nx),
        "jnj": (length, bps, bps, ny, nx),
        "jnjn": (length, bps, bps, ny, nx),
        "jj": (length, bps, bps, ny, nx),
    }


def _write_bin(
    path,
    *,
    ibin,
    sign,
    n_sample,
    mu=-1.0,
    beta=0.2,
    dt=0.1,
    nx=1,
    ny=1,
    bps=4,
    b2ps=12,
    tp=0.2,
):
    length = int(round(beta / dt))
    correlators = {}
    with h5py.File(str(path), "w") as handle:
        metadata = handle.create_group("metadata")
        for key, value in {
            "beta": beta,
            "Nx": nx,
            "Ny": ny,
            "U": 4.0,
            "mu": mu,
            "nflux": 0,
            "t'": tp,
            "bps": bps,
            "b2ps": b2ps,
        }.items():
            metadata.create_dataset(key, data=value)

        params = handle.create_group("params")
        params.create_dataset("dt", data=dt)
        params.create_dataset("L", data=length)
        params.create_dataset("N", data=nx * ny)

        uneqlt = handle.create_group("meas_uneqlt")
        uneqlt.create_dataset("n_sample", data=n_sample)
        uneqlt.create_dataset("sign", data=sign)
        for index, (name, shape) in enumerate(
            _thermal_shapes(length, nx, ny, bps, b2ps).items()
        ):
            values = np.arange(np.prod(shape), dtype=float).reshape(shape)
            values = values + 1.0 + 0.1 * index + 0.01 * ibin
            correlators[name] = values
            uneqlt.create_dataset(name, data=values.reshape(-1))
    return correlators


def _run_extractor(input_path, output_path, *extra_args):
    env = dict(os.environ)
    mpl_cache = output_path.parent / "mpl-cache"
    mpl_cache.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_cache)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            RUNNER,
            str(SCRIPT),
            "--path",
            str(input_path),
            "--outdir",
            str(output_path),
            *extra_args,
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
    )


def _electrical_xx_raw(jj, tp):
    dx = np.array([1.0, 0.0, 1.0, -1.0])
    hopping = np.array([1.0, 1.0, tp, tp])
    factors = hopping * dx
    q0 = jj.sum(axis=(-1, -2))
    return -np.einsum("i,j,tij->t", factors, factors, q0)


def test_all_observables_components_and_completed_mask(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()

    completed_jj = []
    for ibin, (n_sample, sign) in enumerate(
        ((10, 8.0), (10, 0.0), (10, 7.0), (5, 4.0))
    ):
        correlators = _write_bin(
            input_path / "bin_{}.h5".format(ibin),
            ibin=ibin,
            sign=sign,
            n_sample=n_sample,
        )
        if n_sample == 10:
            completed_jj.append(correlators["jj"])

    result = _run_extractor(
        input_path,
        output_path,
        "--output_type",
        "all",
        "--component",
        "all",
    )
    assert result.returncode == 0, result.stderr
    assert "Processed 3 completed bins from 4 files" in result.stdout
    assert "Saved 16 paired bundles" in result.stdout

    expected_names = sorted(
        "{}_{}_paired.npz".format(observable, component)
        for observable in ("JNJN", "JQJQ", "JQJN", "JNJQ")
        for component in ("xx", "yy", "xy", "yx")
    )
    output_files = sorted(path.name for path in output_path.iterdir())
    assert output_files == expected_names
    assert not list(output_path.glob("*.npy"))

    for name in expected_names:
        bundle = pb.load_paired_bundle(output_path / name)
        assert bundle.nbin == 3
        assert bundle.ntau == 2
        np.testing.assert_array_equal(bundle.sign, np.array([8.0, 0.0, 7.0]))
        np.testing.assert_array_equal(
            bundle.source_files,
            np.array(["bin_0.h5", "bin_1.h5", "bin_2.h5"]),
        )
        assert bundle.metadata["normalization"] == "per_site_q0"

    jnjn_xx = pb.load_paired_bundle(output_path / "JNJN_xx_paired.npz")
    expected_numerator = np.stack(
        [_electrical_xx_raw(jj, tp=0.2) for jj in completed_jj]
    )
    np.testing.assert_allclose(jnjn_xx.numerator, expected_numerator)
    np.testing.assert_allclose(
        jnjn_xx.mean,
        expected_numerator.sum(axis=0) / np.array([8.0, 0.0, 7.0]).sum(),
    )


def test_rejects_metadata_mismatch(tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    input_path.mkdir()
    _write_bin(
        input_path / "bin_0.h5",
        ibin=0,
        sign=8.0,
        n_sample=10,
        mu=-1.0,
    )
    _write_bin(
        input_path / "bin_1.h5",
        ibin=1,
        sign=7.0,
        n_sample=10,
        mu=-0.5,
    )

    result = _run_extractor(
        input_path,
        output_path,
        "--output_type",
        "electric",
        "--component",
        "xx",
    )

    assert result.returncode != 0
    assert "Inconsistent mu" in result.stderr
    assert not list(output_path.glob("*"))
