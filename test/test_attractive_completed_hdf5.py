"""Read-only D1-D6 tests for completed attractive Hubbard HDF5 files.

The original files under ``dqmc_data/test/T_0.4`` are never modified.  D1
validates the complete-file schema and provenance, D2 drives the real
``src/data.c`` reader through all simulation files, and D3 checks strict
equal-time measurement identities. D4 checks exact density-channel spin/sign
relations, while D5-D6 use the 100 files as independent statistical bins.
No DQMC sweep is run.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.stats import t as student_t


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = Path("/Users/a9012/Desktop/dqmc_data/test/T_0.4")
PARAMS_FILE = DATA_DIR / "_T0.4_.h5.params"
SIM_FILE_RE = re.compile(r"_T0\.4__(\d+)\.h5$")

N_FILES = 100
N = 64
NX = 8
NY = 8
L = 50
U = -6.0
DT = 0.05
BETA = 2.5
N_SWEEP = 1_050_000
N_SWEEP_WARM = 50_000
N_SWEEP_MEAS = 1_000_000
PERIOD_EQLT = 5
PERIOD_UNEQLT = 2
N_SAMPLE_EQLT = 10_000_000
N_SAMPLE_UNEQLT = 500_000

# The observables below were accumulated independently over 10^7 samples.
# A 2e-11 relative allowance remains more than two orders of magnitude below
# the standard worst-case gamma_n roundoff bound for that many double-precision
# additions, while accommodating different but algebraically equivalent
# accumulation orders.
EXACT_RTOL = 2e-11
EXACT_ATOL = 1e-10
BOOTSTRAP_SEED = 20260829
BOOTSTRAP_RESAMPLES = 10_000


def _run_checked(command, *, env=None):
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, env=env
    )
    assert result.returncode == 0, (
        f"command failed with exit code {result.returncode}:\n"
        f"{' '.join(map(str, command))}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def _clang():
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("No C compiler is available")
    return compiler


def _hdf5_compile_flags():
    pkg_config = shutil.which("pkg-config")
    if pkg_config is None:
        pytest.skip("pkg-config is unavailable for the HDF5 C test")
    result = subprocess.run(
        [pkg_config, "--cflags", "--libs", "hdf5_hl"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"HDF5 C/HL development files are unavailable: {result.stderr}")
    return shlex.split(result.stdout)


def _decode(dataset):
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@pytest.fixture(scope="module")
def completed_files():
    assert DATA_DIR.is_dir(), f"completed-data directory is missing: {DATA_DIR}"
    assert PARAMS_FILE.is_file(), f"shared params file is missing: {PARAMS_FILE}"

    indexed = []
    for path in DATA_DIR.glob("_T0.4__*.h5"):
        match = SIM_FILE_RE.fullmatch(path.name)
        assert match is not None, f"unexpected simulation filename: {path.name}"
        indexed.append((int(match.group(1)), path))
    indexed.sort()

    assert [index for index, _ in indexed] == list(range(N_FILES))
    files = [path for _, path in indexed]
    assert len(files) == N_FILES
    log_indices = sorted(
        int(match.group(1))
        for path in DATA_DIR.glob("_T0.4__*.h5.log")
        if (match := re.fullmatch(r"_T0\.4__(\d+)\.h5\.log", path.name))
    )
    assert log_indices == list(range(N_FILES))
    return files


def _assert_shape_dtype(group, name, shape, dtype):
    dataset = group[name]
    assert dataset.shape == shape, f"{dataset.name}: {dataset.shape} != {shape}"
    assert dataset.dtype == np.dtype(dtype), (
        f"{dataset.name}: {dataset.dtype} != {np.dtype(dtype)}"
    )


def _assert_all_numeric_datasets_finite(group, file_name):
    failures = []

    def check(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.dtype.kind not in "fc":
            return
        if not np.all(np.isfinite(obj[...])):
            failures.append(name)

    group.visititems(check)
    assert failures == [], f"{file_name}: non-finite datasets: {failures}"


def _independent_density_hs_parameters(U_i, dt, map_i):
    x = 0.5 * dt * np.abs(U_i)
    lambdas = 2.0 * np.arcsinh(np.sqrt(0.5 * np.expm1(x)))
    positive = np.exp(lambdas)[map_i]
    exp_lambda = np.stack((positive**-1, positive))
    delta = np.stack((positive**2 - 1.0, positive**-2 - 1.0))
    return exp_lambda, delta


def test_d1_shared_parameters_schema_and_attractive_values(completed_files):
    """D1: the common 8x8 square attractive parameter file is consistent."""
    with h5py.File(PARAMS_FILE, "r") as h5:
        assert set(("metadata", "params")) <= set(h5)
        metadata = h5["metadata"]
        params = h5["params"]

        assert _decode(metadata["geometry"]) == "square"
        assert _decode(metadata["model"]) == "Hubbard"
        assert _decode(metadata["hs_channel"]) == "density"
        assert metadata["Norb"][()] == 1
        assert metadata["Nx"][()] == NX
        assert metadata["Ny"][()] == NY
        assert metadata["U"][()] == pytest.approx(U)
        assert metadata["mu"][()] == pytest.approx(0.0)
        assert metadata["h"][()] == pytest.approx(0.0)
        assert metadata["beta"][()] == pytest.approx(BETA)
        assert metadata["geometry"].shape == ()

        expected_scalars = {
            "N": N,
            "Nx": NX,
            "Ny": NY,
            "L": L,
            "hs_channel": 1,
            "n_delay": 16,
            "n_matmul": 5,
            "n_sweep": N_SWEEP,
            "n_sweep_warm": N_SWEEP_WARM,
            "n_sweep_meas": N_SWEEP_MEAS,
            "period_eqlt": PERIOD_EQLT,
            "period_uneqlt": PERIOD_UNEQLT,
            "num_i": 1,
            "num_ij": N,
        }
        for name, expected in expected_scalars.items():
            assert params[name].shape == ()
            assert int(params[name][()]) == expected
        assert params["dt"][()] == pytest.approx(DT)
        assert params["L"][()] * params["dt"][()] == pytest.approx(BETA)

        _assert_shape_dtype(params, "U", (1,), np.float64)
        _assert_shape_dtype(params, "map_i", (N,), np.int32)
        _assert_shape_dtype(params, "map_ij", (N, N), np.int32)
        _assert_shape_dtype(params, "exp_lambda", (2, N), np.float64)
        _assert_shape_dtype(params, "del", (2, N), np.float64)
        for name in (
            "Ku",
            "Kd",
            "exp_Ku",
            "exp_Kd",
            "inv_exp_Ku",
            "inv_exp_Kd",
            "exp_halfKu",
            "exp_halfKd",
            "inv_exp_halfKu",
            "inv_exp_halfKd",
        ):
            _assert_shape_dtype(params, name, (N, N), np.float64)

        U_i = params["U"][...]
        map_i = params["map_i"][...]
        assert np.array_equal(U_i, np.array([U], dtype=np.float64))
        assert np.array_equal(map_i, np.zeros(N, dtype=np.int32))
        expected_exp_lambda, expected_delta = _independent_density_hs_parameters(
            U_i, DT, map_i
        )
        np.testing.assert_allclose(
            params["exp_lambda"][...], expected_exp_lambda, rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            params["del"][...], expected_delta, rtol=1e-12, atol=1e-14
        )

        for up, down in (
            ("Ku", "Kd"),
            ("exp_Ku", "exp_Kd"),
            ("inv_exp_Ku", "inv_exp_Kd"),
            ("exp_halfKu", "exp_halfKd"),
            ("inv_exp_halfKu", "inv_exp_halfKd"),
        ):
            np.testing.assert_array_equal(params[up][...], params[down][...])

        identity = np.eye(N)
        for forward, inverse in (
            ("exp_Ku", "inv_exp_Ku"),
            ("exp_Kd", "inv_exp_Kd"),
            ("exp_halfKu", "inv_exp_halfKu"),
            ("exp_halfKd", "inv_exp_halfKd"),
        ):
            product = params[forward][...] @ params[inverse][...]
            np.testing.assert_allclose(product, identity, rtol=1e-12, atol=1e-12)

        _assert_all_numeric_datasets_finite(h5, PARAMS_FILE.name)


def test_d1_all_simulations_are_complete_well_formed_and_unique(completed_files):
    """D1: all 100 independent chains are complete, finite, and nonduplicated."""
    init_rng_states = set()
    final_rng_states = set()
    final_hs_states = set()

    required_eqlt = {
        "n_sample": (),
        "sign": (),
        "density": (1,),
        "density_u": (1,),
        "density_d": (1,),
        "double_occ": (1,),
        "g00": (N,),
        "g00_u": (N,),
        "g00_d": (N,),
        "nn": (N,),
        "xx": (N,),
        "zz": (N,),
        "pair_sw": (N,),
        "kk": (256,),
        "kn": (128,),
        "kv": (128,),
        "vv": (N,),
        "vn": (N,),
    }
    required_uneqlt = {
        "n_sample": (),
        "sign": (),
        "gt0": (N * L,),
        "gt0_u": (N * L,),
        "gt0_d": (N * L,),
        "nn": (N * L,),
        "xx": (N * L,),
        "zz": (N * L,),
        "pair_sw": (N * L,),
        "pair_bb": (256 * L,),
        "jj": (256 * L,),
        "jsjs": (256 * L,),
        "kk": (256 * L,),
        "ksks": (256 * L,),
        "kn": (128 * L,),
        "kv": (128 * L,),
        "vv": (N * L,),
        "vn": (N * L,),
    }

    for path in completed_files:
        log_path = Path(f"{path}.log")
        assert log_path.is_file(), f"missing log: {log_path}"
        log = log_path.read_text(encoding="utf-8")
        assert f"{N_SWEEP}/{N_SWEEP} sweeps completed" in log
        assert "sim_data_save() succeeded" in log

        with h5py.File(path, "r") as h5:
            assert isinstance(h5.get("params", getlink=True), h5py.ExternalLink)
            assert isinstance(h5.get("metadata", getlink=True), h5py.ExternalLink)
            assert h5.get("params", getlink=True).filename == PARAMS_FILE.name
            assert h5.get("metadata", getlink=True).filename == PARAMS_FILE.name
            assert _decode(h5["params_file"]) == PARAMS_FILE.name

            assert h5["state/sweep"][()] == N_SWEEP
            assert h5["state/partial_write"][()] == 0
            assert h5["params/n_sweep"][()] == N_SWEEP
            assert h5["params/hs_channel"][()] == 1
            _assert_shape_dtype(h5["state"], "init_rng", (17,), np.uint64)
            _assert_shape_dtype(h5["state"], "rng", (17,), np.uint64)
            _assert_shape_dtype(h5["state"], "hs", (L, N), np.int32)
            _assert_shape_dtype(h5["state"], "sweep", (), np.int32)
            assert np.all(np.isin(h5["state/hs"][...], (0, 1)))

            for name, shape in required_eqlt.items():
                assert h5[f"meas_eqlt/{name}"].shape == shape
                expected_dtype = np.int32 if name == "n_sample" else np.float64
                assert h5[f"meas_eqlt/{name}"].dtype == np.dtype(expected_dtype)
            for name, shape in required_uneqlt.items():
                assert h5[f"meas_uneqlt/{name}"].shape == shape
                expected_dtype = np.int32 if name == "n_sample" else np.float64
                assert h5[f"meas_uneqlt/{name}"].dtype == np.dtype(expected_dtype)

            assert h5["meas_eqlt/n_sample"][()] == N_SAMPLE_EQLT
            assert h5["meas_uneqlt/n_sample"][()] == N_SAMPLE_UNEQLT
            assert N_SAMPLE_EQLT == N_SWEEP_MEAS * (L // PERIOD_EQLT)
            assert N_SAMPLE_UNEQLT == N_SWEEP_MEAS // PERIOD_UNEQLT

            _assert_all_numeric_datasets_finite(h5["state"], path.name)
            _assert_all_numeric_datasets_finite(h5["meas_eqlt"], path.name)
            _assert_all_numeric_datasets_finite(h5["meas_uneqlt"], path.name)

            init_rng_states.add(h5["state/init_rng"][...].tobytes())
            final_rng_states.add(h5["state/rng"][...].tobytes())
            final_hs_states.add(h5["state/hs"][...].tobytes())

    assert len(init_rng_states) == N_FILES
    assert len(final_rng_states) == N_FILES
    assert len(final_hs_states) == N_FILES


def test_d1_logged_commits_are_compatible_with_current_relevant_code(
    completed_files,
):
    """D1: the logged commit mismatch contains no relevant production diff."""
    executable_commits = set()
    generator_commits = set()
    executable_pattern = re.compile(r"executable commit id ([0-9a-f]+)")
    generator_pattern = re.compile(r"hdf5 generation script commit id ([0-9a-f]+)")

    for path in completed_files:
        log = Path(f"{path}.log").read_text(encoding="utf-8")
        executable_commits.update(executable_pattern.findall(log))
        generator_commits.update(generator_pattern.findall(log))

    assert executable_commits == {"c48e98f"}
    assert generator_commits == {"c35cce3"}
    executable_commit = next(iter(executable_commits))
    generator_commit = next(iter(generator_commits))
    relevant_paths = [
        "src",
        "util/gen_1band_unified_hub.py",
        "util/gen_util_shared.py",
    ]

    git = ["git", "-C", str(REPO_ROOT)]
    _run_checked(
        [*git, "merge-base", "--is-ancestor", executable_commit, generator_commit]
    )
    _run_checked(
        [
            *git,
            "diff",
            "--quiet",
            f"{executable_commit}..{generator_commit}",
            "--",
            *relevant_paths,
        ]
    )
    _run_checked(
        [
            *git,
            "diff",
            "--quiet",
            f"{executable_commit}..HEAD",
            "--",
            *relevant_paths,
        ]
    )
    _run_checked([*git, "diff", "--quiet", "--", *relevant_paths])


def _write_native_allocation_stub(include_dir):
    include_dir.mkdir()
    (include_dir / "xmmintrin.h").write_text(
        r"""
#pragma once
#include <stddef.h>
#include <stdlib.h>
static inline void *_mm_malloc(size_t size, size_t alignment)
{
    void *pointer = NULL;
    if (posix_memalign(&pointer, alignment, size) != 0) return NULL;
    return pointer;
}
static inline void _mm_free(void *pointer)
{
    free(pointer);
}
""".lstrip(),
        encoding="utf-8",
    )


def _unused_greens_workspace_stubs_source():
    return r"""
int get_lwork_eq_g(const int N)
{
    (void)N;
    return 0;
}

int get_lwork_ue_g(const int N, const int L)
{
    (void)N;
    (void)L;
    return 0;
}
""".lstrip()


def _d2_reader_harness_source():
    return r"""
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "data.h"

static uint64_t fnv1a(const void *data, size_t size)
{
    const unsigned char *bytes = data;
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int check_int(
        const char *file, const char *name, int actual, int expected)
{
    if (actual == expected) return 0;
    fprintf(stderr, "%s: %s=%d expected=%d\n", file, name, actual, expected);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) return 64;
    if (set_num_h5t() != 0) return 65;

    for (int file_index = 1; file_index < argc; ++file_index) {
        struct sim_data sim = {0};
        sim.file = argv[file_index];
        const int status = sim_data_read_alloc(&sim);
        if (status != 0) {
            fprintf(stderr, "%s: sim_data_read_alloc=%d\n", sim.file, status);
            return 66;
        }

        int failed = 0;
        failed |= check_int(sim.file, "N", sim.p.N, 64);
        failed |= check_int(sim.file, "Nx", sim.p.Nx, 8);
        failed |= check_int(sim.file, "Ny", sim.p.Ny, 8);
        failed |= check_int(sim.file, "L", sim.p.L, 50);
        failed |= check_int(sim.file, "num_i", sim.p.num_i, 1);
        failed |= check_int(sim.file, "num_ij", sim.p.num_ij, 64);
        failed |= check_int(sim.file, "hs_channel", sim.p.hs_channel, 1);
        failed |= check_int(sim.file, "n_sweep", sim.p.n_sweep, 1050000);
        failed |= check_int(sim.file, "sweep", sim.s.sweep, 1050000);
        failed |= check_int(sim.file, "eqlt n_sample", sim.m_eq.n_sample, 10000000);
        failed |= check_int(sim.file, "uneqlt n_sample", sim.m_ue.n_sample, 500000);
        if (failed) {
            sim_data_free(&sim);
            return 67;
        }

        printf(
            "D2 %d"
            " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
            " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
            " %016" PRIx64 " %016" PRIx64 " %016" PRIx64
            " %016" PRIx64 " %016" PRIx64 "\n",
            file_index - 1,
            fnv1a(sim.s.rng, 17*sizeof(sim.s.rng[0])),
            fnv1a(sim.s.hs, sim.p.N*sim.p.L*sizeof(sim.s.hs[0])),
            fnv1a(sim.p.exp_lambda, 2*sim.p.N*sizeof(sim.p.exp_lambda[0])),
            fnv1a(sim.p.del, 2*sim.p.N*sizeof(sim.p.del[0])),
            fnv1a(sim.m_eq.density, sim.p.num_i*sizeof(sim.m_eq.density[0])),
            fnv1a(sim.m_eq.g00, sim.p.num_ij*sizeof(sim.m_eq.g00[0])),
            fnv1a(sim.m_eq.nn, sim.p.num_ij*sizeof(sim.m_eq.nn[0])),
            fnv1a(sim.m_eq.pair_sw, sim.p.num_ij*sizeof(sim.m_eq.pair_sw[0])),
            fnv1a(sim.m_ue.gt0, sim.p.num_ij*sim.p.L*sizeof(sim.m_ue.gt0[0])),
            fnv1a(sim.m_ue.nn, sim.p.num_ij*sim.p.L*sizeof(sim.m_ue.nn[0])),
            fnv1a(sim.m_ue.pair_sw,
                  sim.p.num_ij*sim.p.L*sizeof(sim.m_ue.pair_sw[0])));

        sim_data_free(&sim);
    }
    return 0;
}
""".lstrip()


@pytest.fixture(scope="module")
def d2_reader_harness(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("attractive_completed_hdf5_reader")
    include_dir = build_dir / "test_includes"
    _write_native_allocation_stub(include_dir)
    harness_source = build_dir / "completed_hdf5_reader.c"
    greens_stubs_source = build_dir / "unused_greens_workspace_stubs.c"
    executable = build_dir / "completed_hdf5_reader"
    harness_source.write_text(_d2_reader_harness_source(), encoding="utf-8")
    greens_stubs_source.write_text(
        _unused_greens_workspace_stubs_source(), encoding="utf-8"
    )

    _run_checked(
        [
            _clang(),
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
            '-DGIT_ID="attractive-D2-test"',
            f"-I{include_dir}",
            f"-I{SRC_DIR}",
            str(harness_source),
            str(SRC_DIR / "data.c"),
            str(greens_stubs_source),
            *_hdf5_compile_flags(),
            "-lm",
            "-o",
            str(executable),
        ]
    )
    return executable


def _fnv1a(array, dtype):
    data = np.ascontiguousarray(array, dtype=dtype).view(np.uint8).ravel()
    value = 14695981039346656037
    for byte in data:
        value ^= int(byte)
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def _expected_d2_hashes(path):
    with h5py.File(path, "r") as h5:
        return [
            _fnv1a(h5["state/rng"][...], np.uint64),
            _fnv1a(h5["state/hs"][...], np.int32),
            _fnv1a(h5["params/exp_lambda"][...], np.float64),
            _fnv1a(h5["params/del"][...], np.float64),
            _fnv1a(h5["meas_eqlt/density"][...], np.float64),
            _fnv1a(h5["meas_eqlt/g00"][...], np.float64),
            _fnv1a(h5["meas_eqlt/nn"][...], np.float64),
            _fnv1a(h5["meas_eqlt/pair_sw"][...], np.float64),
            _fnv1a(h5["meas_uneqlt/gt0"][...], np.float64),
            _fnv1a(h5["meas_uneqlt/nn"][...], np.float64),
            _fnv1a(h5["meas_uneqlt/pair_sw"][...], np.float64),
        ]


def test_d2_real_data_c_reads_all_completed_files(
    completed_files, d2_reader_harness
):
    """D2: real data.c reads every completed state and measurement array."""
    result = _run_checked([str(d2_reader_harness), *map(str, completed_files)])
    assert "defaulting hs_channel=0" not in result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == N_FILES

    for expected_index, (path, line) in enumerate(zip(completed_files, lines)):
        fields = line.split()
        assert fields[:2] == ["D2", str(expected_index)]
        assert fields[2:] == _expected_d2_hashes(path)


def _assert_scaled_identity(path, label, actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.maximum(np.abs(actual), np.abs(expected))
    tolerance = EXACT_ATOL + EXACT_RTOL * scale
    residual = np.abs(actual - expected)
    assert np.all(residual <= tolerance), (
        f"{path.name} {label}: max residual={residual.max():.17g}, "
        f"max tolerance={tolerance.max():.17g}"
    )


def test_d3_equal_time_accumulators_satisfy_strict_identities(completed_files):
    """D3: all files satisfy onsite and component measurement identities."""
    with h5py.File(PARAMS_FILE, "r") as params_h5:
        map_ij = params_h5["params/map_ij"][...]
        diagonal_slots = np.unique(np.diag(map_ij))
        np.testing.assert_array_equal(diagonal_slots, np.array([0], dtype=np.int32))
        onsite = int(diagonal_slots[0])

    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            sign = float(eq["sign"][()])
            density = eq["density"][...]
            density_u = eq["density_u"][...]
            density_d = eq["density_d"][...]
            double_occ = eq["double_occ"][...]
            g00 = eq["g00"][...]
            g00_u = eq["g00_u"][...]
            g00_d = eq["g00_d"][...]
            nn = eq["nn"][...]
            xx = eq["xx"][...]
            zz = eq["zz"][...]
            pair_sw = eq["pair_sw"][...]

            _assert_scaled_identity(
                path, "density components", density, density_u + density_d
            )
            _assert_scaled_identity(
                path, "g00 spin average", g00, 0.5 * (g00_u + g00_d)
            )
            _assert_scaled_identity(
                path,
                "nn onsite",
                nn[onsite],
                density[0] + 2.0 * double_occ[0],
            )
            local_spin = 0.25 * (density[0] - 2.0 * double_occ[0])
            _assert_scaled_identity(path, "xx onsite", xx[onsite], local_spin)
            _assert_scaled_identity(path, "zz onsite", zz[onsite], local_spin)
            _assert_scaled_identity(
                path,
                "pair onsite",
                pair_sw[onsite],
                sign - density[0] + double_occ[0],
            )
            _assert_scaled_identity(
                path,
                "density-g00 onsite",
                density[0] + 2.0 * g00[onsite],
                2.0 * sign,
            )


def test_d4_density_channel_spin_components_and_sign_are_exact(completed_files):
    """D4: balanced density-channel files have identical spin sectors/sign 1."""
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            ue = h5["meas_uneqlt"]

            np.testing.assert_array_equal(eq["density_u"][...], eq["density_d"][...])
            np.testing.assert_array_equal(eq["g00_u"][...], eq["g00_d"][...])
            np.testing.assert_array_equal(ue["gt0_u"][...], ue["gt0_d"][...])
            np.testing.assert_array_equal(eq["xx"][...], eq["zz"][...])

            assert eq["sign"][()] == float(eq["n_sample"][()]), path.name
            assert ue["sign"][()] == float(ue["n_sample"][()]), path.name


def _chain_level_half_filling_estimators(completed_files):
    estimators = []
    diagnostics = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            sign = float(eq["sign"][()])
            density = float(eq["density"][0]) / sign
            density_u = float(eq["density_u"][0]) / sign
            density_d = float(eq["density_d"][0]) / sign
            g00_onsite = float(eq["g00"][0]) / sign
            double_occ = float(eq["double_occ"][0]) / sign
            estimators.append((density, density_u, density_d, g00_onsite))
            diagnostics.append((double_occ, density - 2.0 * double_occ))
    return np.asarray(estimators), np.asarray(diagnostics)


def _bootstrap_mean_percentile_interval(
    samples,
    *,
    confidence=0.99,
    n_resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED,
):
    samples = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_chains = samples.shape[0]
    bootstrap_means = np.empty((n_resamples, samples.shape[1]), dtype=np.float64)
    chunk_size = 500
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_chains, size=(stop - start, n_chains))
        bootstrap_means[start:stop] = samples[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return np.quantile(bootstrap_means, (alpha / 2.0, 1.0 - alpha / 2.0), axis=0)


def test_d5_half_filling_particle_hole_targets(completed_files):
    """D5: chain-level estimates agree with the exact half-filling targets."""
    estimators, diagnostics = _chain_level_half_filling_estimators(completed_files)
    labels = ("density", "density_u", "density_d", "g00 onsite")
    targets = np.array((1.0, 0.5, 0.5, 0.5), dtype=np.float64)
    means = estimators.mean(axis=0)
    standard_errors = estimators.std(axis=0, ddof=1) / np.sqrt(len(estimators))
    lower, upper = _bootstrap_mean_percentile_interval(estimators)
    z_scores = (means - targets) / standard_errors

    for index, label in enumerate(labels):
        assert lower[index] <= targets[index] <= upper[index], (
            f"{label}: target={targets[index]:.17g}, mean={means[index]:.17g}, "
            f"99% CI=[{lower[index]:.17g}, {upper[index]:.17g}]"
        )
        assert abs(z_scores[index]) <= 3.0, (
            f"{label}: mean={means[index]:.17g}, SE={standard_errors[index]:.17g}, "
            f"z={z_scores[index]:.17g}"
        )

    double_occ = diagnostics[:, 0]
    local_moment = diagnostics[:, 1]
    assert np.all((0.0 <= double_occ) & (double_occ <= estimators[:, 0] / 2.0))
    assert np.all(local_moment >= 0.0)


def _chain_level_equal_unequal_tau0_differences(completed_files):
    differences = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            ue = h5["meas_uneqlt"]
            eq_sign = float(eq["sign"][()])
            ue_sign = float(ue["sign"][()])
            per_observable = []
            for equal_name, unequal_name in (
                ("g00", "gt0"),
                ("nn", "nn"),
                ("xx", "xx"),
                ("zz", "zz"),
                ("pair_sw", "pair_sw"),
            ):
                equal = eq[equal_name][...] / eq_sign
                unequal_tau0 = ue[unequal_name][:N] / ue_sign
                per_observable.append(equal - unequal_tau0)
            differences.append(np.stack(per_observable))
    return np.asarray(differences)


def _studentized_statistics(means, standard_errors):
    means = np.asarray(means, dtype=np.float64)
    standard_errors = np.asarray(standard_errors, dtype=np.float64)
    nonzero = standard_errors > 0.0
    statistics = np.divide(
        np.abs(means),
        standard_errors,
        out=np.zeros_like(means, dtype=np.float64),
        where=nonzero,
    )
    return np.where(nonzero | (means == 0.0), statistics, np.inf)


def _paired_bootstrap_max_t_critical_value(
    differences,
    *,
    confidence=0.99,
    n_resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED + 1,
):
    flat = np.asarray(differences, dtype=np.float64).reshape(len(differences), -1)
    centered = flat - flat.mean(axis=0)
    fixed_standard_errors = flat.std(axis=0, ddof=1) / np.sqrt(len(flat))
    rng = np.random.default_rng(seed)
    n_chains = flat.shape[0]
    max_statistics = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 50
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_chains, size=(stop - start, n_chains))
        resamples = centered[indices]
        means = resamples.mean(axis=1)
        # Use the original chain-level standard errors as fixed scales.  Some
        # displacement slots have nearly zero variance; re-estimating their
        # standard errors inside every bootstrap resample makes the max-t
        # distribution spuriously heavy-tailed and the simultaneous test
        # effectively powerless.
        statistics = _studentized_statistics(means, fixed_standard_errors)
        max_statistics[start:stop] = statistics.max(axis=1)
    return float(np.quantile(max_statistics, confidence))


def test_d6_equal_and_unequal_time_tau0_are_statistically_consistent(
    completed_files,
):
    """D6: all five tau=0 correlators pass a simultaneous paired test."""
    differences = _chain_level_equal_unequal_tau0_differences(completed_files)
    flat = differences.reshape(len(differences), -1)
    means = flat.mean(axis=0)
    standard_errors = flat.std(axis=0, ddof=1) / np.sqrt(len(flat))
    observed = _studentized_statistics(means, standard_errors)
    observed_max = float(observed.max())
    critical = _paired_bootstrap_max_t_critical_value(differences)

    worst = int(np.argmax(observed))
    observable_names = ("g00/gt0", "nn", "xx", "zz", "pair_sw")
    observable_index, displacement = divmod(worst, N)
    assert observed_max <= critical, (
        f"worst tau=0 difference is {observable_names[observable_index]} "
        f"at displacement slot {displacement}: max |t|={observed_max:.8g}, "
        f"99% paired-bootstrap simultaneous critical value={critical:.8g}"
    )


def _square_displacement_maps():
    inversion = []
    rotation = []
    for y in range(NY):
        for x in range(NX):
            inversion.append((-x) % NX + NX * ((-y) % NY))
            rotation.append((-y) % NX + NX * (x % NY))
    return np.asarray(inversion), np.asarray(rotation)


def _chain_level_equal_time_correlations(completed_files):
    rows = []
    names = ("g00", "nn", "xx", "zz", "pair_sw")
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            sign = float(eq["sign"][()])
            rows.append(np.stack([eq[name][...] / sign for name in names]))
    return names, np.asarray(rows)


def test_d7_square_spatial_inversion_and_c4_rotation(completed_files):
    """D7: all equal-time scalar correlators respect square-lattice symmetry."""
    with h5py.File(PARAMS_FILE, "r") as h5:
        actual_map = h5["params/map_ij"][...]
    expected_map = np.empty((N, N), dtype=np.int32)
    for j in range(N):
        xj, yj = j % NX, j // NX
        for i in range(N):
            xi, yi = i % NX, i // NX
            expected_map[i, j] = (xj - xi) % NX + NX * ((yj - yi) % NY)
    np.testing.assert_array_equal(actual_map, expected_map)

    names, correlations = _chain_level_equal_time_correlations(completed_files)
    inversion, rotation = _square_displacement_maps()
    differences = np.concatenate(
        (correlations - correlations[:, :, inversion],
         correlations - correlations[:, :, rotation]),
        axis=1,
    )
    flat = differences.reshape(len(differences), -1)
    means = flat.mean(axis=0)
    standard_errors = flat.std(axis=0, ddof=1) / np.sqrt(len(flat))
    observed = _studentized_statistics(means, standard_errors)
    observed_max = float(observed.max())
    critical = _paired_bootstrap_max_t_critical_value(
        differences, seed=BOOTSTRAP_SEED + 2
    )
    worst = int(np.argmax(observed))
    comparison, remainder = divmod(worst, len(names) * N)
    observable, displacement = divmod(remainder, N)
    comparison_name = ("inversion", "C4 rotation")[comparison]
    assert observed_max <= critical, (
        f"worst spatial difference is {comparison_name} for {names[observable]} "
        f"at slot {displacement}: max |t|={observed_max:.8g}, "
        f"99% simultaneous critical={critical:.8g}"
    )


def _self_inverse_momentum_weights():
    weights = []
    labels = []
    for qx_index, qy_index, label in (
        (0, 0, "(0,0)"),
        (1, 0, "(pi,0)"),
        (0, 1, "(0,pi)"),
        (1, 1, "(pi,pi)"),
    ):
        weights.append(
            [
                (-1.0) ** (qx_index * (r % NX) + qy_index * (r // NX))
                for r in range(N)
            ]
        )
        labels.append(label)
    return labels, np.asarray(weights)


def _chain_level_bosonic_time_reflection_differences(completed_files):
    observable_names = ("nn", "xx", "zz", "pair_sw")
    momentum_labels, weights = _self_inverse_momentum_weights()
    rows = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            ue = h5["meas_uneqlt"]
            sign = float(ue["sign"][()])
            per_observable = []
            for name in observable_names:
                correlation = (ue[name][...] / sign).reshape(L, N)
                momentum_correlation = correlation @ weights.T
                per_observable.append(
                    np.stack(
                        [
                            momentum_correlation[t] - momentum_correlation[L - t]
                            for t in range(1, L // 2)
                        ]
                    ).T
                )
            rows.append(np.stack(per_observable))
    return observable_names, momentum_labels, np.asarray(rows)


def test_d7_bosonic_imaginary_time_reflection(completed_files):
    """D7: Hermitian self-inverse momenta satisfy C(q,tau)=C(q,beta-tau)."""
    names, momenta, differences = _chain_level_bosonic_time_reflection_differences(
        completed_files
    )
    flat = differences.reshape(len(differences), -1)
    means = flat.mean(axis=0)
    standard_errors = flat.std(axis=0, ddof=1) / np.sqrt(len(flat))
    observed = _studentized_statistics(means, standard_errors)
    observed_max = float(observed.max())
    critical = _paired_bootstrap_max_t_critical_value(
        differences, seed=BOOTSTRAP_SEED + 3
    )
    worst = int(np.argmax(observed))
    observable, remainder = divmod(worst, len(momenta) * (L // 2 - 1))
    momentum, time_offset = divmod(remainder, L // 2 - 1)
    assert observed_max <= critical, (
        f"worst time-reflection difference is {names[observable]} q={momenta[momentum]} "
        f"at tau index {time_offset + 1}: max |t|={observed_max:.8g}, "
        f"99% simultaneous critical={critical:.8g}"
    )


def test_d7_equal_time_structure_factors_are_positive(completed_files):
    """D7: Hermitian equal-time structure factors have positive 99% bounds."""
    names, correlations = _chain_level_equal_time_correlations(completed_files)
    selected = [names.index(name) for name in ("nn", "xx", "zz", "pair_sw")]
    structure_factors = np.fft.fft2(
        correlations[:, selected].reshape(len(correlations), len(selected), NY, NX),
        axes=(-2, -1),
    )
    real = structure_factors.real.reshape(len(correlations), len(selected), N)
    imaginary = structure_factors.imag.reshape(len(correlations), len(selected), N)

    imaginary_means = imaginary.mean(axis=0)
    imaginary_se = imaginary.std(axis=0, ddof=1) / np.sqrt(len(imaginary))
    imaginary_t = _studentized_statistics(imaginary_means, imaginary_se)
    imaginary_critical = _paired_bootstrap_max_t_critical_value(
        imaginary, seed=BOOTSTRAP_SEED + 4
    )
    assert float(imaginary_t.max()) <= imaginary_critical

    means = real.mean(axis=0)
    standard_errors = real.std(axis=0, ddof=1) / np.sqrt(len(real))
    positivity_critical = _paired_bootstrap_max_t_critical_value(
        real, seed=BOOTSTRAP_SEED + 5
    )
    lower_simultaneous_bound = means - positivity_critical * standard_errors
    assert np.all(lower_simultaneous_bound >= 0.0), (
        f"minimum 99% simultaneous lower structure-factor bound is "
        f"{lower_simultaneous_bound.min():.17g}"
    )


def _chain_level_pseudospin_estimators(completed_files):
    eta = _self_inverse_momentum_weights()[1][-1]
    rows = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            ue = h5["meas_uneqlt"]
            eq_sign = float(eq["sign"][()])
            ue_sign = float(ue["sign"][()])
            nn = eq["nn"][...] / eq_sign
            pair = eq["pair_sw"][...] / eq_sign
            nn_tau = (ue["nn"][...] / ue_sign).reshape(L, N)
            pair_tau = (ue["pair_sw"][...] / ue_sign).reshape(L, N)
            rows.append(
                (
                    eta @ nn,
                    pair.sum(),
                    DT * np.sum(nn_tau @ eta),
                    DT * pair_tau.sum(),
                )
            )
    return np.asarray(rows)


def _bootstrap_ratios(
    samples,
    numerator,
    denominator,
    *,
    n_resamples=BOOTSTRAP_RESAMPLES,
    seed=BOOTSTRAP_SEED + 6,
):
    rng = np.random.default_rng(seed)
    n_chains = len(samples)
    ratios = np.empty(n_resamples, dtype=np.float64)
    chunk_size = 500
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_chains, size=(stop - start, n_chains))
        means = samples[indices].mean(axis=1)
        ratios[start:stop] = means[:, numerator] / means[:, denominator]
    return np.quantile(ratios, (0.005, 0.995))


def test_d8_half_filled_pseudospin_su2_relations(completed_files):
    """D8: CDW and s-wave pair structure factors/susceptibilities obey SU(2)."""
    estimators = _chain_level_pseudospin_estimators(completed_files)
    differences = np.column_stack(
        (estimators[:, 0] - 2.0 * estimators[:, 1],
         estimators[:, 2] - 2.0 * estimators[:, 3])
    )
    lower, upper = _bootstrap_mean_percentile_interval(
        differences, seed=BOOTSTRAP_SEED + 7
    )
    means = differences.mean(axis=0)
    standard_errors = differences.std(axis=0, ddof=1) / np.sqrt(len(differences))
    z_scores = means / standard_errors
    labels = ("equal-time", "susceptibility")
    for index, label in enumerate(labels):
        assert lower[index] <= 0.0 <= upper[index], (
            f"{label} pseudospin difference mean={means[index]:.17g}, "
            f"99% CI=[{lower[index]:.17g}, {upper[index]:.17g}]"
        )
        assert abs(z_scores[index]) <= 3.0

    equal_ratio_interval = _bootstrap_ratios(estimators, 0, 1)
    susceptibility_ratio_interval = _bootstrap_ratios(
        estimators, 2, 3, seed=BOOTSTRAP_SEED + 8
    )
    assert equal_ratio_interval[0] <= 2.0 <= equal_ratio_interval[1]
    assert susceptibility_ratio_interval[0] <= 2.0 <= susceptibility_ratio_interval[1]


def _chain_level_d9_metrics(completed_files):
    eta = _self_inverse_momentum_weights()[1][-1]
    rows = []
    seed_summaries = []
    restart_counts = []
    initial_states = set()
    final_states = set()
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            ue = h5["meas_uneqlt"]
            eq_sign = float(eq["sign"][()])
            ue_sign = float(ue["sign"][()])
            density = float(eq["density"][0]) / eq_sign
            double_occ = float(eq["double_occ"][0]) / eq_sign
            nn = eq["nn"][...] / eq_sign
            pair = eq["pair_sw"][...] / eq_sign
            nn_tau = (ue["nn"][...] / ue_sign).reshape(L, N)
            pair_tau = (ue["pair_sw"][...] / ue_sign).reshape(L, N)
            rows.append(
                (
                    density,
                    double_occ,
                    density - 2.0 * double_occ,
                    eta @ nn,
                    pair.sum(),
                    DT * np.sum(nn_tau @ eta),
                    DT * pair_tau.sum(),
                )
            )
            init_rng = h5["state/init_rng"][...]
            seed_summaries.append(float(init_rng[0]) / float(2**64))
            initial_states.add(init_rng.tobytes())
            final_states.add(h5["state/hs"][...].tobytes())
        restart_counts.append(
            Path(f"{path}.log").read_text(encoding="utf-8").count("starting dqmc")
        )
    return (
        np.asarray(rows),
        np.asarray(seed_summaries),
        np.asarray(restart_counts),
        initial_states,
        final_states,
    )


def _welch_t(first, second):
    difference = first.mean(axis=0) - second.mean(axis=0)
    standard_error = np.sqrt(
        first.var(axis=0, ddof=1) / len(first)
        + second.var(axis=0, ddof=1) / len(second)
    )
    return _studentized_statistics(difference, standard_error)


def _d9_group_max_t_critical(metrics, *, seed=BOOTSTRAP_SEED + 9):
    rng = np.random.default_rng(seed)
    n_chains = len(metrics)
    max_statistics = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        permutation = rng.permutation(n_chains)
        permuted = metrics[permutation]
        first_last = _welch_t(permuted[:50], permuted[50:])
        even_odd = _welch_t(permuted[::2], permuted[1::2])
        max_statistics[index] = max(first_last.max(), even_odd.max())
    return float(np.quantile(max_statistics, 0.99))


def _correlation_t(predictors, metrics):
    predictor_z = (predictors - predictors.mean(axis=0)) / predictors.std(axis=0, ddof=1)
    metric_z = (metrics - metrics.mean(axis=0)) / metrics.std(axis=0, ddof=1)
    correlations = predictor_z.T @ metric_z / (len(metrics) - 1)
    denominator = np.maximum(1.0 - correlations**2, np.finfo(float).tiny)
    return np.abs(correlations) * np.sqrt((len(metrics) - 2) / denominator)


def _d9_trend_max_t_critical(
    predictors, metrics, *, seed=BOOTSTRAP_SEED + 10
):
    rng = np.random.default_rng(seed)
    max_statistics = np.empty(BOOTSTRAP_RESAMPLES)
    for index in range(BOOTSTRAP_RESAMPLES):
        permuted = metrics[rng.permutation(len(metrics))]
        max_statistics[index] = _correlation_t(predictors, permuted).max()
    return float(np.quantile(max_statistics, 0.99))


def test_d9_independent_chain_consistency(completed_files):
    """D9: detect duplicate, influential, outlying, grouped, or trended chains."""
    names = (
        "density",
        "double_occ",
        "local_moment",
        "S_cdw",
        "S_pair",
        "chi_cdw",
        "chi_pair",
    )
    metrics, seed_summaries, restarts, initial_states, final_states = (
        _chain_level_d9_metrics(completed_files)
    )
    failures = []
    if len(initial_states) != len(completed_files):
        failures.append("initial RNG states are not unique")
    if len(final_states) != len(completed_files):
        failures.append("final HS states are not unique")
    if not np.all(metrics.std(axis=0, ddof=1) > 0.0):
        failures.append("one or more chain-level metrics have zero variance")

    full_mean = metrics.mean(axis=0)
    full_se = metrics.std(axis=0, ddof=1) / np.sqrt(len(metrics))
    leave_one_out = (metrics.sum(axis=0) - metrics) / (len(metrics) - 1)
    influence = np.abs(leave_one_out - full_mean) / full_se
    worst_influence = np.unravel_index(np.argmax(influence), influence.shape)
    if influence[worst_influence] > 1.0:
        failures.append(
            f"leave-one-out influence {influence[worst_influence]:.6g} SE for "
            f"{completed_files[worst_influence[0]].name} {names[worst_influence[1]]}"
        )

    externally_studentized = np.empty_like(metrics)
    for chain in range(len(metrics)):
        others = np.delete(metrics, chain, axis=0)
        externally_studentized[chain] = (
            (metrics[chain] - others.mean(axis=0))
            / (others.std(axis=0, ddof=1) * np.sqrt(1.0 + 1.0 / len(others)))
        )
    outlier_critical = float(
        student_t.ppf(
            1.0 - 0.01 / (2.0 * externally_studentized.size),
            df=len(metrics) - 2,
        )
    )
    worst_outlier = np.unravel_index(
        np.argmax(np.abs(externally_studentized)), externally_studentized.shape
    )
    if abs(externally_studentized[worst_outlier]) > outlier_critical:
        failures.append(
            f"externally studentized residual "
            f"{externally_studentized[worst_outlier]:.6g} exceeds 99% "
            f"Bonferroni critical {outlier_critical:.6g} for "
            f"{completed_files[worst_outlier[0]].name} {names[worst_outlier[1]]}"
        )

    median = np.median(metrics, axis=0)
    mad = np.median(np.abs(metrics - median), axis=0)
    robust_z = 0.67448975 * np.abs(metrics - median) / mad
    worst_robust = np.unravel_index(np.argmax(robust_z), robust_z.shape)
    if robust_z[worst_robust] > 6.0:
        failures.append(
            f"robust z {robust_z[worst_robust]:.6g} exceeds 6 for "
            f"{completed_files[worst_robust[0]].name} {names[worst_robust[1]]}"
        )

    first_last = _welch_t(metrics[:50], metrics[50:])
    even_odd = _welch_t(metrics[::2], metrics[1::2])
    group_observed = max(first_last.max(), even_odd.max())
    group_critical = _d9_group_max_t_critical(metrics)
    if group_observed > group_critical:
        failures.append(
            f"pre-registered group max |t|={group_observed:.6g} exceeds "
            f"99% permutation critical={group_critical:.6g}"
        )

    predictors = np.column_stack(
        (np.arange(len(metrics), dtype=np.float64), seed_summaries, restarts)
    )
    trend_statistics = _correlation_t(predictors, metrics)
    trend_observed = float(trend_statistics.max())
    trend_critical = _d9_trend_max_t_critical(predictors, metrics)
    if trend_observed > trend_critical:
        failures.append(
            f"trend max |t|={trend_observed:.6g} exceeds "
            f"99% permutation critical={trend_critical:.6g}"
        )

    assert failures == [], "D9 diagnostics requiring investigation:\n" + "\n".join(
        failures
    )
