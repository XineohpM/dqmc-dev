"""Read-only Stage-I investigation of the attractive-Hubbard D9 outlier.

This file independently reconstructs the chain-level pairing susceptibility,
localizes the anomalous contribution, checks per-chain internal constraints and
log/restart provenance, and measures the sensitivity of ensemble conclusions.
The completed HDF5 files and logs are fingerprinted before and after the test
session.  No DQMC executable is called and no production source is imported.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.stats import t as student_t


DATA_DIR = Path("/Users/a9012/Desktop/dqmc_data/test/T_0.4")
PARAMS_FILE = DATA_DIR / "_T0.4_.h5.params"
SIM_FILE_RE = re.compile(r"_T0\.4__(\d+)\.h5$")
N_FILES = 100
TARGET = 85
SECOND_HIGH = 35
N = 64
NX = 8
NY = 8
L = 50
DT = 0.05
N_SWEEP = 1_050_000
N_SAMPLE_EQLT = 10_000_000
N_SAMPLE_UNEQLT = 500_000
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_RESAMPLES = 10_000
METRIC_NAMES = (
    "density",
    "double_occ",
    "local_moment",
    "S_cdw",
    "S_pair",
    "chi_cdw",
    "chi_pair",
)


def _json_line(label, values):
    print(f"{label}={json.dumps(values, sort_keys=True)}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(paths):
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns, _sha256(path))
        for path in paths
    }


@pytest.fixture(scope="session")
def completed_files():
    indexed = []
    for path in DATA_DIR.glob("_T0.4__*.h5"):
        match = SIM_FILE_RE.fullmatch(path.name)
        assert match is not None
        indexed.append((int(match.group(1)), path))
    indexed.sort()
    assert [index for index, _ in indexed] == list(range(N_FILES))
    return [path for _, path in indexed]


@pytest.fixture(scope="session", autouse=True)
def original_files_remain_unchanged(completed_files):
    paths = [PARAMS_FILE]
    paths.extend(completed_files)
    paths.extend(Path(f"{path}.log") for path in completed_files)
    before = _manifest(paths)
    yield
    after = _manifest(paths)
    assert after == before, "one or more original HDF5/log files changed"


def _staggered_weights():
    return np.asarray(
        [(-1.0) ** ((slot % NX) + (slot // NX)) for slot in range(N)]
    )


@pytest.fixture(scope="session")
def chain_data(completed_files):
    eta = _staggered_weights()
    density = np.empty(N_FILES)
    double_occ = np.empty(N_FILES)
    equal_pair = np.empty((N_FILES, N))
    equal_nn = np.empty((N_FILES, N))
    pair_tau = np.empty((N_FILES, L, N))
    nn_tau = np.empty((N_FILES, L, N))
    equal_sign = np.empty(N_FILES)
    unequal_sign = np.empty(N_FILES)
    equal_samples = np.empty(N_FILES, dtype=np.int64)
    unequal_samples = np.empty(N_FILES, dtype=np.int64)
    init_rng_hashes = []
    final_hs_hashes = []

    for index, path in enumerate(completed_files):
        with h5py.File(path, "r") as h5:
            eq = h5["meas_eqlt"]
            ue = h5["meas_uneqlt"]
            equal_sign[index] = float(eq["sign"][()])
            unequal_sign[index] = float(ue["sign"][()])
            equal_samples[index] = int(eq["n_sample"][()])
            unequal_samples[index] = int(ue["n_sample"][()])
            density[index] = float(eq["density"][0]) / equal_sign[index]
            double_occ[index] = (
                float(eq["double_occ"][0]) / equal_sign[index]
            )
            equal_pair[index] = eq["pair_sw"][...] / equal_sign[index]
            equal_nn[index] = eq["nn"][...] / equal_sign[index]
            pair_tau[index] = (
                ue["pair_sw"][...] / unequal_sign[index]
            ).reshape(L, N)
            nn_tau[index] = (ue["nn"][...] / unequal_sign[index]).reshape(L, N)
            init_rng_hashes.append(hashlib.sha256(h5["state/init_rng"][...]).hexdigest())
            final_hs_hashes.append(hashlib.sha256(h5["state/hs"][...]).hexdigest())

    metrics = np.column_stack(
        (
            density,
            double_occ,
            density - 2.0 * double_occ,
            equal_nn @ eta,
            equal_pair.sum(axis=1),
            DT * np.sum(nn_tau @ eta, axis=1),
            DT * pair_tau.sum(axis=(1, 2)),
        )
    )
    return SimpleNamespace(
        density=density,
        double_occ=double_occ,
        equal_pair=equal_pair,
        equal_nn=equal_nn,
        pair_tau=pair_tau,
        nn_tau=nn_tau,
        equal_sign=equal_sign,
        unequal_sign=unequal_sign,
        equal_samples=equal_samples,
        unequal_samples=unequal_samples,
        init_rng_hashes=init_rng_hashes,
        final_hs_hashes=final_hs_hashes,
        metrics=metrics,
    )


def _external_residual(values, target=TARGET):
    values = np.asarray(values, dtype=np.float64)
    others = np.delete(values, target, axis=0)
    scale = others.std(axis=0, ddof=1) * np.sqrt(1.0 + 1.0 / len(others))
    difference = values[target] - others.mean(axis=0)
    return np.divide(
        difference,
        scale,
        out=np.zeros_like(difference, dtype=np.float64),
        where=scale > 0.0,
    )


def _bonferroni_critical(n_comparisons, *, df=N_FILES - 2):
    return float(student_t.ppf(1.0 - 0.01 / (2.0 * n_comparisons), df=df))


def _rank(values, index):
    return int(np.argsort(np.argsort(values))[index] + 1)


def _bootstrap_mean_interval(samples, *, seed):
    samples = np.asarray(samples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty((BOOTSTRAP_RESAMPLES, samples.shape[1]))
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(
            0, len(samples), size=(stop - start, len(samples))
        )
        means[start:stop] = samples[indices].mean(axis=1)
    return np.quantile(means, (0.005, 0.995), axis=0)


def test_i0_i1_independent_reconstruction_and_input_invariants(
    completed_files, chain_data
):
    manual_chi_pair = []
    coordinate_chi_pair = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            raw = h5["meas_uneqlt/pair_sw"]
            sign = float(h5["meas_uneqlt/sign"][()])
            total = 0.0
            for flat_index in range(raw.shape[0]):
                total += float(raw[flat_index]) / sign
            manual_chi_pair.append(DT * total)

            cube = np.empty((L, NY, NX), dtype=np.float64)
            for tau in range(L):
                for y in range(NY):
                    for x in range(NX):
                        cube[tau, y, x] = (
                            float(raw[tau * N + y * NX + x]) / sign
                        )
            coordinate_chi_pair.append(DT * cube.sum())

    manual_chi_pair = np.asarray(manual_chi_pair)
    coordinate_chi_pair = np.asarray(coordinate_chi_pair)
    vectorized = chain_data.metrics[:, METRIC_NAMES.index("chi_pair")]
    np.testing.assert_allclose(manual_chi_pair, vectorized, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(
        coordinate_chi_pair, vectorized, rtol=1e-12, atol=1e-13
    )

    assert np.all(chain_data.equal_sign == chain_data.equal_samples)
    assert np.all(chain_data.unequal_sign == chain_data.unequal_samples)
    assert np.all(chain_data.equal_samples == N_SAMPLE_EQLT)
    assert np.all(chain_data.unequal_samples == N_SAMPLE_UNEQLT)
    assert len(set(chain_data.init_rng_hashes)) == N_FILES
    assert len(set(chain_data.final_hs_hashes)) == N_FILES

    with h5py.File(PARAMS_FILE, "r") as h5:
        actual_map = h5["params/map_ij"][...]
    expected_map = np.empty((N, N), dtype=np.int32)
    for j in range(N):
        xj, yj = j % NX, j // NX
        for i in range(N):
            xi, yi = i % NX, i // NX
            expected_map[i, j] = (xj - xi) % NX + NX * ((yj - yi) % NY)
    np.testing.assert_array_equal(actual_map, expected_map)

    metrics = chain_data.metrics
    externally_studentized = np.empty_like(metrics)
    for chain in range(N_FILES):
        externally_studentized[chain] = _external_residual(metrics, chain)
    critical = _bonferroni_critical(metrics.size)
    median = np.median(metrics, axis=0)
    mad = np.median(np.abs(metrics - median), axis=0)
    robust_z = 0.67448975 * np.abs(metrics - median) / mad
    worst = np.unravel_index(
        np.argmax(np.abs(externally_studentized)), externally_studentized.shape
    )
    robust_worst = np.unravel_index(np.argmax(robust_z), robust_z.shape)
    assert worst == (TARGET, METRIC_NAMES.index("chi_pair"))
    assert robust_worst == worst
    _json_line(
        "I0_I1",
        {
            "chi_pair_85": vectorized[TARGET],
            "max_reconstruction_abs_error": float(
                max(
                    np.max(np.abs(manual_chi_pair - vectorized)),
                    np.max(np.abs(coordinate_chi_pair - vectorized)),
                )
            ),
            "outlier_critical": critical,
            "outlier_residual": externally_studentized[worst],
            "robust_z": robust_z[robust_worst],
            "worst_file": completed_files[worst[0]].name,
            "worst_metric": METRIC_NAMES[worst[1]],
        },
    )


def test_i2_localize_pairing_susceptibility_excess(completed_files, chain_data):
    profiles = chain_data.pair_tau
    other_mean = np.delete(profiles, TARGET, axis=0).mean(axis=0)
    delta = profiles[TARGET] - other_mean
    tau_contribution = DT * delta.sum(axis=1)
    spatial_contribution = DT * delta.sum(axis=0)
    cell_residual = _external_residual(profiles)
    tau_residual = _external_residual(DT * profiles.sum(axis=2))
    spatial_residual = _external_residual(DT * profiles.sum(axis=1))
    cell_residual_35 = _external_residual(profiles, SECOND_HIGH)
    tau_residual_35 = _external_residual(
        DT * profiles.sum(axis=2), SECOND_HIGH
    )
    spatial_residual_35 = _external_residual(
        DT * profiles.sum(axis=1), SECOND_HIGH
    )

    cell_worst = np.unravel_index(np.argmax(np.abs(cell_residual)), (L, N))
    tau_worst = int(np.argmax(np.abs(tau_residual)))
    spatial_worst = int(np.argmax(np.abs(spatial_residual)))
    cell_critical = _bonferroni_critical(L * N)
    tau_critical = _bonferroni_critical(L)
    spatial_critical = _bonferroni_critical(N)
    total_excess = float(tau_contribution.sum())
    assert total_excess > 0.0
    assert total_excess == pytest.approx(
        chain_data.metrics[TARGET, -1]
        - np.delete(chain_data.metrics[:, -1], TARGET).mean(),
        abs=1e-13,
    )

    sorted_tau = np.sort(np.abs(tau_contribution))[::-1]
    sorted_cell = np.sort(np.abs(DT * delta).ravel())[::-1]
    top_tau_indices = np.argsort(np.abs(tau_contribution))[::-1][:5]
    top_spatial_indices = np.argsort(np.abs(spatial_contribution))[::-1][:5]
    _json_line(
        "I2",
        {
            "cell_critical": cell_critical,
            "cell_flag_count": int(np.sum(np.abs(cell_residual) > cell_critical)),
            "cell_flag_count_35": int(
                np.sum(np.abs(cell_residual_35) > cell_critical)
            ),
            "cell_max_abs_residual": float(np.max(np.abs(cell_residual))),
            "cell_worst_absolute_delta": float(abs(delta[cell_worst])),
            "cell_worst_r": int(cell_worst[1]),
            "cell_worst_tau": int(cell_worst[0]),
            "cell_worst_target_value": float(profiles[TARGET][cell_worst]),
            "cell_worst_other_mean": float(other_mean[cell_worst]),
            "positive_cell_fraction": float(np.mean(delta > 0.0)),
            "positive_tau_count": int(np.sum(tau_contribution > 0.0)),
            "spatial_critical": spatial_critical,
            "spatial_flag_count": int(
                np.sum(np.abs(spatial_residual) > spatial_critical)
            ),
            "spatial_flag_count_35": int(
                np.sum(np.abs(spatial_residual_35) > spatial_critical)
            ),
            "spatial_max_abs_residual": float(
                np.max(np.abs(spatial_residual))
            ),
            "spatial_worst_absolute_contribution": float(
                abs(spatial_contribution[spatial_worst])
            ),
            "spatial_worst_r": spatial_worst,
            "tau_critical": tau_critical,
            "tau_flag_count": int(np.sum(np.abs(tau_residual) > tau_critical)),
            "tau_flag_count_35": int(
                np.sum(np.abs(tau_residual_35) > tau_critical)
            ),
            "tau_max_abs_residual": float(np.max(np.abs(tau_residual))),
            "tau_worst_absolute_contribution": float(
                abs(tau_contribution[tau_worst])
            ),
            "tau_worst": tau_worst,
            "top_5_cell_abs_share": float(
                sorted_cell[:5].sum() / sorted_cell.sum()
            ),
            "top_5_spatial_indices": top_spatial_indices.tolist(),
            "top_5_tau_abs_share": float(sorted_tau[:5].sum() / sorted_tau.sum()),
            "top_5_tau_indices": top_tau_indices.tolist(),
            "total_chi_excess": total_excess,
            "chi_pair_35": chain_data.metrics[SECOND_HIGH, -1],
            "chi_pair_85": chain_data.metrics[TARGET, -1],
        },
    )


def _spatial_maps():
    inversion = []
    rotation = []
    for y in range(NY):
        for x in range(NX):
            inversion.append((-x) % NX + NX * ((-y) % NY))
            rotation.append((-y) % NX + NX * (x % NY))
    return np.asarray(inversion), np.asarray(rotation)


def test_i3_target_chain_internal_symmetry_and_pseudospin(chain_data):
    q0_tau = chain_data.pair_tau.sum(axis=2)
    time_reflection = np.stack(
        [q0_tau[:, tau] - q0_tau[:, L - tau] for tau in range(1, L // 2)],
        axis=1,
    )
    time_residual = _external_residual(time_reflection)
    time_residual_35 = _external_residual(time_reflection, SECOND_HIGH)

    inversion, rotation = _spatial_maps()
    spatial_differences = np.concatenate(
        (
            chain_data.pair_tau - chain_data.pair_tau[:, :, inversion],
            chain_data.pair_tau - chain_data.pair_tau[:, :, rotation],
        ),
        axis=1,
    )
    spatial_residual = _external_residual(spatial_differences)
    spatial_residual_35 = _external_residual(spatial_differences, SECOND_HIGH)
    tau0_difference = chain_data.pair_tau[:, 0] - chain_data.equal_pair
    tau0_residual = _external_residual(tau0_difference)
    tau0_residual_35 = _external_residual(tau0_difference, SECOND_HIGH)

    eta = _staggered_weights()
    equal_pseudospin = chain_data.equal_nn @ eta - 2.0 * chain_data.equal_pair.sum(
        axis=1
    )
    chi_pseudospin = (
        DT * np.sum(chain_data.nn_tau @ eta, axis=1)
        - 2.0 * chain_data.metrics[:, -1]
    )
    pseudo = np.column_stack((equal_pseudospin, chi_pseudospin))
    pseudo_residual = _external_residual(pseudo)
    pseudo_residual_35 = _external_residual(pseudo, SECOND_HIGH)

    time_critical = _bonferroni_critical(time_reflection.shape[1])
    spatial_critical = _bonferroni_critical(spatial_differences[0].size)
    tau0_critical = _bonferroni_critical(N)
    pseudo_critical = _bonferroni_critical(2)
    time_max_by_chain = np.max(np.abs(time_reflection), axis=1)
    spatial_max_by_chain = np.max(
        np.abs(spatial_differences).reshape(N_FILES, -1), axis=1
    )
    tau0_max_by_chain = np.max(np.abs(tau0_difference), axis=1)
    chi_pair = chain_data.metrics[:, -1]
    ordinary = np.ones(N_FILES, dtype=bool)
    ordinary[[SECOND_HIGH, TARGET]] = False
    time_worst = int(np.argmax(np.abs(time_residual))) + 1
    spatial_worst = np.unravel_index(
        np.argmax(np.abs(spatial_residual)), (2, L, N)
    )
    tau0_worst = int(np.argmax(np.abs(tau0_residual)))
    _json_line(
        "I3",
        {
            "chi_pseudospin_85": chi_pseudospin[TARGET],
            "equal_pseudospin_85": equal_pseudospin[TARGET],
            "pseudospin_critical": pseudo_critical,
            "pseudospin_max_abs_residual_85": float(
                np.max(np.abs(pseudo_residual))
            ),
            "pseudospin_max_abs_residual_35": float(
                np.max(np.abs(pseudo_residual_35))
            ),
            "spatial_critical": spatial_critical,
            "spatial_flag_count_85": int(
                np.sum(np.abs(spatial_residual) > spatial_critical)
            ),
            "spatial_flag_count_35": int(
                np.sum(np.abs(spatial_residual_35) > spatial_critical)
            ),
            "spatial_max_abs_residual_85": float(
                np.max(np.abs(spatial_residual))
            ),
            "spatial_max_absolute_difference_85": float(
                np.max(np.abs(spatial_differences[TARGET]))
            ),
            "spatial_max_absolute_difference_35": float(
                np.max(np.abs(spatial_differences[SECOND_HIGH]))
            ),
            "spatial_max_absolute_difference_median": float(
                np.median(spatial_max_by_chain)
            ),
            "spatial_max_absolute_difference_rank_85": _rank(
                spatial_max_by_chain, TARGET
            ),
            "spatial_max_absolute_difference_rank_35": _rank(
                spatial_max_by_chain, SECOND_HIGH
            ),
            "spatial_max_vs_chi_pair_correlation": float(
                np.corrcoef(spatial_max_by_chain, chi_pair)[0, 1]
            ),
            "spatial_max_vs_chi_pair_correlation_without_35_85": float(
                np.corrcoef(spatial_max_by_chain[ordinary], chi_pair[ordinary])[0, 1]
            ),
            "spatial_worst_comparison": ("inversion", "C4")[spatial_worst[0]],
            "spatial_worst_r": int(spatial_worst[2]),
            "spatial_worst_tau": int(spatial_worst[1]),
            "tau0_critical": tau0_critical,
            "tau0_flag_count_85": int(
                np.sum(np.abs(tau0_residual) > tau0_critical)
            ),
            "tau0_flag_count_35": int(
                np.sum(np.abs(tau0_residual_35) > tau0_critical)
            ),
            "tau0_max_abs_residual_85": float(
                np.max(np.abs(tau0_residual))
            ),
            "tau0_max_absolute_difference_85": float(
                np.max(np.abs(tau0_difference[TARGET]))
            ),
            "tau0_max_absolute_difference_35": float(
                np.max(np.abs(tau0_difference[SECOND_HIGH]))
            ),
            "tau0_max_absolute_difference_median": float(
                np.median(tau0_max_by_chain)
            ),
            "tau0_max_absolute_difference_rank_85": _rank(
                tau0_max_by_chain, TARGET
            ),
            "tau0_max_absolute_difference_rank_35": _rank(
                tau0_max_by_chain, SECOND_HIGH
            ),
            "tau0_max_vs_chi_pair_correlation": float(
                np.corrcoef(tau0_max_by_chain, chi_pair)[0, 1]
            ),
            "tau0_max_vs_chi_pair_correlation_without_35_85": float(
                np.corrcoef(tau0_max_by_chain[ordinary], chi_pair[ordinary])[0, 1]
            ),
            "tau0_worst_r": tau0_worst,
            "time_reflection_critical": time_critical,
            "time_reflection_flag_count_85": int(
                np.sum(np.abs(time_residual) > time_critical)
            ),
            "time_reflection_flag_count_35": int(
                np.sum(np.abs(time_residual_35) > time_critical)
            ),
            "time_reflection_max_abs_residual_85": float(
                np.max(np.abs(time_residual))
            ),
            "time_reflection_max_absolute_difference_85": float(
                np.max(np.abs(time_reflection[TARGET]))
            ),
            "time_reflection_max_absolute_difference_35": float(
                np.max(np.abs(time_reflection[SECOND_HIGH]))
            ),
            "time_reflection_max_absolute_difference_median": float(
                np.median(time_max_by_chain)
            ),
            "time_reflection_max_absolute_difference_rank_85": _rank(
                time_max_by_chain, TARGET
            ),
            "time_reflection_max_absolute_difference_rank_35": _rank(
                time_max_by_chain, SECOND_HIGH
            ),
            "time_reflection_max_vs_chi_pair_correlation": float(
                np.corrcoef(time_max_by_chain, chi_pair)[0, 1]
            ),
            "time_reflection_max_vs_chi_pair_correlation_without_35_85": float(
                np.corrcoef(time_max_by_chain[ordinary], chi_pair[ordinary])[0, 1]
            ),
            "time_reflection_worst_tau": time_worst,
        },
    )


def _parse_log_runs(text):
    parts = text.split("starting dqmc")
    runs = []
    progress = re.compile(r"(\d+)/1050000 sweeps completed")
    for index in range(1, len(parts)):
        start_matches = progress.findall(parts[index - 1])
        end_matches = progress.findall(parts[index])
        assert start_matches and end_matches
        # The text after this start also contains the next run's header.  The
        # first progress entry is this run's stopping/completion point; taking
        # the last one would incorrectly substitute the next checkpoint start.
        runs.append((int(start_matches[-1]), int(end_matches[0])))
    return runs


def test_i4_hdf5_log_and_restart_forensics(completed_files, chain_data):
    restart_counts = []
    lost_work = []
    commit_pairs = set()
    generator_commits = set()
    warning_counts = []
    target_runs = None
    link_targets = set()

    for index, path in enumerate(completed_files):
        with h5py.File(path, "r") as h5:
            params_link = h5.get("params", getlink=True)
            metadata_link = h5.get("metadata", getlink=True)
            assert isinstance(params_link, h5py.ExternalLink)
            assert isinstance(metadata_link, h5py.ExternalLink)
            link_targets.add((params_link.filename, metadata_link.filename))
            assert int(h5["state/sweep"][()]) == N_SWEEP
            assert int(h5["state/partial_write"][()]) == 0
            assert int(h5["meas_eqlt/n_sample"][()]) == N_SAMPLE_EQLT
            assert int(h5["meas_uneqlt/n_sample"][()]) == N_SAMPLE_UNEQLT

        text = Path(f"{path}.log").read_text(encoding="utf-8")
        runs = _parse_log_runs(text)
        starts = [start for start, _ in runs]
        ends = [end for _, end in runs]
        assert starts[0] == 0
        assert ends[-1] == N_SWEEP
        assert all(start % 10_000 == 0 for start in starts[1:])
        assert all(later >= earlier for earlier, later in zip(starts, starts[1:]))
        discarded = []
        for run_index in range(len(runs) - 1):
            start, end = runs[run_index]
            next_start = runs[run_index + 1][0]
            assert start <= next_start <= end
            assert end - next_start < 10_000
            discarded.append(end - next_start)
        restart_counts.append(len(runs))
        lost_work.append(sum(discarded))
        if index == TARGET:
            target_runs = runs

        executable = re.findall(r"executable commit id (\S+)", text)
        generator = re.findall(r"hdf5 generation script commit id (\S+)", text)
        assert executable and generator
        commit_pairs.update(executable)
        generator_commits.update(generator)
        assert "sim_data_save() succeeded" in text
        assert "1050000/1050000 sweeps completed" in text
        warning_counts.append(
            len(re.findall(r"(?i)\b(?:nan|inf|warning|error|failed)\b", text))
        )

    assert link_targets == {(PARAMS_FILE.name, PARAMS_FILE.name)}
    assert commit_pairs == {"c48e98f"}
    assert generator_commits == {"c35cce3"}
    assert target_runs is not None
    chi_pair = chain_data.metrics[:, -1]
    restart_correlation = float(np.corrcoef(restart_counts, chi_pair)[0, 1])
    lost_work_correlation = float(np.corrcoef(lost_work, chi_pair)[0, 1])
    _json_line(
        "I4",
        {
            "chi_pair_restart_correlation": restart_correlation,
            "chi_pair_discarded_work_correlation": lost_work_correlation,
            "max_restart_count": int(np.max(restart_counts)),
            "median_restart_count": float(np.median(restart_counts)),
            "target_discarded_uncheckpointed_sweeps": int(lost_work[TARGET]),
            "target_restart_count": int(restart_counts[TARGET]),
            "target_runs": target_runs,
            "target_warning_token_count": int(warning_counts[TARGET]),
            "warning_token_count_all_logs": int(sum(warning_counts)),
        },
    )


def _d8_summary(estimators, *, seed):
    differences = np.column_stack(
        (estimators[:, 0] - 2.0 * estimators[:, 1], estimators[:, 2] - 2.0 * estimators[:, 3])
    )
    difference_interval = _bootstrap_mean_interval(differences, seed=seed)
    rng = np.random.default_rng(seed + 1)
    ratios = np.empty((BOOTSTRAP_RESAMPLES, 2))
    for start in range(0, BOOTSTRAP_RESAMPLES, 500):
        stop = min(start + 500, BOOTSTRAP_RESAMPLES)
        indices = rng.integers(
            0, len(estimators), size=(stop - start, len(estimators))
        )
        means = estimators[indices].mean(axis=1)
        ratios[start:stop] = means[:, (0, 2)] / means[:, (1, 3)]
    ratio_interval = np.quantile(ratios, (0.005, 0.995), axis=0)
    means = estimators.mean(axis=0)
    return {
        "difference_means": (means[[0, 2]] - 2.0 * means[[1, 3]]).tolist(),
        "difference_interval": difference_interval.tolist(),
        "ratios": (means[[0, 2]] / means[[1, 3]]).tolist(),
        "ratio_interval": ratio_interval.tolist(),
    }


def test_i5_robustness_and_ensemble_sensitivity(chain_data):
    metrics = chain_data.metrics
    chi_pair = metrics[:, -1]
    sorted_chi = np.sort(chi_pair)
    trimmed_mean = float(sorted_chi[10:-10].mean())
    group_means = chi_pair.reshape(10, 10).mean(axis=1)
    median_of_means = float(np.median(group_means))
    full_mean = float(chi_pair.mean())
    full_se = float(chi_pair.std(ddof=1) / np.sqrt(N_FILES))
    without_target = np.delete(chi_pair, TARGET)
    without_mean = float(without_target.mean())
    without_se = float(without_target.std(ddof=1) / np.sqrt(len(without_target)))

    eta = _staggered_weights()
    d8 = np.column_stack(
        (
            chain_data.equal_nn @ eta,
            chain_data.equal_pair.sum(axis=1),
            DT * np.sum(chain_data.nn_tau @ eta, axis=1),
            chi_pair,
        )
    )
    full_d8 = _d8_summary(d8, seed=BOOTSTRAP_SEED)
    without_d8 = _d8_summary(
        np.delete(d8, TARGET, axis=0), seed=BOOTSTRAP_SEED + 2
    )

    medians = np.median(metrics, axis=0)
    mads = np.median(np.abs(metrics - medians), axis=0)
    robust_coordinates = 0.67448975 * (metrics - medians) / mads
    robust_norms = np.sqrt(np.sum(robust_coordinates**2, axis=1))
    ranks = np.argsort(np.argsort(chi_pair)) + 1
    assert ranks[TARGET] == N_FILES
    assert ranks[SECOND_HIGH] == N_FILES - 1
    _json_line(
        "I5",
        {
            "chi_pair_35_rank": int(ranks[SECOND_HIGH]),
            "chi_pair_85_rank": int(ranks[TARGET]),
            "empirical_upper_tail_probability_85": 1.0 / N_FILES,
            "full_d8": full_d8,
            "full_mean": full_mean,
            "full_se": full_se,
            "leave_85_out_shift_in_full_se": (without_mean - full_mean) / full_se,
            "median": float(np.median(chi_pair)),
            "median_of_means": median_of_means,
            "robust_norm_85": float(robust_norms[TARGET]),
            "robust_norm_rank_85": int(
                np.argsort(np.argsort(robust_norms))[TARGET] + 1
            ),
            "trimmed_mean_10_percent": trimmed_mean,
            "without_85_d8": without_d8,
            "without_85_mean": without_mean,
            "without_85_se": without_se,
        },
    )


def test_i6_stage_i_classification(completed_files, chain_data):
    time_bin_datasets = []
    for path in completed_files:
        with h5py.File(path, "r") as h5:
            names = []
            h5.visit(names.append)
            time_bin_datasets.extend(
                f"{path.name}:{name}"
                for name in names
                if re.search(r"(?i)(?:time.?bin|bin(?:ned)?|history|trajectory)", name)
            )
    assert time_bin_datasets == []
    _json_line(
        "I6",
        {
            "classification": "E_evidence_insufficient",
            "d9_status": "FAIL_needs_further_investigation",
            "missing_evidence": [
                "time-binned measurements",
                "chain-level autocorrelation time",
                "effective sample size",
                "restart-controlled reproduction",
            ],
            "phase_ii_executed": False,
            "phase_iii_executed": False,
            "production_recommendation": "controlled_pilot_only",
        },
    )
