#!/usr/bin/env python3
"""Shared validation and paired-bootstrap utilities for correlator data.

The canonical input is a self-contained ``*_paired.npz`` bundle whose rows are
aligned Monte Carlo bins:

    numerator[b, ...] = sum_{i in bin b} sign_i * observable_i
    sign[b]           = sum_{i in bin b} sign_i
    n_sample[b]       = number of measurements in bin b

Physical estimates are ratios of sums.  Per-bin ratios must never be averaged
or resampled without the matching accumulated sign.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


FORMAT_VERSION = 1
REQUIRED_ARRAY_KEYS = (
    "numerator",
    "sign",
    "n_sample",
    "tau",
    "mean",
    "stderr",
)
REQUIRED_METADATA_KEYS = (
    "format_version",
    "observable",
    "beta",
    "dt",
    "normalization",
)
DEFAULT_DENOMINATOR_RTOL = 1e-12


class PairedBundleError(ValueError):
    """Raised when a paired correlator bundle violates the data contract."""


@dataclass(frozen=True)
class PairedBundle:
    """Validated paired correlator data."""

    numerator: np.ndarray
    sign: np.ndarray
    n_sample: np.ndarray
    tau: np.ndarray
    mean: np.ndarray
    stderr: np.ndarray
    metadata: Mapping[str, Any]
    source_files: Optional[np.ndarray] = None

    @property
    def nbin(self) -> int:
        return int(self.numerator.shape[0])

    @property
    def ntau(self) -> int:
        return int(self.numerator.shape[1])


def _require_numeric(name: str, array: np.ndarray, allow_complex: bool) -> None:
    if array.dtype.kind not in "biufc":
        raise PairedBundleError(
            "{} must have a numeric dtype, got {}".format(name, array.dtype)
        )
    if not allow_complex and np.iscomplexobj(array):
        raise PairedBundleError("{} must be real-valued".format(name))
    if not np.all(np.isfinite(array)):
        raise PairedBundleError("{} contains non-finite values".format(name))


def _validate_tolerances(denominator_rtol: float, denominator_atol: float) -> None:
    if not np.isfinite(denominator_rtol) or denominator_rtol < 0:
        raise ValueError("denominator_rtol must be finite and non-negative")
    if not np.isfinite(denominator_atol) or denominator_atol < 0:
        raise ValueError("denominator_atol must be finite and non-negative")


def _denominator_is_small(
    denominator: np.ndarray,
    absolute_weight: np.ndarray,
    denominator_rtol: float,
    denominator_atol: float,
) -> np.ndarray:
    return np.abs(denominator) <= (
        denominator_atol + denominator_rtol * absolute_weight
    )


def _metadata_scalar(key: str, value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise PairedBundleError(
            "metadata {!r} has object dtype; bundles must load with "
            "allow_pickle=False".format(key)
        )
    if array.size != 1:
        raise PairedBundleError("metadata {!r} must be scalar".format(key))
    return array.reshape(-1)[0].item()


def ratio_of_sums(
    numerator: np.ndarray,
    sign: np.ndarray,
    *,
    denominator_rtol: float = DEFAULT_DENOMINATOR_RTOL,
    denominator_atol: float = 0.0,
) -> np.ndarray:
    """Return ``sum(numerator, axis=0) / sum(sign)`` after validation."""

    _validate_tolerances(denominator_rtol, denominator_atol)
    numerator = np.asarray(numerator)
    sign = np.asarray(sign)

    if numerator.ndim < 2:
        raise PairedBundleError(
            "numerator must have shape (Nbin, ...), got {}".format(
                numerator.shape
            )
        )
    if sign.ndim != 1:
        raise PairedBundleError(
            "sign must have shape (Nbin,), got {}".format(sign.shape)
        )
    if numerator.shape[0] != sign.shape[0]:
        raise PairedBundleError(
            "numerator/sign Nbin mismatch: {} vs {}".format(
                numerator.shape[0], sign.shape[0]
            )
        )
    if numerator.shape[0] == 0:
        raise PairedBundleError("paired data contains no bins")

    _require_numeric("numerator", numerator, allow_complex=True)
    _require_numeric("sign", sign, allow_complex=True)

    denominator = sign.sum()
    absolute_weight = np.abs(sign).sum()
    if _denominator_is_small(
        denominator,
        absolute_weight,
        denominator_rtol,
        denominator_atol,
    ):
        raise PairedBundleError(
            "total accumulated sign is too close to zero for a stable ratio"
        )
    return numerator.sum(axis=0) / denominator


def validate_paired_bundle(
    *,
    numerator: np.ndarray,
    sign: np.ndarray,
    n_sample: np.ndarray,
    tau: np.ndarray,
    mean: np.ndarray,
    stderr: np.ndarray,
    metadata: Mapping[str, Any],
    source_files: Optional[Sequence[str]] = None,
    require_equal_n_sample: bool = True,
    denominator_rtol: float = DEFAULT_DENOMINATOR_RTOL,
    denominator_atol: float = 0.0,
    mean_rtol: float = 1e-10,
    mean_atol: float = 1e-12,
) -> PairedBundle:
    """Validate arrays and return a normalized :class:`PairedBundle`.

    ``require_equal_n_sample=True`` enforces the completed-bin contract used by
    the correlator extractors.  Set it to ``False`` only when a consumer has a
    statistically justified unequal-cluster bootstrap.
    """

    numerator = np.asarray(numerator)
    sign = np.asarray(sign)
    n_sample = np.asarray(n_sample)
    tau = np.asarray(tau)
    mean = np.asarray(mean)
    stderr = np.asarray(stderr)

    if numerator.ndim != 2:
        raise PairedBundleError(
            "numerator must have shape (Nbin, L), got {}".format(
                numerator.shape
            )
        )
    nbin, ntau = numerator.shape
    if nbin == 0 or ntau == 0:
        raise PairedBundleError("numerator must contain at least one bin and tau")
    if sign.shape != (nbin,):
        raise PairedBundleError(
            "sign must have shape ({},), got {}".format(nbin, sign.shape)
        )
    if n_sample.shape != (nbin,):
        raise PairedBundleError(
            "n_sample must have shape ({},), got {}".format(
                nbin, n_sample.shape
            )
        )
    if tau.shape != (ntau,):
        raise PairedBundleError(
            "tau must have shape ({},), got {}".format(ntau, tau.shape)
        )
    if mean.shape != (ntau,):
        raise PairedBundleError(
            "mean must have shape ({},), got {}".format(ntau, mean.shape)
        )
    if stderr.shape != (ntau,):
        raise PairedBundleError(
            "stderr must have shape ({},), got {}".format(ntau, stderr.shape)
        )

    _require_numeric("numerator", numerator, allow_complex=True)
    _require_numeric("sign", sign, allow_complex=True)
    _require_numeric("n_sample", n_sample, allow_complex=False)
    _require_numeric("tau", tau, allow_complex=False)
    _require_numeric("mean", mean, allow_complex=True)
    _require_numeric("stderr", stderr, allow_complex=False)

    if np.any(n_sample <= 0):
        raise PairedBundleError("n_sample must be strictly positive")
    if require_equal_n_sample and not np.all(n_sample == n_sample[0]):
        raise PairedBundleError(
            "n_sample is not equal across bins; apply the completed-bin mask "
            "before creating the bundle"
        )
    if ntau > 1 and np.any(np.diff(tau) <= 0):
        raise PairedBundleError("tau must be strictly increasing")
    if np.any(stderr < 0):
        raise PairedBundleError("stderr must be non-negative")

    normalized_metadata: Dict[str, Any] = {}
    for key in REQUIRED_METADATA_KEYS:
        if key not in metadata:
            raise PairedBundleError(
                "missing required metadata key {!r}".format(key)
            )
    for key, value in metadata.items():
        normalized_metadata[str(key)] = _metadata_scalar(str(key), value)

    if normalized_metadata["format_version"] != FORMAT_VERSION:
        raise PairedBundleError(
            "unsupported format_version {!r}; expected {}".format(
                normalized_metadata["format_version"], FORMAT_VERSION
            )
        )
    if not str(normalized_metadata["observable"]).strip():
        raise PairedBundleError("metadata 'observable' must be non-empty")
    if not str(normalized_metadata["normalization"]).strip():
        raise PairedBundleError("metadata 'normalization' must be non-empty")
    for key in ("beta", "dt"):
        try:
            value = float(normalized_metadata[key])
        except (TypeError, ValueError):
            raise PairedBundleError(
                "metadata {!r} must be a positive finite scalar".format(key)
            )
        if not np.isfinite(value) or value <= 0:
            raise PairedBundleError(
                "metadata {!r} must be a positive finite scalar".format(key)
            )

    expected_mean = ratio_of_sums(
        numerator,
        sign,
        denominator_rtol=denominator_rtol,
        denominator_atol=denominator_atol,
    )
    if not np.allclose(
        mean,
        expected_mean,
        rtol=mean_rtol,
        atol=mean_atol,
        equal_nan=False,
    ):
        raise PairedBundleError(
            "mean is inconsistent with sum(numerator) / sum(sign)"
        )

    normalized_sources: Optional[np.ndarray]
    if source_files is None:
        normalized_sources = None
    else:
        normalized_sources = np.asarray(source_files)
        if normalized_sources.dtype.kind not in "SU":
            raise PairedBundleError("source_files must use a string dtype")
        if normalized_sources.shape != (nbin,):
            raise PairedBundleError(
                "source_files must have shape ({},), got {}".format(
                    nbin, normalized_sources.shape
                )
            )

    return PairedBundle(
        numerator=numerator,
        sign=sign,
        n_sample=n_sample,
        tau=tau,
        mean=mean,
        stderr=stderr,
        metadata=normalized_metadata,
        source_files=normalized_sources,
    )


def load_paired_bundle(
    path: Any,
    *,
    require_equal_n_sample: bool = True,
    denominator_rtol: float = DEFAULT_DENOMINATOR_RTOL,
    denominator_atol: float = 0.0,
) -> PairedBundle:
    """Load and validate a self-contained ``*_paired.npz`` file."""

    bundle_path = Path(path)
    if bundle_path.suffix != ".npz":
        raise PairedBundleError(
            "paired bundle must use the .npz suffix: {}".format(bundle_path)
        )
    if not bundle_path.is_file():
        raise FileNotFoundError(str(bundle_path))

    try:
        with np.load(str(bundle_path), allow_pickle=False) as archive:
            missing = [
                key
                for key in REQUIRED_ARRAY_KEYS + REQUIRED_METADATA_KEYS
                if key not in archive.files
            ]
            if missing:
                raise PairedBundleError(
                    "bundle {} is missing keys: {}".format(
                        bundle_path, ", ".join(missing)
                    )
                )

            metadata = {
                key: archive[key]
                for key in archive.files
                if key not in REQUIRED_ARRAY_KEYS and key != "source_files"
            }
            source_files = (
                archive["source_files"] if "source_files" in archive.files else None
            )
            return validate_paired_bundle(
                numerator=archive["numerator"],
                sign=archive["sign"],
                n_sample=archive["n_sample"],
                tau=archive["tau"],
                mean=archive["mean"],
                stderr=archive["stderr"],
                metadata=metadata,
                source_files=source_files,
                require_equal_n_sample=require_equal_n_sample,
                denominator_rtol=denominator_rtol,
                denominator_atol=denominator_atol,
            )
    except ValueError as exc:
        if isinstance(exc, PairedBundleError):
            raise
        raise PairedBundleError(
            "failed to load paired bundle {}: {}".format(bundle_path, exc)
        ) from exc


def save_paired_bundle(
    path: Any,
    bundle: PairedBundle,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and save a bundle without pickle-dependent fields."""

    bundle_path = Path(path)
    if bundle_path.suffix != ".npz":
        raise PairedBundleError(
            "paired bundle must use the .npz suffix: {}".format(bundle_path)
        )
    if bundle_path.exists() and not overwrite:
        raise FileExistsError(str(bundle_path))
    if not bundle_path.parent.is_dir():
        raise FileNotFoundError(str(bundle_path.parent))

    checked = validate_paired_bundle(
        numerator=bundle.numerator,
        sign=bundle.sign,
        n_sample=bundle.n_sample,
        tau=bundle.tau,
        mean=bundle.mean,
        stderr=bundle.stderr,
        metadata=bundle.metadata,
        source_files=bundle.source_files,
    )
    payload: Dict[str, Any] = {
        "numerator": checked.numerator,
        "sign": checked.sign,
        "n_sample": checked.n_sample,
        "tau": checked.tau,
        "mean": checked.mean,
        "stderr": checked.stderr,
    }
    payload.update(checked.metadata)
    if checked.source_files is not None:
        payload["source_files"] = checked.source_files
    np.savez_compressed(str(bundle_path), **payload)
    return bundle_path


def bootstrap_indices(
    nbin: int,
    n_resamples: int,
    *,
    sample_size: Optional[int] = None,
    block_size: int = 1,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate paired bootstrap row indices.

    ``block_size=1`` gives the ordinary nonparametric bootstrap.  Larger block
    sizes give a circular moving-block bootstrap while keeping every observable
    aligned through the returned index matrix.
    """

    if not isinstance(nbin, (int, np.integer)) or nbin < 2:
        raise ValueError("nbin must be an integer >= 2")
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be an integer >= 1")
    if sample_size is None:
        sample_size = int(nbin)
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 1:
        raise ValueError("sample_size must be an integer >= 1")
    if (
        not isinstance(block_size, (int, np.integer))
        or block_size < 1
        or block_size > nbin
    ):
        raise ValueError("block_size must be an integer in [1, nbin]")
    if seed is not None and rng is not None:
        raise ValueError("pass either seed or rng, not both")
    if rng is None:
        rng = np.random.default_rng(seed)
    if block_size == 1:
        return rng.integers(
            0,
            int(nbin),
            size=(int(n_resamples), int(sample_size)),
            dtype=np.int64,
        )

    n_blocks = (int(sample_size) + int(block_size) - 1) // int(block_size)
    starts = rng.integers(
        0,
        int(nbin),
        size=(int(n_resamples), n_blocks),
        dtype=np.int64,
    )
    offsets = np.arange(int(block_size), dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % int(nbin)
    return indices.reshape(int(n_resamples), -1)[:, :int(sample_size)]


def bootstrap_ratio_of_sums(
    numerator: np.ndarray,
    sign: np.ndarray,
    indices: np.ndarray,
    *,
    denominator_rtol: float = DEFAULT_DENOMINATOR_RTOL,
    denominator_atol: float = 0.0,
    chunk_size: int = 128,
) -> np.ndarray:
    """Evaluate paired ratio-of-sums estimates for pre-generated indices.

    Passing the same ``indices`` to several calls preserves correlations between
    observables measured in the same bins.
    """

    _validate_tolerances(denominator_rtol, denominator_atol)
    numerator = np.asarray(numerator)
    sign = np.asarray(sign)
    indices = np.asarray(indices)

    if numerator.ndim < 2:
        raise PairedBundleError(
            "numerator must have shape (Nbin, ...), got {}".format(
                numerator.shape
            )
        )
    nbin = numerator.shape[0]
    if sign.shape != (nbin,):
        raise PairedBundleError(
            "sign must have shape ({},), got {}".format(nbin, sign.shape)
        )
    _require_numeric("numerator", numerator, allow_complex=True)
    _require_numeric("sign", sign, allow_complex=True)

    if indices.ndim != 2 or indices.shape[0] == 0 or indices.shape[1] == 0:
        raise ValueError(
            "indices must have non-empty shape (Nresample, sample_size)"
        )
    if indices.dtype.kind not in "iu":
        raise ValueError("indices must have an integer dtype")
    if np.any(indices < 0) or np.any(indices >= nbin):
        raise IndexError("bootstrap indices are outside [0, Nbin)")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be an integer >= 1")

    n_resamples = indices.shape[0]
    output_shape = (n_resamples,) + numerator.shape[1:]
    output_dtype = np.result_type(numerator.dtype, sign.dtype, np.float64)
    estimates = np.empty(output_shape, dtype=output_dtype)
    numerator_flat = numerator.reshape(nbin, -1)

    for start in range(0, n_resamples, int(chunk_size)):
        stop = min(start + int(chunk_size), n_resamples)
        chunk_indices = indices[start:stop]
        counts = np.zeros((stop - start, nbin), dtype=np.int64)
        rows = np.repeat(np.arange(stop - start), chunk_indices.shape[1])
        np.add.at(counts, (rows, chunk_indices.reshape(-1)), 1)

        denominator = counts @ sign
        absolute_weight = counts @ np.abs(sign)
        bad = _denominator_is_small(
            denominator,
            absolute_weight,
            denominator_rtol,
            denominator_atol,
        )
        if np.any(bad):
            first = (np.flatnonzero(bad) + start)[:10]
            raise PairedBundleError(
                "bootstrap accumulated sign is too close to zero in "
                "replicate(s) {}".format(first.tolist())
            )

        numerator_sum = counts @ numerator_flat
        estimates[start:stop] = (
            numerator_sum / denominator[:, None]
        ).reshape((stop - start,) + numerator.shape[1:])

    return estimates


def bootstrap_covariance_rows(
    bootstrap_estimates: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Adapt bootstrap estimates to ``maxent.py``'s ``cov(rows)/Nrow`` API.

    The returned rows have mean ``center`` and satisfy
    ``cov(rows) / Nrow == cov(bootstrap_estimates)`` up to floating-point
    roundoff.  These are statistical surrogate rows, not Monte Carlo bins, and
    must not be used as input to nonlinear proxy calculations.
    """

    bootstrap_estimates = np.asarray(bootstrap_estimates)
    center = np.asarray(center)
    if bootstrap_estimates.ndim < 2:
        raise ValueError(
            "bootstrap_estimates must have shape (Nresample, ...)"
        )
    n_resamples = bootstrap_estimates.shape[0]
    if n_resamples < 2:
        raise ValueError("at least two bootstrap estimates are required")
    if center.shape != bootstrap_estimates.shape[1:]:
        raise ValueError(
            "center shape {} does not match estimate shape {}".format(
                center.shape, bootstrap_estimates.shape[1:]
            )
        )
    _require_numeric(
        "bootstrap_estimates", bootstrap_estimates, allow_complex=True
    )
    _require_numeric("center", center, allow_complex=True)

    bootstrap_mean = bootstrap_estimates.mean(axis=0)
    return center + np.sqrt(float(n_resamples)) * (
        bootstrap_estimates - bootstrap_mean
    )
