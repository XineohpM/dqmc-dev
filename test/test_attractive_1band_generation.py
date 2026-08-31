"""Focused unit tests for attractive single-band Hubbard generation.

These tests implement cases A1, A2, B1, B2, B3, B4, and B5 from
``test/attractive_Hubbard_tests.md``.  They do not run a DQMC simulation and
do not exercise any three-band generator.
"""

from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np
import pytest

import gen_1band_unified_hub as ghub
import gen_util_shared as gus


RTOL = 1e-12
ATOL = 1e-14
SEED = 1234


def _independent_hs_parameters(U_i, dt, map_i):
    """Return the density-channel HS parameters from the defining equation."""
    x = 0.5 * dt * np.abs(U_i)
    # cosh(lambda) = exp(x).  The equivalent asinh/expm1 form avoids the
    # loss of precision in arccosh(exp(x)) when the attraction is very weak.
    lambdas = 2.0 * np.arcsinh(np.sqrt(0.5 * np.expm1(x)))
    exp_positive_lambda = np.exp(lambdas)
    mapped = exp_positive_lambda[map_i]
    exp_lambda = np.stack((mapped**-1, mapped))
    delta = np.stack((mapped**2 - 1.0, mapped**-2 - 1.0))
    return exp_lambda, delta


def _create_square_hdf5(
    path,
    *,
    hs_channel="auto",
    Nx=2,
    Ny=2,
    trans_sym=1,
    U=-4.0,
    dt=0.1,
    seed=SEED,
):
    """Generate a minimal, unrun square-lattice file with a fixed seed."""
    ghub.create_1(
        file_sim=path,
        file_params=path,
        init_rng=gus.rand_seed_splitmix64(seed),
        geometry="square",
        Nx=Nx,
        Ny=Ny,
        trans_sym=trans_sym,
        U=U,
        hs_channel=hs_channel,
        dt=dt,
        L=4,
        n_delay=1,
        n_matmul=2,
        n_sweep_warm=0,
        n_sweep_meas=0,
        period_eqlt=2,
        period_uneqlt=0,
        checkpoint_every=0,
        overwrite=1,
    )


def _read_h5_string(dataset):
    return dataset.asstr()[()]


@pytest.mark.parametrize("U", [-2.0, -4.0, -8.0])
@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
def test_a1_attractive_hs_parameters_match_independent_formula(U, dt):
    """A1: negative-U HS parameters satisfy the density-channel formula."""
    degen_i = np.array([2, 3, 1], dtype=np.int32)
    map_i = np.array([0, 1, 2, 0, 2, 1], dtype=np.int32)
    num_i = degen_i.size

    U_i, exp_lambda, delta = gus.set_U(U, dt, num_i, map_i, degen_i)
    expected_exp_lambda, expected_delta = _independent_hs_parameters(
        U_i, dt, map_i
    )

    assert U_i.shape == (num_i,)
    assert exp_lambda.shape == delta.shape == (2, map_i.size)
    assert U_i.dtype == exp_lambda.dtype == delta.dtype == np.float64
    assert np.all(U_i == U)
    assert np.all(U_i < 0.0)
    assert np.all(np.isfinite(exp_lambda))
    assert np.all(np.isfinite(delta))

    np.testing.assert_allclose(
        exp_lambda, expected_exp_lambda, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(delta, expected_delta, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        exp_lambda[0] * exp_lambda[1], 1.0, rtol=RTOL, atol=ATOL
    )


def test_a2_zero_interaction_is_identity_hs_transform():
    """A2: U=0 is the finite, identity-valued HS boundary."""
    degen_i = np.array([1, 2, 1], dtype=np.int32)
    map_i = np.array([2, 0, 1, 2, 1], dtype=np.int32)

    U_i, exp_lambda, delta = gus.set_U(
        0.0, 0.1, degen_i.size, map_i, degen_i
    )

    np.testing.assert_array_equal(U_i, np.zeros(degen_i.size, dtype=np.float64))
    np.testing.assert_array_equal(
        exp_lambda, np.ones((2, map_i.size), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        delta, np.zeros((2, map_i.size), dtype=np.float64)
    )


@pytest.mark.parametrize("U", [-1e-12, -1e-9, -1e-6])
def test_a2_weak_attraction_is_finite_reciprocal_and_well_mapped(U):
    """A2: weak attraction remains finite and respects repeated site maps."""
    dt = 0.1
    degen_i = np.array([3, 2, 4], dtype=np.int32)
    map_i = np.array([2, 0, 2, 1, 0, 1, 2], dtype=np.int32)

    U_i, exp_lambda, delta = gus.set_U(
        U, dt, degen_i.size, map_i, degen_i
    )
    expected_exp_lambda, expected_delta = _independent_hs_parameters(
        U_i, dt, map_i
    )

    assert np.all(np.isfinite(exp_lambda))
    assert np.all(np.isfinite(delta))
    np.testing.assert_allclose(
        exp_lambda, expected_exp_lambda, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(delta, expected_delta, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        exp_lambda[0] * exp_lambda[1], 1.0, rtol=RTOL, atol=ATOL
    )

    for site_class in range(degen_i.size):
        sites = np.flatnonzero(map_i == site_class)
        np.testing.assert_array_equal(
            exp_lambda[:, sites],
            np.repeat(exp_lambda[:, sites[:1]], sites.size, axis=1),
        )
        np.testing.assert_array_equal(
            delta[:, sites],
            np.repeat(delta[:, sites[:1]], sites.size, axis=1),
        )


def test_a2_rejects_num_i_inconsistent_with_degeneracy_array():
    """A2: an inconsistent number of inequivalent sites is rejected."""
    degen_i = np.array([1, 1], dtype=np.int32)
    map_i = np.array([0, 1], dtype=np.int32)

    with pytest.raises(AssertionError):
        gus.set_U(-4.0, 0.1, 3, map_i, degen_i)


@pytest.mark.parametrize(
    "invalid_channel",
    ["", "charge", "Density", None, 1],
    ids=["empty", "charge", "wrong-case", "none", "integer"],
)
def test_b3_invalid_hs_channel_is_rejected_without_creating_hdf5(
    tmp_path, invalid_channel
):
    """B3: invalid channel input fails before any HDF5 file is created."""
    output = tmp_path / "invalid_channel.h5"
    init_rng = gus.rand_seed_splitmix64(1234)

    with pytest.raises(ValueError) as exc_info:
        ghub.create_1(
            file_sim=output,
            file_params=output,
            init_rng=init_rng,
            geometry="square",
            Nx=2,
            Ny=2,
            U=-4.0,
            hs_channel=invalid_channel,
            L=4,
            n_delay=1,
            n_matmul=2,
            n_sweep_warm=0,
            n_sweep_meas=0,
            period_eqlt=2,
            overwrite=1,
        )

    message = str(exc_info.value)
    assert "Invalid hs_channel=" in message
    assert "auto" in message
    assert "spin" in message
    assert "density" in message
    assert not output.exists()
    assert list(tmp_path.glob("*.h5*")) == []


def test_b1_auto_channel_generates_valid_attractive_square_hdf5(tmp_path):
    """B1: auto selects density and writes correct attractive HDF5 values."""
    output = tmp_path / "square_auto.h5"
    _create_square_hdf5(output, hs_channel="auto")

    assert output.is_file()
    with h5py.File(output, "r") as h5:
        assert _read_h5_string(h5["metadata/geometry"]) == "square"
        assert _read_h5_string(h5["metadata/hs_channel"]) == "density"
        assert h5["metadata/U"][()] == pytest.approx(-4.0)

        hs_channel = h5["params/hs_channel"]
        assert hs_channel.shape == ()
        assert hs_channel.dtype == np.dtype(np.int32)
        assert hs_channel[()] == 1

        U_i = h5["params/U"][...]
        map_i = h5["params/map_i"][...]
        dt = h5["params/dt"][()]
        exp_lambda = h5["params/exp_lambda"][...]
        delta = h5["params/del"][...]
        expected_exp_lambda, expected_delta = _independent_hs_parameters(
            U_i, dt, map_i
        )

        assert np.all(U_i == -4.0)
        assert exp_lambda.shape == delta.shape == (2, 4)
        assert np.all(np.isfinite(exp_lambda))
        assert np.all(np.isfinite(delta))
        np.testing.assert_allclose(
            exp_lambda, expected_exp_lambda, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            delta, expected_delta, rtol=RTOL, atol=ATOL
        )

        initial_seed = gus.rand_seed_splitmix64(SEED)
        np.testing.assert_array_equal(h5["state/init_rng"][...], initial_seed)
        hs = h5["state/hs"][...]
        assert hs.shape == (4, 4)
        assert hs.dtype == np.dtype(np.int32)
        assert set(np.unique(hs)).issubset({0, 1})
        assert h5["state/sweep"][()] == 0
        assert h5["params/n_sweep"][()] == 0


def test_b2_auto_and_explicit_density_generate_identical_relevant_data(tmp_path):
    """B2: auto and explicit density are equivalent for negative U."""
    auto_path = tmp_path / "square_auto.h5"
    density_path = tmp_path / "square_density.h5"
    _create_square_hdf5(auto_path, hs_channel="auto")
    _create_square_hdf5(density_path, hs_channel="density")

    array_paths = (
        "params/hs_channel",
        "params/U",
        "params/map_i",
        "params/exp_lambda",
        "params/del",
        "params/exp_Ku",
        "params/exp_Kd",
        "params/inv_exp_Ku",
        "params/inv_exp_Kd",
        "state/init_rng",
        "state/rng",
        "state/hs",
    )
    with h5py.File(auto_path, "r") as auto_h5, h5py.File(
        density_path, "r"
    ) as density_h5:
        assert _read_h5_string(auto_h5["metadata/hs_channel"]) == "density"
        assert _read_h5_string(density_h5["metadata/hs_channel"]) == "density"
        for dataset_path in array_paths:
            np.testing.assert_array_equal(
                auto_h5[dataset_path][...], density_h5[dataset_path][...]
            )


@pytest.mark.parametrize(
    "Nx,Ny,trans_sym",
    [(2, 2, 1), (3, 2, 0)],
    ids=["2x2-translational", "3x2-uncompressed"],
)
def test_b4_square_lattice_attractive_parameter_shapes_and_mapping(
    tmp_path, Nx, Ny, trans_sym
):
    """B4: square-lattice mappings preserve attractive parameter shapes."""
    output = tmp_path / f"square_{Nx}x{Ny}_trans{trans_sym}.h5"
    _create_square_hdf5(
        output,
        hs_channel="auto",
        Nx=Nx,
        Ny=Ny,
        trans_sym=trans_sym,
    )

    with h5py.File(output, "r") as h5:
        N = Nx * Ny
        num_i = int(h5["params/num_i"][()])
        map_i = h5["params/map_i"][...]
        U_i = h5["params/U"][...]
        exp_lambda = h5["params/exp_lambda"][...]
        delta = h5["params/del"][...]

        assert _read_h5_string(h5["metadata/geometry"]) == "square"
        assert _read_h5_string(h5["metadata/hs_channel"]) == "density"
        assert h5["params/hs_channel"][()] == 1
        assert h5["params/N"][()] == N
        assert map_i.shape == (N,)
        assert U_i.shape == (num_i,)
        assert exp_lambda.shape == delta.shape == (2, N)
        assert np.all((0 <= map_i) & (map_i < num_i))
        assert np.all(U_i < 0.0)
        assert np.all(np.isfinite(exp_lambda))
        assert np.all(np.isfinite(delta))

        if trans_sym:
            assert num_i == 1
            np.testing.assert_array_equal(map_i, np.zeros(N, dtype=np.int32))
        else:
            assert num_i == N
            np.testing.assert_array_equal(map_i, np.arange(N, dtype=np.int32))


def test_b5_cli_generates_unrun_attractive_square_hdf5(tmp_path):
    """B5: the CLI writes linked square files without starting a simulation."""
    script = ghub.__file__
    prefix = "cli_square"
    command = [
        sys.executable,
        script,
        f"--prefix={prefix}",
        "--Nfiles=1",
        f"--seed={SEED}",
        "--overwrite=1",
        "--printout=0",
        "--geometry=square",
        "--Nx=2",
        "--Ny=2",
        "--U=-4",
        "--hs_channel=auto",
        "--dt=0.1",
        "--L=4",
        "--n_delay=1",
        "--n_matmul=2",
        "--n_sweep_warm=0",
        "--n_sweep_meas=0",
        "--period_eqlt=2",
        "--period_uneqlt=0",
        "--checkpoint_every=0",
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    sim_path = tmp_path / f"{prefix}_0.h5"
    params_path = tmp_path / f"{prefix}.h5.params"
    assert sim_path.is_file()
    assert params_path.is_file()
    assert "created simulation files:" in result.stdout

    with h5py.File(sim_path, "r") as sim_h5:
        assert _read_h5_string(sim_h5["metadata/geometry"]) == "square"
        assert _read_h5_string(sim_h5["metadata/hs_channel"]) == "density"
        assert sim_h5["params/hs_channel"][()] == 1
        assert sim_h5["metadata/U"][()] == pytest.approx(-4.0)
        assert sim_h5["state/sweep"][()] == 0
        assert sim_h5["params/n_sweep"][()] == 0
        assert sim_h5["state/hs"].shape == (4, 4)

    generated = sorted(path.name for path in tmp_path.glob("*.h5*"))
    assert generated == [f"{prefix}.h5.params", f"{prefix}_0.h5"]
    assert not (tmp_path / "stack").exists()
    assert list(tmp_path.glob("*.log")) == []
