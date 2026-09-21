"""Focused C-source tests for attractive Hubbard HS-channel handling.

These tests implement C0, C2, and C3 from
``test/attractive_Hubbard_tests.md``.  They compile production source or
macros in temporary directories, do not modify ``src/``, do not create HDF5
files, and do not run a DQMC simulation.
"""

from __future__ import annotations

import shutil
import subprocess
import platform
import shlex
from pathlib import Path

import h5py
import pytest

import gen_1band_unified_hub as ghub
import gen_util_shared as gus


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _run_checked(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
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


def _production_target_flags():
    """Compile x86-only production headers as their intended target on ARM Macs."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ["-arch", "x86_64"]
    return []


def _write_linalg_declaration_stubs(include_dir):
    """Provide declarations needed for syntax-only generic BLAS compilation."""
    include_dir.mkdir()
    (include_dir / "cblas.h").write_text(
        """
#pragma once
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
void cblas_dgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE,
    enum CBLAS_TRANSPOSE, int, int, int, double, const double *, int,
    const double *, int, double, double *, int);
void cblas_zgemm(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE,
    enum CBLAS_TRANSPOSE, int, int, int, const void *, const void *, int,
    const void *, int, const void *, void *, int);
void cblas_dgemv(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, int, int, double,
    const double *, int, const double *, int, double, double *, int);
void cblas_zgemv(enum CBLAS_ORDER, enum CBLAS_TRANSPOSE, int, int,
    const void *, const void *, int, const void *, int, const void *,
    void *, int);
void cblas_dtrmm(enum CBLAS_ORDER, enum CBLAS_SIDE, enum CBLAS_UPLO,
    enum CBLAS_TRANSPOSE, enum CBLAS_DIAG, int, int, double,
    const double *, int, double *, int);
void cblas_ztrmm(enum CBLAS_ORDER, enum CBLAS_SIDE, enum CBLAS_UPLO,
    enum CBLAS_TRANSPOSE, enum CBLAS_DIAG, int, int, const void *,
    const void *, int, void *, int);
""".lstrip(),
        encoding="utf-8",
    )
    (include_dir / "lapacke.h").write_text(
        """
#pragma once
void LAPACK_dgetrf();
void LAPACK_zgetrf();
void LAPACK_dgetri();
void LAPACK_zgetri();
void LAPACK_dgetrs();
void LAPACK_zgetrs();
void LAPACK_dgeqp3();
void LAPACK_zgeqp3();
void LAPACK_dgeqrf();
void LAPACK_zgeqrf();
void LAPACK_dormqr();
void LAPACK_zunmqr();
void LAPACK_dtrtri();
void LAPACK_ztrtri();
""".lstrip(),
        encoding="utf-8",
    )


def _extract_function_like_macro(source, name):
    """Extract a complete multiline function-like macro from C source."""
    lines = source.splitlines()
    prefix = f"#define {name}("
    for start, line in enumerate(lines):
        if line.startswith(prefix):
            macro = [line]
            for continuation in lines[start + 1 :]:
                macro.append(continuation)
                if not continuation.rstrip().endswith("\\"):
                    return "\n".join(macro)
    raise AssertionError(f"macro {name} was not found or was incomplete")


def _actual_dqmc_propagator_macros():
    source = (SRC_DIR / "dqmc.c").read_text(encoding="utf-8")
    names = ("calcBu", "calcBd", "calciBu", "calciBd")
    return "\n\n".join(_extract_function_like_macro(source, name) for name in names)


def _macro_harness_source():
    macros = _actual_dqmc_propagator_macros()
    return f"""
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef double num;

{macros}

static int close_enough(const double actual, const double expected)
{{
    const double scale = fmax(1.0, fabs(expected));
    return fabs(actual - expected) <= 1e-12 * scale;
}}

static int require_close(
        const char *label, const int index,
        const double actual, const double expected)
{{
    if (close_enough(actual, expected)) return 0;
    fprintf(stderr, "%s[%d]: actual=%.17g expected=%.17g\\n",
            label, index, actual, expected);
    return 1;
}}

static int test_c2(void)
{{
    const int N = 2;
    const int hs[] = {{0, 1}};
    const double exp_lambda[] = {{2.0, 3.0, 5.0, 7.0}};
    const num exp_Ku[] = {{1.0, 3.0, 2.0, 4.0}};
    const num exp_Kd[] = {{1.0, 3.0, 2.0, 4.0}};
    num Bu[4] = {{0}};
    num Bd[4] = {{0}};
    int hs_channel = 1;
    int failed = 0;

    calcBu(Bu, 0)
    calcBd(Bd, 0)

    for (int j = 0; j < N; ++j) {{
        const int density_idx = hs[j];
        const double density_factor = exp_lambda[j + N*density_idx];
        for (int i = 0; i < N; ++i) {{
            const int ij = i + N*j;
            failed |= require_close(
                "density Bu", ij, Bu[ij], exp_Ku[ij] * density_factor);
            failed |= require_close(
                "density Bd", ij, Bd[ij], exp_Kd[ij] * density_factor);
            failed |= require_close("density up/down", ij, Bd[ij], Bu[ij]);
        }}
    }}

    hs_channel = 0;
    memset(Bd, 0, sizeof(Bd));
    calcBd(Bd, 0)
    for (int j = 0; j < N; ++j) {{
        const int spin_idx = hs[j] ^ 1;
        const double spin_factor = exp_lambda[j + N*spin_idx];
        for (int i = 0; i < N; ++i) {{
            const int ij = i + N*j;
            failed |= require_close(
                "spin contrast Bd", ij, Bd[ij], exp_Kd[ij] * spin_factor);
        }}
    }}

    return failed;
}}

static int test_c3(void)
{{
    const int N = 2;
    const int hs[] = {{0, 1}};
    const int hs_channel = 1;
    const double exp_lambda[] = {{0.5, 0.25, 2.0, 4.0}};
    const num exp_Kd[] = {{2.0, 1.0, 1.0, 1.0}};
    const num inv_exp_Kd[] = {{1.0, -1.0, -1.0, 2.0}};
    num Bd[4] = {{0}};
    num iBd[4] = {{0}};
    num product[4] = {{0}};
    int failed = 0;

    calcBd(Bd, 0)
    calciBd(iBd, 0)

    const double forward_factor[] = {{0.5, 4.0}};
    const double inverse_factor[] = {{2.0, 0.25}};
    for (int j = 0; j < N; ++j) {{
        for (int i = 0; i < N; ++i) {{
            const int ij = i + N*j;
            failed |= require_close(
                "Bd", ij, Bd[ij], exp_Kd[ij] * forward_factor[j]);
            failed |= require_close(
                "iBd", ij, iBd[ij], inverse_factor[i] * inv_exp_Kd[ij]);
        }}
    }}

    for (int j = 0; j < N; ++j) {{
        for (int i = 0; i < N; ++i) {{
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += iBd[i + N*k] * Bd[k + N*j];
            product[i + N*j] = sum;
            failed |= require_close(
                "iBd*Bd", i + N*j, sum, i == j ? 1.0 : 0.0);
        }}
    }}

    return failed;
}}

int main(int argc, char **argv)
{{
    if (argc != 2) return 64;
    if (strcmp(argv[1], "c2") == 0) return test_c2();
    if (strcmp(argv[1], "c3") == 0) return test_c3();
    return 64;
}}
""".lstrip()


@pytest.fixture(scope="module")
def propagator_macro_harness(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("attractive_macro_harness")
    source = build_dir / "attractive_dqmc_macro_harness.c"
    executable = build_dir / "attractive_dqmc_macro_harness"
    source.write_text(_macro_harness_source(), encoding="utf-8")
    _run_checked(
        [
            _clang(),
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-o",
            str(executable),
        ]
    )
    return executable


def test_c0_attractive_source_signatures_compile(tmp_path):
    """C0: the actual dqmc/update sources compile with matching signatures."""
    stub_include = tmp_path / "linalg_declarations"
    _write_linalg_declaration_stubs(stub_include)

    _run_checked(
        [
            _clang(),
            *_production_target_flags(),
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
            "-DGENERIC_LINALG",
            '-DGIT_ID="attractive-test"',
            '-DGIT_REPO="local-test"',
            f"-I{stub_include}",
            f"-I{SRC_DIR}",
            "-fsyntax-only",
            str(SRC_DIR / "updates.c"),
            str(SRC_DIR / "dqmc.c"),
        ]
    )

    signature_harness = tmp_path / "update_signature_harness.c"
    signature_harness.write_text(
        """
#include "updates.h"
int main(void)
{
    (void)update_delayed;
    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    linalg_symbols = tmp_path / "linalg_link_stubs.c"
    linalg_symbols.write_text(
        """
void cblas_dgemm(void) {}
void cblas_dgemv(void) {}
void cblas_dtrmm(void) {}
void cblas_zgemm(void) {}
void cblas_zgemv(void) {}
void cblas_ztrmm(void) {}
""".lstrip(),
        encoding="utf-8",
    )
    linked_harness = tmp_path / "update_signature_harness"
    _run_checked(
        [
            _clang(),
            *_production_target_flags(),
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-DGENERIC_LINALG",
            f"-I{stub_include}",
            f"-I{SRC_DIR}",
            str(signature_harness),
            str(SRC_DIR / "updates.c"),
            str(linalg_symbols),
            "-o",
            str(linked_harness),
        ]
    )
    assert linked_harness.is_file()


def test_c2_density_channel_uses_the_same_hs_index_for_both_spins(
    propagator_macro_harness,
):
    """C2: density channel uses hsbit for both up and down propagators."""
    _run_checked([str(propagator_macro_harness), "c2"])


def test_c3_down_spin_forward_and_inverse_propagators_are_inverses(
    propagator_macro_harness,
):
    """C3: calcBd/calciBd use complementary factors and multiply to I."""
    _run_checked([str(propagator_macro_harness), "c3"])


def _write_native_allocation_stub(include_dir):
    """Shadow the x86-only allocation header for native ARM test execution."""
    (include_dir / "xmmintrin.h").write_text(
        """
#pragma once
#include <stdlib.h>
static inline void *_mm_malloc(size_t size, size_t alignment)
{
    (void)alignment;
    return malloc(size);
}
static inline void _mm_free(void *pointer)
{
    free(pointer);
}
""".lstrip(),
        encoding="utf-8",
    )


def _numeric_blas_stub_source():
    """Small column-major BLAS implementation sufficient for update_delayed."""
    return """
#include <cblas.h>

void cblas_dgemm(
    enum CBLAS_ORDER order,
    enum CBLAS_TRANSPOSE trans_a,
    enum CBLAS_TRANSPOSE trans_b,
    int m, int n, int k,
    double alpha, const double *a, int lda,
    const double *b, int ldb,
    double beta, double *c, int ldc)
{
    (void)order;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int inner = 0; inner < k; ++inner) {
                const double av = trans_a == CblasNoTrans
                    ? a[row + lda*inner] : a[inner + lda*row];
                const double bv = trans_b == CblasNoTrans
                    ? b[inner + ldb*col] : b[col + ldb*inner];
                sum += av * bv;
            }
            c[row + ldc*col] = alpha*sum + beta*c[row + ldc*col];
        }
    }
}

void cblas_dgemv(
    enum CBLAS_ORDER order,
    enum CBLAS_TRANSPOSE trans,
    int m, int n,
    double alpha, const double *a, int lda,
    const double *x, int incx,
    double beta, double *y, int incy)
{
    (void)order;
    const int output_size = trans == CblasNoTrans ? m : n;
    const int inner_size = trans == CblasNoTrans ? n : m;
    for (int out = 0; out < output_size; ++out) {
        double sum = 0.0;
        for (int inner = 0; inner < inner_size; ++inner) {
            const double av = trans == CblasNoTrans
                ? a[out + lda*inner] : a[inner + lda*out];
            sum += av * x[inner*incx];
        }
        y[out*incy] = alpha*sum + beta*y[out*incy];
    }
}

void cblas_dtrmm(
    enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO uplo,
    enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag,
    int m, int n, double alpha, const double *a, int lda,
    double *b, int ldb)
{
    (void)order; (void)side; (void)uplo; (void)trans; (void)diag;
    (void)m; (void)n; (void)alpha; (void)a; (void)lda; (void)b; (void)ldb;
}

void cblas_zgemm(
    enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
    enum CBLAS_TRANSPOSE trans_b, int m, int n, int k,
    const void *alpha, const void *a, int lda, const void *b, int ldb,
    const void *beta, void *c, int ldc)
{
    (void)order; (void)trans_a; (void)trans_b; (void)m; (void)n; (void)k;
    (void)alpha; (void)a; (void)lda; (void)b; (void)ldb;
    (void)beta; (void)c; (void)ldc;
}

void cblas_zgemv(
    enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans, int m, int n,
    const void *alpha, const void *a, int lda, const void *x, int incx,
    const void *beta, void *y, int incy)
{
    (void)order; (void)trans; (void)m; (void)n; (void)alpha;
    (void)a; (void)lda; (void)x; (void)incx; (void)beta; (void)y; (void)incy;
}

void cblas_ztrmm(
    enum CBLAS_ORDER order, enum CBLAS_SIDE side, enum CBLAS_UPLO uplo,
    enum CBLAS_TRANSPOSE trans, enum CBLAS_DIAG diag, int m, int n,
    const void *alpha, const void *a, int lda, void *b, int ldb)
{
    (void)order; (void)side; (void)uplo; (void)trans; (void)diag;
    (void)m; (void)n; (void)alpha; (void)a; (void)lda; (void)b; (void)ldb;
}
""".lstrip()


def _update_delayed_harness_source():
    return r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rand.h"
#include "updates.h"

static int close_enough(double actual, double expected)
{
    const double scale = fmax(1.0, fabs(expected));
    return fabs(actual - expected) <= 1e-12 * scale;
}

static int require_close(
        const char *label, int index, double actual, double expected)
{
    if (close_enough(actual, expected)) return 0;
    fprintf(stderr, "%s[%d]: actual=%.17g expected=%.17g\n",
            label, index, actual, expected);
    return 1;
}

static int require_int(const char *label, int actual, int expected)
{
    if (actual == expected) return 0;
    fprintf(stderr, "%s: actual=%d expected=%d\n", label, actual, expected);
    return 1;
}

static void init_rng(uint64_t rng[17])
{
    for (int i = 0; i < 16; ++i)
        rng[i] = UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(i + 1);
    rng[16] = 0;
}

static int require_rng(const uint64_t actual[17], const uint64_t expected[17])
{
    for (int i = 0; i < 17; ++i) {
        if (actual[i] != expected[i]) {
            fprintf(stderr, "rng[%d]: actual=%llu expected=%llu\n", i,
                    (unsigned long long)actual[i],
                    (unsigned long long)expected[i]);
            return 1;
        }
    }
    return 0;
}

static double one_site_expected_green(double green, double delta)
{
    const double ratio = 1.0 + (1.0 - green) * delta;
    return green + (green - 1.0) * (delta / ratio) * green;
}

static int test_c4(void)
{
    const int N = 1;
    const int n_delay = 1;
    const int site_order[1] = {0};
    const int hs_channel = 1;
    const double delta_current = 1.0;
    const double gu_initial = 0.4;
    const double gd_initial = 0.6;
    const double ru = 1.0 + (1.0 - gu_initial) * delta_current;
    const double rd = 1.0 + (1.0 - gd_initial) * delta_current;
    const double determinant_ratio = ru * rd;
    int failed = 0;

    uint64_t rng_for_draw[17];
    init_rng(rng_for_draw);
    const double draw = rand_doub(rng_for_draw);

    {
        uint64_t rng[17];
        uint64_t expected_rng[17];
        init_rng(rng);
        memcpy(expected_rng, rng, sizeof(rng));
        (void)rand_doub(expected_rng);

        int hs[1] = {0};
        double gu[1] = {gu_initial};
        double gd[1] = {gd_initial};
        double phase = 1.0;
        double au[1] = {0}, bu[1] = {0}, du[1] = {0};
        double ad[1] = {0}, bd[1] = {0}, dd[1] = {0};
        double del[2] = {delta_current, -0.5};
        const double target_probability = 0.5 * draw;
        const double el = sqrt(target_probability / determinant_ratio);
        double exp_lambda[2] = {el, el == 0.0 ? 1.0 : 1.0/el};

        update_delayed(N, n_delay, del, exp_lambda, hs_channel,
            site_order, rng, hs, gu, gd, &phase,
            au, bu, du, ad, bd, dd);

        failed |= require_int("C4 rejected hs", hs[0], 0);
        failed |= require_close("C4 rejected gu", 0, gu[0], gu_initial);
        failed |= require_close("C4 rejected gd", 0, gd[0], gd_initial);
        failed |= require_close("C4 rejected phase", 0, phase, 1.0);
        failed |= require_rng(rng, expected_rng);
    }

    {
        uint64_t rng[17];
        uint64_t expected_rng[17];
        init_rng(rng);
        memcpy(expected_rng, rng, sizeof(rng));
        (void)rand_doub(expected_rng);

        int hs[1] = {0};
        double gu[1] = {gu_initial};
        double gd[1] = {gd_initial};
        double phase = 1.0;
        double au[1] = {0}, bu[1] = {0}, du[1] = {0};
        double ad[1] = {0}, bd[1] = {0}, dd[1] = {0};
        double del[2] = {delta_current, -0.5};
        const double target_probability = 0.5 * (draw + 1.0);
        const double el = sqrt(target_probability / determinant_ratio);
        double exp_lambda[2] = {el, 1.0/el};

        update_delayed(N, n_delay, del, exp_lambda, hs_channel,
            site_order, rng, hs, gu, gd, &phase,
            au, bu, du, ad, bd, dd);

        failed |= require_int("C4 accepted hs", hs[0], 1);
        failed |= require_close("C4 accepted gu", 0, gu[0],
            one_site_expected_green(gu_initial, delta_current));
        failed |= require_close("C4 accepted gd", 0, gd[0],
            one_site_expected_green(gd_initial, delta_current));
        failed |= require_close("C4 accepted phase", 0, phase, 1.0);
        failed |= require_rng(rng, expected_rng);
    }

    return failed;
}

static void immediate_rank_one_update(
        int N, double *green, int site, double delta)
{
    double old[4];
    memcpy(old, green, (size_t)(N*N) * sizeof(double));
    const double ratio = 1.0 + (1.0 - old[site + N*site]) * delta;
    for (int col = 0; col < N; ++col) {
        for (int row = 0; row < N; ++row) {
            const double left = old[row + N*site] - (row == site ? 1.0 : 0.0);
            const double right = old[site + N*col];
            green[row + N*col] = old[row + N*col]
                + left * (delta / ratio) * right;
        }
    }
}

static int test_c5(int n_delay)
{
    const int N = 2;
    const int site_order[2] = {0, 1};
    const int hs_channel = 1;
    const double del[4] = {0.5, 0.8, -0.25, -0.4};
    const double exp_lambda[4] = {2.0, 2.0, 0.5, 0.5};
    const double gu_initial[4] = {0.4, 0.2, 0.1, 0.6};
    const double gd_initial[4] = {0.55, 0.15, -0.05, 0.45};
    double expected_gu[4];
    double expected_gd[4];
    double gu[4];
    double gd[4];
    memcpy(expected_gu, gu_initial, sizeof(gu));
    memcpy(expected_gd, gd_initial, sizeof(gd));
    memcpy(gu, gu_initial, sizeof(gu));
    memcpy(gd, gd_initial, sizeof(gd));

    immediate_rank_one_update(N, expected_gu, 0, del[0]);
    immediate_rank_one_update(N, expected_gd, 0, del[0]);
    immediate_rank_one_update(N, expected_gu, 1, del[1]);
    immediate_rank_one_update(N, expected_gd, 1, del[1]);

    uint64_t rng[17];
    uint64_t expected_rng[17];
    init_rng(rng);
    memcpy(expected_rng, rng, sizeof(rng));
    (void)rand_doub(expected_rng);
    (void)rand_doub(expected_rng);

    int hs[2] = {0, 0};
    double phase = 1.0;
    double au[4] = {0}, bu[4] = {0}, du[2] = {0};
    double ad[4] = {0}, bd[4] = {0}, dd[2] = {0};

    update_delayed(N, n_delay, del, exp_lambda, hs_channel,
        site_order, rng, hs, gu, gd, &phase,
        au, bu, du, ad, bd, dd);

    int failed = 0;
    failed |= require_int("C5 hs[0]", hs[0], 1);
    failed |= require_int("C5 hs[1]", hs[1], 1);
    failed |= require_close("C5 phase", 0, phase, 1.0);
    failed |= require_rng(rng, expected_rng);
    for (int index = 0; index < N*N; ++index) {
        failed |= require_close("C5 gu", index, gu[index], expected_gu[index]);
        failed |= require_close("C5 gd", index, gd[index], expected_gd[index]);
    }
    return failed;
}

int main(int argc, char **argv)
{
    if (argc != 2) return 64;
    if (strcmp(argv[1], "c4") == 0) return test_c4();
    if (strcmp(argv[1], "c5-1") == 0) return test_c5(1);
    if (strcmp(argv[1], "c5-2") == 0) return test_c5(2);
    return 64;
}
""".lstrip()


@pytest.fixture(scope="module")
def update_delayed_harness(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("attractive_update_harness")
    include_dir = build_dir / "test_includes"
    _write_linalg_declaration_stubs(include_dir)
    _write_native_allocation_stub(include_dir)

    harness_source = build_dir / "attractive_update_harness.c"
    blas_source = build_dir / "numeric_blas_stub.c"
    executable = build_dir / "attractive_update_harness"
    harness_source.write_text(_update_delayed_harness_source(), encoding="utf-8")
    blas_source.write_text(_numeric_blas_stub_source(), encoding="utf-8")

    _run_checked(
        [
            _clang(),
            "-std=gnu11",
            "-Wall",
            "-Wextra",
            "-Werror=implicit-function-declaration",
            "-DGENERIC_LINALG",
            f"-I{include_dir}",
            f"-I{SRC_DIR}",
            str(harness_source),
            str(SRC_DIR / "updates.c"),
            str(blas_source),
            "-lm",
            "-o",
            str(executable),
        ]
    )
    return executable


def test_c4_density_channel_local_update_acceptance_and_rejection(
    update_delayed_harness,
):
    """C4: density weight controls rejection/acceptance and one-site update."""
    _run_checked([str(update_delayed_harness), "c4"])


@pytest.mark.parametrize("n_delay", [1, 2])
def test_c5_delayed_update_matches_immediate_rank_one_updates(
    update_delayed_harness, n_delay
):
    """C5: delayed batches agree with independent immediate rank-one updates."""
    _run_checked([str(update_delayed_harness), f"c5-{n_delay}"])


def _data_read_harness_source():
    return r"""
#include <math.h>
#include <stdio.h>

#include "data.h"

static int close_enough(double actual, double expected)
{
    const double scale = fmax(1.0, fabs(expected));
    return fabs(actual - expected) <= 1e-12 * scale;
}

static int require_int(const char *label, int actual, int expected)
{
    if (actual == expected) return 0;
    fprintf(stderr, "%s: actual=%d expected=%d\n", label, actual, expected);
    return 1;
}

static int require_close(
        const char *label, int index, double actual, double expected)
{
    if (close_enough(actual, expected)) return 0;
    fprintf(stderr, "%s[%d]: actual=%.17g expected=%.17g\n",
            label, index, actual, expected);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 2) return 64;
    if (set_num_h5t() != 0) {
        fprintf(stderr, "set_num_h5t failed\n");
        return 65;
    }

    struct sim_data sim = {0};
    sim.file = argv[1];
    const int read_status = sim_data_read_alloc(&sim);
    if (read_status != 0) {
        fprintf(stderr, "sim_data_read_alloc returned %d\n", read_status);
        return 66;
    }

    int failed = 0;
    failed |= require_int("hs_channel", sim.p.hs_channel, 1);
    failed |= require_int("N", sim.p.N, 4);
    failed |= require_int("L", sim.p.L, 4);
    failed |= require_int("Nx", sim.p.Nx, 2);
    failed |= require_int("Ny", sim.p.Ny, 2);
    failed |= require_int("num_i", sim.p.num_i, 1);
    failed |= require_int("n_sweep", sim.p.n_sweep, 0);
    failed |= require_int("state sweep", sim.s.sweep, 0);

    if (sim.p.exp_lambda == NULL || sim.p.del == NULL || sim.s.hs == NULL) {
        fprintf(stderr, "required allocated array is NULL\n");
        failed = 1;
    } else {
        const double lambda = acosh(exp(0.5 * 0.1 * 4.0));
        const double exp_positive = exp(lambda);
        const double expected_exp_negative = 1.0 / exp_positive;
        const double expected_delta_positive = exp_positive*exp_positive - 1.0;
        const double expected_delta_negative =
            1.0/(exp_positive*exp_positive) - 1.0;

        for (int site = 0; site < sim.p.N; ++site) {
            failed |= require_close("exp_lambda negative", site,
                sim.p.exp_lambda[site], expected_exp_negative);
            failed |= require_close("exp_lambda positive", site,
                sim.p.exp_lambda[site + sim.p.N], exp_positive);
            failed |= require_close("delta positive", site,
                sim.p.del[site], expected_delta_positive);
            failed |= require_close("delta negative", site,
                sim.p.del[site + sim.p.N], expected_delta_negative);
        }
        for (int index = 0; index < sim.p.N * sim.p.L; ++index) {
            if (sim.s.hs[index] != 0 && sim.s.hs[index] != 1) {
                fprintf(stderr, "hs[%d]=%d is not binary\n", index, sim.s.hs[index]);
                failed = 1;
            }
        }
    }

    sim_data_free(&sim);
    if (failed) return 1;
    puts("C1_OK");
    return 0;
}
""".lstrip()


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


def _unused_greens_workspace_stubs_source():
    """Resolve data.c workspace helpers not reached by the C1 read path."""
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


def test_c1_data_c_reads_generated_attractive_density_channel(tmp_path):
    """C1: real data.c reads density-channel values from generated HDF5."""
    h5_path = tmp_path / "c1_square_attractive.h5"
    ghub.create_1(
        file_sim=h5_path,
        file_params=h5_path,
        init_rng=gus.rand_seed_splitmix64(1234),
        geometry="square",
        Nx=2,
        Ny=2,
        trans_sym=1,
        U=-4.0,
        hs_channel="auto",
        dt=0.1,
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

    build_dir = tmp_path / "c1_build"
    build_dir.mkdir()
    include_dir = build_dir / "test_includes"
    include_dir.mkdir()
    _write_native_allocation_stub(include_dir)
    harness_source = build_dir / "attractive_data_read_harness.c"
    greens_stubs_source = build_dir / "unused_greens_workspace_stubs.c"
    executable = build_dir / "attractive_data_read_harness"
    harness_source.write_text(_data_read_harness_source(), encoding="utf-8")
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
            '-DGIT_ID="attractive-test"',
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

    result = _run_checked([str(executable), str(h5_path)])
    assert "C1_OK" in result.stdout
    assert "defaulting hs_channel=0" not in result.stderr

    with h5py.File(h5_path, "r") as h5:
        assert h5["params/hs_channel"][()] == 1
        assert h5["state/sweep"][()] == 0
        assert h5["params/n_sweep"][()] == 0
