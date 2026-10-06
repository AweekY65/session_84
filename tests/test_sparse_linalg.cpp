// Terminal test suite for the local sparse linear algebra module.
// Every test prints its outcome; solver tests also print the final residual.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "sparse_linalg/cg_solver.hpp"
#include "sparse_linalg/csr_matrix.hpp"

using sparse_linalg::CgOptions;
using sparse_linalg::CgStatus;
using sparse_linalg::CsrMatrix;
using sparse_linalg::Triplet;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& name) {
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (!ok) ++g_failures;
}

bool nearly_equal(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * std::max({1.0, std::fabs(a), std::fabs(b)});
}

// Dense reference matvec, used only on small test problems.
std::vector<double> dense_matvec(const std::vector<std::vector<double>>& A,
                                 const std::vector<double>& x) {
    std::vector<double> y(A.size(), 0.0);
    for (std::size_t i = 0; i < A.size(); ++i)
        for (std::size_t j = 0; j < x.size(); ++j) y[i] += A[i][j] * x[j];
    return y;
}

void test_csr_layout() {
    // 3x4 matrix with unsorted triplets and an empty middle row.
    std::vector<Triplet> t = {{2, 3, 7.0}, {0, 2, 1.0}, {2, 0, 5.0},
                              {0, 0, 2.0}, {2, 1, 6.0}, {0, 1, 3.0}};
    auto A = CsrMatrix::from_triplets(3, 4, t);

    check(A.rows() == 3 && A.cols() == 4, "csr_layout: dims");
    check(A.nnz() == 6, "csr_layout: nnz");
    check(A.row_ptr() == std::vector<std::size_t>({0, 3, 3, 6}),
          "csr_layout: row_ptr (empty row)");
    check(A.col_idx() == std::vector<std::size_t>({0, 1, 2, 0, 1, 3}),
          "csr_layout: col_idx sorted per row");
    check(A.values() == std::vector<double>({2.0, 3.0, 1.0, 5.0, 6.0, 7.0}),
          "csr_layout: values");
}

void test_duplicate_merge() {
    std::vector<Triplet> t = {{0, 0, 1.0}, {0, 0, 2.5}, {1, 1, 4.0},
                              {0, 1, 1.0}, {0, 1, -1.0}, {1, 0, 3.0}};
    auto A = CsrMatrix::from_triplets(2, 2, t);

    check(A.nnz() == 4, "duplicates: merged nnz");
    check(A.col_idx() == std::vector<std::size_t>({0, 1, 0, 1}),
          "duplicates: col_idx");
    check(A.values() == std::vector<double>({3.5, 0.0, 3.0, 4.0}),
          "duplicates: summed values (incl. cancellation)");
    check(A.row_ptr() == std::vector<std::size_t>({0, 2, 4}),
          "duplicates: row_ptr");
}

void test_matvec_matches_dense() {
    const std::vector<std::vector<double>> dense = {
        {4.0, 0.0, 1.0, 0.0},
        {0.0, 3.0, 0.0, 2.0},
        {1.0, 0.0, 5.0, 0.0},
        {0.0, 2.0, 0.0, 6.0}};
    std::vector<Triplet> t;
    for (std::size_t i = 0; i < dense.size(); ++i)
        for (std::size_t j = 0; j < dense[i].size(); ++j)
            if (dense[i][j] != 0.0) t.push_back({i, j, dense[i][j]});
    auto A = CsrMatrix::from_triplets(4, 4, t);

    std::vector<double> x = {1.0, -2.0, 0.5, 3.0};
    auto y_sparse = A.matvec(x);
    auto y_dense = dense_matvec(dense, x);

    bool ok = y_sparse.size() == y_dense.size();
    for (std::size_t i = 0; ok && i < y_dense.size(); ++i)
        ok = nearly_equal(y_sparse[i], y_dense[i], 1e-12);
    check(ok, "matvec: matches dense reference");

    bool threw = false;
    try {
        A.matvec(std::vector<double>{1.0, 2.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "matvec: dimension mismatch throws");
}

CsrMatrix poisson_1d(std::size_t n) {
    std::vector<Triplet> t;
    t.reserve(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) t.push_back({i, i - 1, -1.0});
        t.push_back({i, i, 2.0});
        if (i + 1 < n) t.push_back({i, i + 1, -1.0});
    }
    return CsrMatrix::from_triplets(n, n, t);
}

void test_known_system() {
    const std::size_t n = 50;
    auto A = poisson_1d(n);
    std::vector<double> x_exact(n);
    for (std::size_t i = 0; i < n; ++i) x_exact[i] = std::sin(0.1 * i) + 1.0;
    auto b = A.matvec(x_exact);

    CgOptions opts;
    opts.tolerance = 1e-12;
    opts.max_iterations = 500;
    auto res = sparse_linalg::cg_solve(A, b, opts);

    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        err = std::max(err, std::fabs(res.solution[i] - x_exact[i]));
    std::printf("       known_system: iters=%zu residual=%.3e max_err=%.3e\n",
                res.iterations, res.final_residual, err);
    check(res.converged && res.status == CgStatus::Converged,
          "known_system: converged");
    check(err < 1e-6, "known_system: solution accurate");
}

void test_preconditioner_effect() {
    // Badly scaled SPD: D * P * D with D a wide-range diagonal.
    const std::size_t n = 80;
    auto P = poisson_1d(n);
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = 1.0 + 0.25 * i;

    std::vector<Triplet> t;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = P.row_ptr()[i]; k < P.row_ptr()[i + 1]; ++k)
            t.push_back({i, P.col_idx()[k], d[i] * P.values()[k] * d[P.col_idx()[k]]});
    auto A = CsrMatrix::from_triplets(n, n, t);

    std::vector<double> b(n, 1.0);
    CgOptions opts;
    opts.tolerance = 1e-10;
    opts.max_iterations = 5000;

    opts.use_jacobi_preconditioner = false;
    auto plain = sparse_linalg::cg_solve(A, b, opts);
    opts.use_jacobi_preconditioner = true;
    auto jacobi = sparse_linalg::cg_solve(A, b, opts);

    std::printf("       jacobi: plain_iters=%zu (res=%.3e) jacobi_iters=%zu (res=%.3e)\n",
                plain.iterations, plain.final_residual, jacobi.iterations,
                jacobi.final_residual);
    check(plain.converged && jacobi.converged, "jacobi: both converged");
    check(jacobi.iterations < plain.iterations,
          "jacobi: fewer iterations than unpreconditioned");
}

void test_zero_matrix() {
    auto A = CsrMatrix::from_triplets(4, 4, {});
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto res = sparse_linalg::cg_solve(A, b);
    std::printf("       zero_matrix: status=%s residual=%.3e\n",
                sparse_linalg::to_string(res.status), res.final_residual);
    check(res.status == CgStatus::NotPositiveDefinite && !res.converged,
          "zero_matrix: reported as not positive definite");
}

void test_invalid_inputs() {
    // Non-square matrix.
    auto rect = CsrMatrix::from_triplets(3, 5, {{0, 0, 1.0}});
    auto r1 = sparse_linalg::cg_solve(rect, {1.0, 2.0, 3.0});
    check(r1.status == CgStatus::NotSquare, "invalid: non-square detected");

    // Right-hand side of the wrong size.
    auto A = poisson_1d(4);
    auto r2 = sparse_linalg::cg_solve(A, {1.0, 2.0});
    check(r2.status == CgStatus::DimensionMismatch,
          "invalid: rhs dimension mismatch");

    // Symmetric but indefinite matrix (eigenvalues 3 and -1); the rhs has a
    // component along the negative eigenvector, so CG must hit p'Ap <= 0.
    auto indef = CsrMatrix::from_triplets(
        2, 2, {{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 2.0}, {1, 1, 1.0}});
    CgOptions opts;
    opts.max_iterations = 100;
    auto r3 = sparse_linalg::cg_solve(indef, {1.0, -1.0}, opts);
    std::printf("       indefinite: status=%s residual=%.3e\n",
                sparse_linalg::to_string(r3.status), r3.final_residual);
    check(r3.status == CgStatus::NotPositiveDefinite && !r3.converged,
          "invalid: indefinite matrix detected");

    // NaN in the right-hand side.
    auto r4 = sparse_linalg::cg_solve(
        A, {1.0, std::nan(""), 3.0, 4.0});
    check(r4.status == CgStatus::NumericalError, "invalid: NaN rhs detected");
}

void test_random_sparse_spd() {
    std::mt19937_64 rng(42);
    const std::size_t n = 200;
    // A = M^T M + n*I with sparse random M: SPD by construction.
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);

    std::vector<std::vector<std::pair<std::size_t, double>>> cols(n);
    for (std::size_t k = 0; k < 8 * n; ++k) {
        std::size_t i = pick(rng), j = pick(rng);
        cols[j].emplace_back(i, val(rng));
    }
    std::vector<Triplet> t;
    for (std::size_t j = 0; j < n; ++j)
        for (auto [i, v] : cols[j])
            for (auto [i2, v2] : cols[j]) t.push_back({i, i2, v * v2});
    for (std::size_t i = 0; i < n; ++i) t.push_back({i, i, static_cast<double>(n)});
    auto A = CsrMatrix::from_triplets(n, n, t);

    std::vector<double> x_exact(n);
    for (std::size_t i = 0; i < n; ++i) x_exact[i] = val(rng);
    auto b = A.matvec(x_exact);

    CgOptions opts;
    opts.tolerance = 1e-10;
    opts.max_iterations = 5000;
    auto res = sparse_linalg::cg_solve(A, b, opts);

    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        err = std::max(err, std::fabs(res.solution[i] - x_exact[i]));
    std::printf("       random_spd: n=%zu nnz=%zu iters=%zu residual=%.3e max_err=%.3e\n",
                n, A.nnz(), res.iterations, res.final_residual, err);
    check(res.converged, "random_spd: converged");
    check(res.final_residual <= 1e-10 * 10, "random_spd: residual small");
    check(err < 1e-4, "random_spd: solution accurate");
}

void test_large_sparse_system() {
    // 2D 5-point Poisson on a 300x300 grid (90k unknowns, ~448k nonzeros).
    // Solved purely in CSR form; never densified.
    const std::size_t g = 300;
    const std::size_t n = g * g;
    std::vector<Triplet> t;
    t.reserve(5 * n);
    for (std::size_t r = 0; r < g; ++r) {
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = r * g + c;
            t.push_back({i, i, 4.0});
            if (r > 0) t.push_back({i, i - g, -1.0});
            if (r + 1 < g) t.push_back({i, i + g, -1.0});
            if (c > 0) t.push_back({i, i - 1, -1.0});
            if (c + 1 < g) t.push_back({i, i + 1, -1.0});
        }
    }
    auto A = CsrMatrix::from_triplets(n, n, t);
    std::vector<double> b(n, 1.0);

    CgOptions opts;
    opts.tolerance = 1e-8;
    opts.max_iterations = 5000;
    auto res = sparse_linalg::cg_solve(A, b, opts);

    // Independent residual check: ||b - A x|| / ||b|| via one extra matvec.
    auto Ax = A.matvec(res.solution);
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        num += (b[i] - Ax[i]) * (b[i] - Ax[i]);
        den += b[i] * b[i];
    }
    const double true_residual = std::sqrt(num / den);
    std::printf("       large_poisson: n=%zu nnz=%zu iters=%zu residual=%.3e verified=%.3e\n",
                n, A.nnz(), res.iterations, res.final_residual, true_residual);
    check(res.converged, "large_poisson: converged");
    check(true_residual < 1e-7, "large_poisson: verified residual small");
}

}  // namespace

int main() {
    std::printf("=== sparse_linalg test suite ===\n");
    test_csr_layout();
    test_duplicate_merge();
    test_matvec_matches_dense();
    test_known_system();
    test_preconditioner_effect();
    test_zero_matrix();
    test_invalid_inputs();
    test_random_sparse_spd();
    test_large_sparse_system();
    std::printf("=== %s (%d failure%s) ===\n",
                g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
