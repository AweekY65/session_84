// Terminal test suite for the local sparse linear algebra module.
// Every test prints its residual / outcome; a non-zero exit code means
// at least one test failed.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "sparse/cg.hpp"
#include "sparse/csr_matrix.hpp"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", name.c_str());
    if (!condition) ++g_failures;
}

// Dense reference matvec, used to validate CSR spmv.
std::vector<double> denseSpmv(std::size_t rows, std::size_t /*cols*/,
                              const std::vector<sparse::Triplet>& triplets,
                              const std::vector<double>& x) {
    std::vector<double> y(rows, 0.0);
    for (const auto& t : triplets) y[t.row] += t.value * x[t.col];
    return y;
}

double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

// Residual ||b - A x||2 computed purely in CSR form.
double csrResidual(const sparse::CsrMatrix& A, const std::vector<double>& b,
                   const std::vector<double>& x) {
    std::vector<double> ax = A.spmv(x);
    double sum = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        double ri = b[i] - ax[i];
        sum += ri * ri;
    }
    return std::sqrt(sum);
}

// 1D Poisson matrix (tridiagonal 2, -1) of size n.
sparse::CsrMatrix poisson1D(std::size_t n) {
    std::vector<sparse::Triplet> t;
    for (std::size_t i = 0; i < n; ++i) {
        t.push_back({i, i, 2.0});
        if (i > 0) t.push_back({i, i - 1, -1.0});
        if (i + 1 < n) t.push_back({i, i + 1, -1.0});
    }
    return sparse::CsrMatrix::fromTriplets(n, n, t);
}

void testCsrBuild() {
    std::printf("== CSR build from triplets ==\n");
    std::vector<sparse::Triplet> t = {
        {0, 0, 4.0}, {0, 2, 1.0}, {1, 1, 3.0}, {2, 0, 2.0}, {2, 2, 5.0},
    };
    auto A = sparse::CsrMatrix::fromTriplets(3, 4, t);  // rectangular on purpose
    check(A.rows() == 3 && A.cols() == 4, "dimensions preserved");
    check(A.nnz() == 5, "nnz matches");
    check((A.rowPtr() == std::vector<std::size_t>{0, 2, 3, 5}), "row_ptr layout");
    check((A.colIdx() == std::vector<std::size_t>{0, 2, 1, 0, 2}), "col_idx layout");
    check((A.values() == std::vector<double>{4.0, 1.0, 3.0, 2.0, 5.0}), "values layout");
}

void testDuplicateMergeAndSorting() {
    std::printf("== Duplicate merge + column sorting ==\n");
    // Deliberately shuffled, duplicated triplets.
    std::vector<sparse::Triplet> t = {
        {1, 2, 1.0}, {0, 3, 5.0}, {1, 0, 2.0}, {0, 3, 7.0},
        {1, 2, 4.0}, {0, 1, 1.0}, {1, 2, -5.0},  // sums to 0 -> dropped
    };
    auto A = sparse::CsrMatrix::fromTriplets(2, 4, t);
    check(A.nnz() == 3, "duplicates merged, exact zeros dropped");
    check((A.rowPtr() == std::vector<std::size_t>{0, 2, 3}), "row_ptr after merge");
    check((A.colIdx() == std::vector<std::size_t>{1, 3, 0}), "columns sorted per row");
    check((A.values() == std::vector<double>{1.0, 12.0, 2.0}), "duplicate values summed");
}

void testSpmvAgainstDense() {
    std::printf("== SpMV vs dense reference ==\n");
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> val(-2.0, 2.0);
    std::uniform_int_distribution<int> coin(0, 1);

    const std::size_t n = 200;
    std::vector<sparse::Triplet> t;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (coin(rng)) t.push_back({i, j, val(rng)});
        }
    }
    std::vector<double> x(n);
    for (auto& v : x) v = val(rng);

    auto A = sparse::CsrMatrix::fromTriplets(n, n, t);
    std::vector<double> y_csr = A.spmv(x);
    std::vector<double> y_dense = denseSpmv(n, n, t, x);

    double err = maxAbsDiff(y_csr, y_dense);
    std::printf("  residual |y_csr - y_dense|_inf = %.3e\n", err);
    check(err < 1e-12, "spmv matches dense computation");
}

void testKnownSystem() {
    std::printf("== Known linear system (1D Poisson, exact solution) ==\n");
    const std::size_t n = 50;
    auto A = poisson1D(n);
    // Manufactured solution x_exact = [1, 2, ..., n].
    std::vector<double> x_exact(n);
    for (std::size_t i = 0; i < n; ++i) x_exact[i] = static_cast<double>(i + 1);
    std::vector<double> b = A.spmv(x_exact);

    sparse::CgOptions opts;
    opts.tolerance = 1e-12;
    opts.maxIterations = 500;
    auto res = sparse::solveCG(A, b, opts);

    double err = maxAbsDiff(res.solution, x_exact);
    std::printf("  status=%s iterations=%zu final_residual=%.3e solution_error=%.3e\n",
                sparse::toString(res.status), res.iterations, res.finalResidual, err);
    check(res.converged(), "CG converged");
    check(err < 1e-8, "solution matches manufactured exact solution");
}

void testPreconditionerEffect() {
    std::printf("== Jacobi preconditioner effect ==\n");
    // A = D^(1/2) L D^(1/2) with a wildly varying diagonal D: the
    // un-preconditioned method sees a terrible condition number, while
    // Jacobi rescales the diagonal back to the well-behaved Laplacian.
    const std::size_t n = 200;
    std::vector<sparse::Triplet> t;
    for (std::size_t i = 0; i < n; ++i) {
        double di = 1.0 + 1e4 * static_cast<double>(i % 7) / 7.0;
        double si = std::sqrt(di);
        t.push_back({i, i, 2.0 * di});
        if (i > 0) {
            double sj = std::sqrt(1.0 + 1e4 * static_cast<double>((i - 1) % 7) / 7.0);
            t.push_back({i, i - 1, -si * sj});
        }
        if (i + 1 < n) {
            double sj = std::sqrt(1.0 + 1e4 * static_cast<double>((i + 1) % 7) / 7.0);
            t.push_back({i, i + 1, -si * sj});
        }
    }
    auto A = sparse::CsrMatrix::fromTriplets(n, n, t);

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    std::vector<double> b(n);
    for (auto& v : b) v = val(rng);

    sparse::CgOptions opts;
    opts.tolerance = 1e-10;
    opts.maxIterations = 5000;

    opts.useJacobiPreconditioner = false;
    auto plain = sparse::solveCG(A, b, opts);
    opts.useJacobiPreconditioner = true;
    auto precond = sparse::solveCG(A, b, opts);

    std::printf("  no preconditioner:   status=%s iterations=%zu residual=%.3e\n",
                sparse::toString(plain.status), plain.iterations, plain.finalResidual);
    std::printf("  jacobi preconditioner: status=%s iterations=%zu residual=%.3e\n",
                sparse::toString(precond.status), precond.iterations, precond.finalResidual);
    check(plain.converged(), "plain CG converged");
    check(precond.converged(), "preconditioned CG converged");
    check(precond.iterations < plain.iterations,
          "jacobi preconditioner reduces iteration count");
}

void testZeroMatrix() {
    std::printf("== Zero matrix ==\n");
    auto A = sparse::CsrMatrix::fromTriplets(4, 4, {});
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0};
    auto res = sparse::solveCG(A, b);
    std::printf("  status=%s iterations=%zu residual=%.3e\n",
                sparse::toString(res.status), res.iterations, res.finalResidual);
    check(res.status == sparse::CgStatus::NotPositiveDefinite,
          "zero matrix reported as not positive definite");
    check(!res.converged(), "zero matrix does not report convergence");
}

void testNonPositiveDefinite() {
    std::printf("== Non-positive-definite inputs ==\n");
    // Negative diagonal entry: obviously not SPD.
    auto A = sparse::CsrMatrix::fromTriplets(2, 2, {{0, 0, 1.0}, {1, 1, -2.0}});
    std::vector<double> b = {1.0, 1.0};
    auto res = sparse::solveCG(A, b);
    std::printf("  negative diagonal: status=%s\n", sparse::toString(res.status));
    check(res.status == sparse::CgStatus::NotPositiveDefinite,
          "negative diagonal flagged");

    // Positive diagonal but indefinite: [[1, 3], [3, 1]] has eigenvalue -2.
    // CG must detect the non-positive p'Ap breakdown instead of looping.
    auto B = sparse::CsrMatrix::fromTriplets(
        2, 2, {{0, 0, 1.0}, {0, 1, 3.0}, {1, 0, 3.0}, {1, 1, 1.0}});
    auto res2 = sparse::solveCG(B, {1.0, 0.0});
    std::printf("  indefinite (p'Ap<=0): status=%s iterations=%zu residual=%.3e\n",
                sparse::toString(res2.status), res2.iterations, res2.finalResidual);
    check(res2.status == sparse::CgStatus::NumericalBreakdown,
          "indefinite matrix caught via p'Ap breakdown");
}

void testInvalidInputs() {
    std::printf("== Invalid input handling ==\n");
    auto rect = sparse::CsrMatrix::fromTriplets(2, 3, {{0, 0, 1.0}, {1, 2, 1.0}});
    auto res = sparse::solveCG(rect, {1.0, 1.0});
    std::printf("  non-square: status=%s\n", sparse::toString(res.status));
    check(res.status == sparse::CgStatus::NotSquare, "non-square matrix rejected");

    auto sq = poisson1D(3);
    auto res2 = sparse::solveCG(sq, {1.0, 1.0});
    std::printf("  dimension mismatch: status=%s\n", sparse::toString(res2.status));
    check(res2.status == sparse::CgStatus::DimensionMismatch, "rhs size mismatch rejected");

    auto res3 = sparse::solveCG(sq, {1.0, std::nan(""), 3.0});
    std::printf("  NaN rhs: status=%s\n", sparse::toString(res3.status));
    check(res3.status == sparse::CgStatus::NumericalBreakdown, "NaN rhs rejected");
}

void testRandomSparseSpd() {
    std::printf("== Random sparse SPD system (Eigen reference) ==\n");
    const std::size_t n = 2000;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::uniform_real_distribution<double> val(-1.0, 1.0);

    // Symmetric, strictly diagonally dominant with positive diagonal => SPD.
    std::vector<sparse::Triplet> t;
    std::vector<std::vector<std::pair<std::size_t, double>>> offdiag(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (prob(rng) < 4.0 / n) {  // ~4 off-diagonals per row
                double v = val(rng);
                offdiag[i].push_back({j, v});
                offdiag[j].push_back({i, v});
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        double diag = 1.0;
        for (auto& [j, v] : offdiag[i]) {
            diag += std::fabs(v);
            t.push_back({i, j, v});
        }
        t.push_back({i, i, diag});
    }
    auto A = sparse::CsrMatrix::fromTriplets(n, n, t);
    std::printf("  n=%zu nnz=%zu (CSR only, never densified)\n", n, A.nnz());

    std::vector<double> b(n);
    for (auto& v : b) v = val(rng);

    sparse::CgOptions opts;
    opts.tolerance = 1e-10;
    opts.maxIterations = 5000;
    auto res = sparse::solveCG(A, b, opts);

    double residual = csrResidual(A, b, res.solution);
    std::printf("  status=%s iterations=%zu final_residual=%.3e recomputed_residual=%.3e\n",
                sparse::toString(res.status), res.iterations, res.finalResidual, residual);
    check(res.converged(), "CG converged on random SPD system");
    check(residual <= 1e-10 * sparse::detail::norm2(b) * 10.0,
          "recomputed residual within tolerance");

    // Cross-check against Eigen's sparse CG on the same system.
    Eigen::SparseMatrix<double> Ae(n, n);
    std::vector<Eigen::Triplet<double>> et;
    et.reserve(t.size());
    for (auto& tr : t) et.emplace_back(tr.row, tr.col, tr.value);
    Ae.setFromTriplets(et.begin(), et.end());
    Eigen::VectorXd be = Eigen::VectorXd::Map(b.data(), n);
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                             Eigen::Lower | Eigen::Upper, Eigen::DiagonalPreconditioner<double>>
        eigen_cg(Ae);
    Eigen::VectorXd xe = eigen_cg.solve(be);

    std::vector<double> x_eigen(xe.data(), xe.data() + n);
    double diff = maxAbsDiff(res.solution, x_eigen);
    std::printf("  |x_ours - x_eigen|_inf = %.3e (eigen iterations=%d)\n", diff,
                static_cast<int>(eigen_cg.iterations()));
    check(diff < 1e-6, "solution matches Eigen reference");
}

}  // namespace

int main() {
    testCsrBuild();
    testDuplicateMergeAndSorting();
    testSpmvAgainstDense();
    testKnownSystem();
    testPreconditionerEffect();
    testZeroMatrix();
    testNonPositiveDefinite();
    testInvalidInputs();
    testRandomSparseSpd();

    std::printf("\n%s (failures: %d)\n", g_failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
