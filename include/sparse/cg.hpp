#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "sparse/csr_matrix.hpp"

namespace sparse {

enum class CgStatus {
    Converged,              // relative residual reached the tolerance
    MaxIterationsReached,   // iteration budget exhausted before convergence
    NotSquare,              // matrix is not square
    DimensionMismatch,      // rhs/initial guess size does not match the matrix
    NotPositiveDefinite,    // obviously not SPD (non-positive diagonal entry)
    NumericalBreakdown,     // NaN/Inf in input or non-positive p'Ap during iteration
};

inline const char* toString(CgStatus status) {
    switch (status) {
        case CgStatus::Converged: return "Converged";
        case CgStatus::MaxIterationsReached: return "MaxIterationsReached";
        case CgStatus::NotSquare: return "NotSquare";
        case CgStatus::DimensionMismatch: return "DimensionMismatch";
        case CgStatus::NotPositiveDefinite: return "NotPositiveDefinite";
        case CgStatus::NumericalBreakdown: return "NumericalBreakdown";
    }
    return "Unknown";
}

struct CgOptions {
    double tolerance = 1e-10;     // relative residual tolerance: ||r|| <= tol * ||b||
    std::size_t maxIterations = 1000;
    bool useJacobiPreconditioner = true;
};

struct CgResult {
    CgStatus status = CgStatus::Converged;
    std::size_t iterations = 0;
    double finalResidual = 0.0;   // absolute L2 norm of b - A*x at termination
    std::vector<double> solution;

    bool converged() const { return status == CgStatus::Converged; }
};

namespace detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

inline double norm2(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

inline bool allFinite(const std::vector<double>& v) {
    for (double x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

}  // namespace detail

// Solve A x = b with the Conjugate Gradient method, optionally
// Jacobi (diagonal) preconditioned. A must be symmetric positive
// definite; only the CSR representation is ever touched, the matrix
// is never densified.
inline CgResult solveCG(const CsrMatrix& A, const std::vector<double>& b,
                        const CgOptions& options = CgOptions{},
                        const std::vector<double>& x0 = {}) {
    CgResult result;

    if (A.rows() != A.cols()) {
        result.status = CgStatus::NotSquare;
        return result;
    }
    const std::size_t n = A.rows();
    if (b.size() != n || (!x0.empty() && x0.size() != n)) {
        result.status = CgStatus::DimensionMismatch;
        return result;
    }
    if (!detail::allFinite(b) || (!x0.empty() && !detail::allFinite(x0))) {
        result.status = CgStatus::NumericalBreakdown;
        return result;
    }

    // Jacobi preconditioner M = diag(A). A symmetric positive definite
    // matrix must have a strictly positive diagonal, so this doubles as
    // a cheap "obviously not SPD" screen.
    std::vector<double> inv_diag(n, 1.0);
    if (options.useJacobiPreconditioner) {
        for (std::size_t i = 0; i < n; ++i) {
            double d = A.diagonal(i);
            if (!(d > 0.0) || !std::isfinite(d)) {
                result.status = CgStatus::NotPositiveDefinite;
                return result;
            }
            inv_diag[i] = 1.0 / d;
        }
    }

    std::vector<double> x = x0.empty() ? std::vector<double>(n, 0.0) : x0;

    // r = b - A x
    std::vector<double> r = b;
    {
        std::vector<double> ax(n, 0.0);
        A.spmvAdd(x, ax);
        for (std::size_t i = 0; i < n; ++i) r[i] -= ax[i];
    }

    const double b_norm = detail::norm2(b);
    if (b_norm == 0.0) {
        // Trivial system: x = 0 is the exact solution.
        std::fill(x.begin(), x.end(), 0.0);
        result.solution = std::move(x);
        result.status = CgStatus::Converged;
        result.iterations = 0;
        result.finalResidual = 0.0;
        return result;
    }
    const double threshold = options.tolerance * b_norm;

    auto applyPreconditioner = [&](const std::vector<double>& in, std::vector<double>& out) {
        for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * inv_diag[i];
    };

    std::vector<double> z(n, 0.0);
    applyPreconditioner(r, z);
    std::vector<double> p = z;
    double rz = detail::dot(r, z);
    if (!std::isfinite(rz)) {
        result.status = CgStatus::NumericalBreakdown;
        return result;
    }

    std::vector<double> ap(n, 0.0);
    double residual = detail::norm2(r);

    std::size_t iter = 0;
    while (iter < options.maxIterations && residual > threshold) {
        std::fill(ap.begin(), ap.end(), 0.0);
        A.spmvAdd(p, ap);

        double pap = detail::dot(p, ap);
        if (!(pap > 0.0) || !std::isfinite(pap)) {
            // p'Ap <= 0 cannot happen for an SPD matrix; also catches
            // NaN/Inf propagation.
            result.status = CgStatus::NumericalBreakdown;
            result.iterations = iter;
            result.finalResidual = residual;
            result.solution = std::move(x);
            return result;
        }

        double alpha = rz / pap;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        ++iter;

        residual = detail::norm2(r);
        if (!std::isfinite(residual)) {
            result.status = CgStatus::NumericalBreakdown;
            result.iterations = iter;
            result.finalResidual = residual;
            result.solution = std::move(x);
            return result;
        }
        if (residual <= threshold) break;

        applyPreconditioner(r, z);
        double rz_new = detail::dot(r, z);
        double beta = rz_new / rz;
        for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    result.iterations = iter;
    result.finalResidual = residual;
    result.solution = std::move(x);
    result.status = (residual <= threshold) ? CgStatus::Converged
                                            : CgStatus::MaxIterationsReached;
    return result;
}

}  // namespace sparse
