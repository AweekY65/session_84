#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "sparse_linalg/csr_matrix.hpp"

namespace sparse_linalg {

enum class CgStatus {
    Converged,             // relative residual reached the tolerance
    MaxIterationsReached,  // iteration budget exhausted before convergence
    NotSquare,             // matrix is not square
    DimensionMismatch,     // right-hand side size does not match the matrix
    NotPositiveDefinite,   // breakdown indicating A is not SPD (e.g. p'Ap <= 0)
    NumericalError         // NaN / Inf encountered during the iteration
};

inline const char* to_string(CgStatus status) {
    switch (status) {
        case CgStatus::Converged: return "Converged";
        case CgStatus::MaxIterationsReached: return "MaxIterationsReached";
        case CgStatus::NotSquare: return "NotSquare";
        case CgStatus::DimensionMismatch: return "DimensionMismatch";
        case CgStatus::NotPositiveDefinite: return "NotPositiveDefinite";
        case CgStatus::NumericalError: return "NumericalError";
    }
    return "Unknown";
}

struct CgOptions {
    double tolerance = 1e-8;        // relative residual tolerance ||r||/||b||
    std::size_t max_iterations = 1000;
    bool use_jacobi_preconditioner = true;
};

struct CgResult {
    CgStatus status = CgStatus::Converged;
    bool converged = false;
    std::size_t iterations = 0;
    double final_residual = 0.0;  // relative residual ||r||/||b|| (or 0 if b==0)
    std::vector<double> solution;
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

inline bool all_finite(const std::vector<double>& v) {
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace detail

// Preconditioned Conjugate Gradient for symmetric positive definite systems
// A x = b. With use_jacobi_preconditioner enabled, M = diag(A) is applied.
// The matrix is never converted to dense form; only matvec is used.
inline CgResult cg_solve(const CsrMatrix& A, const std::vector<double>& b,
                         const CgOptions& options = {}) {
    CgResult result;

    if (A.rows() != A.cols()) {
        result.status = CgStatus::NotSquare;
        return result;
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        result.status = CgStatus::DimensionMismatch;
        return result;
    }
    if (!detail::all_finite(b)) {
        result.status = CgStatus::NumericalError;
        return result;
    }

    result.solution.assign(n, 0.0);
    const double norm_b = detail::norm2(b);
    if (norm_b == 0.0) {
        // x = 0 is the exact solution of A x = 0.
        result.status = CgStatus::Converged;
        result.converged = true;
        result.final_residual = 0.0;
        return result;
    }

    // Jacobi preconditioner M = diag(A). For an SPD matrix every diagonal
    // entry must be strictly positive; otherwise report NotPositiveDefinite.
    std::vector<double> inv_diag;
    if (options.use_jacobi_preconditioner) {
        inv_diag.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double d = A.diag(i);
            if (!(d > 0.0) || !std::isfinite(d)) {
                result.status = CgStatus::NotPositiveDefinite;
                return result;
            }
            inv_diag[i] = 1.0 / d;
        }
    }

    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;  // r0 = b - A x0, x0 = 0
    std::vector<double> z(n, 0.0);
    std::vector<double> p(n, 0.0);
    std::vector<double> Ap(n, 0.0);

    auto apply_preconditioner = [&](const std::vector<double>& in,
                                    std::vector<double>& out) {
        if (options.use_jacobi_preconditioner) {
            for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * inv_diag[i];
        } else {
            out = in;
        }
    };

    apply_preconditioner(r, z);
    p = z;
    double rz = detail::dot(r, z);
    if (!(rz > 0.0) || !std::isfinite(rz)) {
        result.status = CgStatus::NotPositiveDefinite;
        return result;
    }

    const double threshold = options.tolerance * norm_b;
    double residual = detail::norm2(r);

    std::size_t iter = 0;
    for (; iter < options.max_iterations; ++iter) {
        A.matvec(p, Ap);
        const double pAp = detail::dot(p, Ap);
        if (!(pAp > 0.0) || !std::isfinite(pAp)) {
            // p'Ap <= 0 for p != 0 contradicts positive definiteness.
            result.status = CgStatus::NotPositiveDefinite;
            result.iterations = iter;
            result.solution = x;
            result.final_residual = residual / norm_b;
            return result;
        }
        const double alpha = rz / pAp;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        residual = detail::norm2(r);
        if (!std::isfinite(residual)) {
            result.status = CgStatus::NumericalError;
            result.iterations = iter + 1;
            result.solution = x;
            result.final_residual = residual / norm_b;
            return result;
        }
        if (residual <= threshold) {
            result.status = CgStatus::Converged;
            result.converged = true;
            result.iterations = iter + 1;
            result.solution = x;
            result.final_residual = residual / norm_b;
            return result;
        }
        apply_preconditioner(r, z);
        const double rz_new = detail::dot(r, z);
        if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {
            result.status = CgStatus::NotPositiveDefinite;
            result.iterations = iter + 1;
            result.solution = x;
            result.final_residual = residual / norm_b;
            return result;
        }
        const double beta = rz_new / rz;
        for (std::size_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    result.status = CgStatus::MaxIterationsReached;
    result.iterations = iter;
    result.solution = x;
    result.final_residual = residual / norm_b;
    return result;
}

}  // namespace sparse_linalg
