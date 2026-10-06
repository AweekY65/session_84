#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sparse {

// One nonzero entry in coordinate (triplet) form.
struct Triplet {
    std::size_t row;
    std::size_t col;
    double value;
};

// Compressed Sparse Row matrix (double precision).
//
// Layout:
//   values[k]   - nonzero value
//   col_idx[k]  - column index of values[k]
//   row_ptr[i]  - offset into values/col_idx where row i starts
//   Row i occupies [row_ptr[i], row_ptr[i+1]); column indices inside each
//   row are sorted ascending and duplicates are merged by summation.
class CsrMatrix {
public:
    CsrMatrix() = default;
    CsrMatrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), row_ptr_(rows + 1, 0) {}

    // Build a CSR matrix from triplet data. Entries sharing the same
    // (row, col) coordinate are summed; column indices end up sorted
    // within every row. Explicit zero entries are dropped.
    static CsrMatrix fromTriplets(std::size_t rows, std::size_t cols,
                                  std::vector<Triplet> triplets) {
        for (const Triplet& t : triplets) {
            if (t.row >= rows || t.col >= cols) {
                throw std::out_of_range("triplet index out of matrix bounds");
            }
        }

        // Count entries per row.
        std::vector<std::size_t> row_ptr(rows + 1, 0);
        for (const Triplet& t : triplets) {
            ++row_ptr[t.row + 1];
        }
        for (std::size_t i = 0; i < rows; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }

        // Scatter triplets into CSR arrays (unsorted columns for now).
        std::vector<std::size_t> col_idx(triplets.size());
        std::vector<double> values(triplets.size());
        std::vector<std::size_t> next(row_ptr.begin(), row_ptr.end() - 1);
        for (const Triplet& t : triplets) {
            std::size_t slot = next[t.row]++;
            col_idx[slot] = t.col;
            values[slot] = t.value;
        }

        // Per row: sort by column, merge duplicates, drop exact zeros.
        CsrMatrix mat;
        mat.rows_ = rows;
        mat.cols_ = cols;
        mat.row_ptr_.assign(rows + 1, 0);
        mat.col_idx_.reserve(triplets.size());
        mat.values_.reserve(triplets.size());

        for (std::size_t i = 0; i < rows; ++i) {
            std::size_t begin = row_ptr[i];
            std::size_t end = row_ptr[i + 1];
            std::vector<std::size_t> order(end - begin);
            for (std::size_t k = 0; k < order.size(); ++k) {
                order[k] = begin + k;
            }
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return col_idx[a] < col_idx[b];
            });

            for (std::size_t k = 0; k < order.size();) {
                std::size_t col = col_idx[order[k]];
                double sum = 0.0;
                while (k < order.size() && col_idx[order[k]] == col) {
                    sum += values[order[k]];
                    ++k;
                }
                if (sum != 0.0) {
                    mat.col_idx_.push_back(col);
                    mat.values_.push_back(sum);
                }
            }
            mat.row_ptr_[i + 1] = mat.col_idx_.size();
        }
        return mat;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t nnz() const { return values_.size(); }

    const std::vector<std::size_t>& rowPtr() const { return row_ptr_; }
    const std::vector<std::size_t>& colIdx() const { return col_idx_; }
    const std::vector<double>& values() const { return values_; }

    // y = A * x. Throws on dimension mismatch.
    std::vector<double> spmv(const std::vector<double>& x) const {
        if (x.size() != cols_) {
            throw std::invalid_argument("spmv: vector size does not match matrix columns");
        }
        std::vector<double> y(rows_, 0.0);
        spmvAdd(x, y);
        return y;
    }

    // y += A * x (no allocation, used inside the solver loop).
    void spmvAdd(const std::vector<double>& x, std::vector<double>& y) const {
        for (std::size_t i = 0; i < rows_; ++i) {
            double sum = 0.0;
            for (std::size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                sum += values_[k] * x[col_idx_[k]];
            }
            y[i] += sum;
        }
    }

    // Diagonal entry of row i, or 0 if not stored.
    double diagonal(std::size_t i) const {
        for (std::size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
            if (col_idx_[k] == i) return values_[k];
            if (col_idx_[k] > i) break;  // columns are sorted
        }
        return 0.0;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<std::size_t> row_ptr_;
    std::vector<std::size_t> col_idx_;
    std::vector<double> values_;
};

}  // namespace sparse
