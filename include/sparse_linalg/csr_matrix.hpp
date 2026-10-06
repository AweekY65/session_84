#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sparse_linalg {

struct Triplet {
    std::size_t row;
    std::size_t col;
    double value;
};

// Compressed Sparse Row matrix.
// Layout: values[k] / col_idx[k] store the k-th nonzero; entries of row i
// occupy k in [row_ptr[i], row_ptr[i+1]) with column indices sorted ascending.
class CsrMatrix {
public:
    CsrMatrix() = default;

    CsrMatrix(std::size_t rows, std::size_t cols,
              std::vector<std::size_t> row_ptr,
              std::vector<std::size_t> col_idx,
              std::vector<double> values)
        : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)),
          col_idx_(std::move(col_idx)), values_(std::move(values)) {
        if (row_ptr_.size() != rows_ + 1)
            throw std::invalid_argument("row_ptr size must be rows + 1");
        if (col_idx_.size() != values_.size())
            throw std::invalid_argument("col_idx and values size mismatch");
        if (row_ptr_.empty() || row_ptr_.front() != 0)
            throw std::invalid_argument("row_ptr must start at 0");
        if (row_ptr_.back() != values_.size())
            throw std::invalid_argument("row_ptr back must equal nnz");
    }

    // Build CSR from triplet (COO) data. Entries sharing the same (row, col)
    // coordinate are summed together; column indices within each row are kept
    // in ascending order.
    static CsrMatrix from_triplets(std::size_t rows, std::size_t cols,
                                   std::vector<Triplet> triplets) {
        for (const auto& t : triplets) {
            if (t.row >= rows || t.col >= cols)
                throw std::out_of_range("triplet index out of matrix bounds");
        }
        std::stable_sort(triplets.begin(), triplets.end(),
                         [](const Triplet& a, const Triplet& b) {
                             if (a.row != b.row) return a.row < b.row;
                             return a.col < b.col;
                         });

        std::vector<std::size_t> row_ptr(rows + 1, 0);
        std::vector<std::size_t> col_idx;
        std::vector<double> values;
        col_idx.reserve(triplets.size());
        values.reserve(triplets.size());

        std::size_t current_row = 0;
        bool have_last = false;
        std::size_t last_row = 0, last_col = 0;
        for (const auto& t : triplets) {
            while (current_row < t.row) {
                row_ptr[current_row + 1] = values.size();
                ++current_row;
            }
            if (have_last && last_row == t.row && last_col == t.col) {
                values.back() += t.value;  // merge duplicate coordinate
            } else {
                col_idx.push_back(t.col);
                values.push_back(t.value);
                last_row = t.row;
                last_col = t.col;
                have_last = true;
            }
        }
        while (current_row < rows) {
            row_ptr[current_row + 1] = values.size();
            ++current_row;
        }
        return CsrMatrix(rows, cols, std::move(row_ptr), std::move(col_idx),
                         std::move(values));
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t nnz() const { return values_.size(); }

    const std::vector<std::size_t>& row_ptr() const { return row_ptr_; }
    const std::vector<std::size_t>& col_idx() const { return col_idx_; }
    const std::vector<double>& values() const { return values_; }

    // y = A * x. Throws std::invalid_argument on dimension mismatch.
    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        if (x.size() != cols_)
            throw std::invalid_argument("matvec: x size does not match cols");
        y.assign(rows_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i) {
            double sum = 0.0;
            for (std::size_t k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
                sum += values_[k] * x[col_idx_[k]];
            y[i] = sum;
        }
    }

    std::vector<double> matvec(const std::vector<double>& x) const {
        std::vector<double> y;
        matvec(x, y);
        return y;
    }

    // Diagonal entry A(i,i), or 0 if absent from the sparsity pattern.
    double diag(std::size_t i) const {
        if (i >= rows_ || i >= cols_)
            throw std::out_of_range("diag index out of range");
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

}  // namespace sparse_linalg
