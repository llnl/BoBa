// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Compressed sparse row matrix.
 */
template <typename data_t = double>
struct CSRMatrix
{
  using data_type = data_t;
  using value_container = sparse_detail::host_vector<data_t>;
  using index_container = sparse_detail::host_vector<index_t>;
  using const_view_type = SparseMatrixConstView<CSRMatrix>;
  index_t rows = 0, cols = 0;
  value_container values;
  index_container column_indices, row_offsets;
  CSRMatrix() = default;
  CSRMatrix(index_t m, index_t n)
      : rows(m),
        cols(n),
        row_offsets({m + 1})
  {
    row_offsets.fill_with_zeros();
  }
  index_t nrows() const noexcept
  {
    return rows;
  }
  index_t ncols() const noexcept
  {
    return cols;
  }
  Array<index_t, 2> shape() const noexcept
  {
    return {rows, cols};
  }
  index_t nnz() const noexcept
  {
    return values.size();
  }
  const_view_type as_const_view() const noexcept
  {
    return {this};
  }
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= rows || j >= cols)
      return data_t{};
    auto ov = row_offsets.const_view();
    auto cv = column_indices.const_view();
    auto vv = values.const_view();
    for (index_t k = ov(i); k < ov(i + 1); ++k)
      if (cv(k) == j)
        return vv(k);
    return data_t{};
  }
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(rows, cols, x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto ov = row_offsets.const_view();
    auto cv = column_indices.const_view();
    auto vv = values.const_view();
    for (index_t i = 0; i < rows; ++i)
      for (index_t k = ov(i); k < ov(i + 1); ++k)
        yv(i) += alpha * vv(k) * xv(cv(k));
  }
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i >= 0 && i < rows && j >= 0 && j < cols, "Sparse coordinate out of bounds");
    auto ov = row_offsets.view();
    auto cv = column_indices.view();
    auto vv = values.view();
    for (index_t k = ov(i); k < ov(i + 1); ++k)
      if (cv(k) == j)
      {
        vv(k) = value;
        return;
      }
    if (!(abs(value) > 0))
      return;
    index_t at = ov(i + 1);
    values.resize(nnz() + 1);
    column_indices.resize(nnz());
    for (index_t k = nnz() - 1; k > at; --k)
    {
      values.view()(k) = values.view()(k - 1);
      column_indices.view()(k) = column_indices.view()(k - 1);
    }
    values.view()(at) = value;
    column_indices.view()(at) = j;
    for (index_t r = i + 1; r <= rows; ++r)
      row_offsets.view()(r)++;
  }
  void add_element(index_t i, index_t j, data_t value)
  {
    set_element(i, j, get_element(i, j) + value);
  }
  void zero_values()
  {
    values.fill_with_zeros();
  }
  void scale(data_t value)
  {
    values *= value;
  }
};

} // namespace boba
