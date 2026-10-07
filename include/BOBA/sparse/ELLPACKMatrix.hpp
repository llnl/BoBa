// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief ELLPACK sparse matrix with a common row-slot width.
 * Unused column slots contain the signed sentinel `-1`.
 */
template <typename data_t = double>
struct ELLPACKMatrix
{
  using data_type = data_t;
  using const_view_type = SparseMatrixConstView<ELLPACKMatrix>;
  index_t rows = 0, cols = 0, width = 0;
  sparse_detail::host_matrix<data_t> values;
  sparse_detail::host_matrix<std::int64_t> column_indices;
  ELLPACKMatrix() = default;
  ELLPACKMatrix(index_t m, index_t n, index_t w = 0)
      : rows(m),
        cols(n),
        width(w),
        values({m, w}),
        column_indices({m, w})
  {
    for (index_t i = 0; i < m * w; ++i)
      column_indices.data()[i] = -1;
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
    return 0;
  }
  const_view_type as_const_view() const noexcept
  {
    return {this};
  }
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= rows || j >= cols)
      return {};
    auto c = column_indices.const_view();
    auto v = values.const_view();
    for (index_t k = 0; k < width; ++k)
      if (c({i, k}) == static_cast<std::int64_t>(j))
        return v({i, k});
    return {};
  }
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(rows, cols, x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto c = column_indices.const_view();
    auto v = values.const_view();
    for (index_t i = 0; i < rows; ++i)
      for (index_t k = 0; k < width; ++k)
        if (c({i, k}) >= 0)
          yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(c({i, k})));
  }
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < rows && j < cols, "Sparse coordinate out of bounds");
    for (index_t k = 0; k < width; ++k)
      if (column_indices.view()({i, k}) == j)
      {
        values.view()({i, k}) = value;
        return;
      }
    if (!(abs(value) > 0))
      return;
    for (index_t k = 0; k < width; ++k)
      if (column_indices.view()({i, k}) < 0)
      {
        column_indices.view()({i, k}) = j;
        values.view()({i, k}) = value;
        return;
      }
    index_t old = width;
    values.resize({rows, width + 1});
    column_indices.resize({rows, width + 1});
    for (index_t r = 0; r < rows; ++r)
      column_indices.view()({r, old}) = -1;
    width = old + 1;
    column_indices.view()({i, old}) = j;
    values.view()({i, old}) = value;
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
