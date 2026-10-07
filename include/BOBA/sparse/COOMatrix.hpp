// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Coordinate-list sparse matrix with unsorted zero-based coordinates.
 */
template <typename data_t = double>
struct COOMatrix
{
  using data_type = data_t;
  using index_container = sparse_detail::host_matrix<index_t>;
  using value_container = sparse_detail::host_vector<data_t>;
  using const_view_type = SparseMatrixConstView<COOMatrix>;

  index_t rows = 0;
  index_t cols = 0;
  value_container values;
  index_container indices;

  COOMatrix() = default;
  COOMatrix(index_t m, index_t n)
      : rows(m),
        cols(n),
        indices({0, 2})
  {
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
    auto iv = indices.const_view();
    auto vv = values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
      if (iv({k, 0}) == i && iv({k, 1}) == j)
        return vv(k);
    return data_t{};
  }

  void matvec(sparse_detail::host_vector<data_t> const& x,
              sparse_detail::host_vector<data_t>& y,
              data_t alpha = 1,
              data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(rows, cols, x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto iv = indices.const_view();
    auto vv = values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
      yv(iv({k, 0})) += alpha * vv(k) * xv(iv({k, 1}));
  }

  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i >= 0 && i < rows && j >= 0 && j < cols, "Sparse coordinate out of bounds");
    for (index_t k = 0; k < nnz(); ++k)
      if (indices.view()({k, 0}) == i && indices.view()({k, 1}) == j)
      {
        values.view()(k) = value;
        return;
      }
    if (!(abs(value) > 0))
      return;
    index_t old = nnz();
    values.resize(old + 1);
    indices.resize({old + 1, 2});
    values.view()(old) = value;
    indices.view()({old, 0}) = i;
    indices.view()({old, 1}) = j;
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
