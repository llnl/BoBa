// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Block coordinate sparse matrix with dense rectangular blocks.
 */
template <typename data_t = double>
struct BCOOMatrix
{
  using data_type = data_t;
  using const_view_type = SparseMatrixConstView<BCOOMatrix>;
  index_t block_rows = 0, block_cols = 0, block_grid_rows = 0, block_grid_cols = 0;
  sparse_detail::host_tensor3<data_t> values;
  sparse_detail::host_matrix<index_t> indices;
  BCOOMatrix() = default;
  BCOOMatrix(index_t br, index_t bc, index_t gr, index_t gc)
      : block_rows(br),
        block_cols(bc),
        block_grid_rows(gr),
        block_grid_cols(gc),
        values({br, bc, 0}),
        indices({0, 2})
  {
  }
  index_t nrows() const noexcept
  {
    return block_rows * block_grid_rows;
  }
  index_t ncols() const noexcept
  {
    return block_cols * block_grid_cols;
  }
  Array<index_t, 2> shape() const noexcept
  {
    return {nrows(), ncols()};
  }
  index_t nnz() const noexcept
  {
    return indices.sizes(0);
  }
  const_view_type as_const_view() const noexcept
  {
    return {this};
  }
  data_t get_element(index_t i, index_t j) const
  {
    if (i >= nrows() || j >= ncols())
      return {};
    index_t p = i / block_rows, q = j / block_cols, u = i % block_rows, v = j % block_cols;
    auto iv = indices.const_view();
    auto vv = values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
      if (iv({k, 0}) == p && iv({k, 1}) == q)
        return vv({u, v, k});
    return {};
  }
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(nrows(), ncols(), x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto iv = indices.const_view();
    auto vv = values.const_view();
    for (index_t k = 0; k < nnz(); ++k)
    {
      index_t p = iv({k, 0}), q = iv({k, 1});
      for (index_t u = 0; u < block_rows; ++u)
        for (index_t v = 0; v < block_cols; ++v)
          yv(p * block_rows + u) += alpha * vv({u, v, k}) * xv(q * block_cols + v);
    }
  }
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < nrows() && j < ncols(), "Sparse coordinate out of bounds");
    index_t p = i / block_rows, q = j / block_cols, u = i % block_rows, v = j % block_cols;
    for (index_t k = 0; k < nnz(); ++k)
      if (indices.view()({k, 0}) == p && indices.view()({k, 1}) == q)
      {
        values.view()({u, v, k}) = value;
        return;
      }
    if (!(abs(value) > 0))
      return;
    index_t old = nnz();
    indices.resize({old + 1, 2});
    values.resize({block_rows, block_cols, old + 1});
    for (index_t a = 0; a < block_rows; ++a)
      for (index_t b = 0; b < block_cols; ++b)
        values.view()({a, b, old}) = data_t{};
    indices.view()({old, 0}) = p;
    indices.view()({old, 1}) = q;
    values.view()({u, v, old}) = value;
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
