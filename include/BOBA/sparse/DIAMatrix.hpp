// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/SparseMatrixCommon.hpp"

namespace boba
{

/**
 * \brief Sparse matrix stored by diagonal offsets `column - row`.
 */
template <typename data_t = double>
struct DIAMatrix
{
  using data_type = data_t;
  using const_view_type = SparseMatrixConstView<DIAMatrix>;
  index_t rows = 0, cols = 0;
  sparse_detail::host_matrix<data_t> values;
  sparse_detail::host_vector<std::int64_t> offsets;
  DIAMatrix() = default;
  DIAMatrix(index_t m, index_t n)
      : rows(m),
        cols(n),
        values({m, 0})
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
    auto o = offsets.const_view();
    auto v = values.const_view();
    for (index_t k = 0; k < offsets.size(); ++k)
      if (static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i) == o(k))
        return v({i, k});
    return {};
  }
  void matvec(sparse_detail::host_vector<data_t> const& x, sparse_detail::host_vector<data_t>& y, data_t alpha = 1, data_t beta = 0) const
  {
    sparse_detail::check_vector_sizes(rows, cols, x, y);
    sparse_detail::initialize_output(y, beta);
    auto xv = x.const_view();
    auto yv = y.view();
    auto o = offsets.const_view();
    auto v = values.const_view();
    for (index_t i = 0; i < rows; ++i)
      for (index_t k = 0; k < offsets.size(); ++k)
      {
        auto j = static_cast<std::int64_t>(i) + o(k);
        if (j >= 0 && j < static_cast<std::int64_t>(cols))
          yv(i) += alpha * v({i, k}) * xv(static_cast<index_t>(j));
      }
  }
  void set_element(index_t i, index_t j, data_t value)
  {
    boba_always_assert(i < rows && j < cols, "Sparse coordinate out of bounds");
    std::int64_t d = static_cast<std::int64_t>(j) - static_cast<std::int64_t>(i);
    for (index_t k = 0; k < offsets.size(); ++k)
      if (offsets.view()(k) == d)
      {
        values.view()({i, k}) = value;
        return;
      }
    if (!(abs(value) > 0))
      return;
    index_t old = offsets.size();
    offsets.resize(old + 1);
    values.resize({rows, old + 1});
    offsets.view()(old) = d;
    for (index_t r = 0; r < rows; ++r)
      values.view()({r, old}) = data_t{};
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
