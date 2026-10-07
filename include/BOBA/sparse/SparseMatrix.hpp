// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/Matrix.hpp"
#include "BOBA/tensors/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace boba
{

namespace sparse_detail
{

template <typename data_t>
using host_vector = Vector<host_space, data_t>;

template <typename data_t>
using host_matrix = Matrix<host_space, data_t>;

template <typename data_t>
using host_tensor3 = Tensor<3, host_space, data_t>;

template <typename data_t>
void check_tolerance(data_t tolerance)
{
  boba_always_assert(tolerance >= data_t{0}, "Sparse tolerance must be nonnegative");
  boba_always_assert(std::isfinite(static_cast<double>(tolerance)), "Sparse tolerance must be finite");
}

template <typename data_t>
bool keep(data_t value, data_t tolerance)
{
  return abs(value) > tolerance;
}

template <typename data_t>
void check_vector_sizes(index_t rows, index_t cols, host_vector<data_t> const& x, host_vector<data_t> const& y)
{
  boba_always_assert_equal(x.size(), cols, "Sparse matvec input has the wrong size");
  boba_always_assert_equal(y.size(), rows, "Sparse matvec output has the wrong size");
}

template <typename data_t>
void initialize_output(host_vector<data_t>& y, data_t beta)
{
  auto y_view = y.view();
  if (abs(beta) <= 0)
  {
    y.fill_with_zeros();
  }
  else
  {
    for (index_t i = 0; i < y.size(); ++i)
      y_view(i) *= beta;
  }
}

template <typename data_t>
host_matrix<data_t> copy_dense(host_matrix<data_t> const& dense)
{
  return dense;
}

} // namespace sparse_detail

template <typename owner_t>
struct SparseMatrixConstView
{
  owner_t const* owner = nullptr;

  index_t nrows() const noexcept
  {
    return owner->nrows();
  }
  index_t ncols() const noexcept
  {
    return owner->ncols();
  }
  auto shape() const noexcept
  {
    return owner->shape();
  }
  auto get_element(index_t i, index_t j) const
  {
    return owner->get_element(i, j);
  }

  void matvec(Vector<host_space, typename owner_t::data_type> const& x,
              Vector<host_space, typename owner_t::data_type>& y,
              typename owner_t::data_type alpha = typename owner_t::data_type{1},
              typename owner_t::data_type beta = typename owner_t::data_type{0}) const
  {
    owner->matvec(x, y, alpha, beta);
  }
};

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

template <typename MatrixType>
Vector<host_space, typename MatrixType::data_type> matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x)
{
  Vector<host_space, typename MatrixType::data_type> y({A.nrows()});
  A.matvec(x, y);
  return y;
}

template <typename MatrixType>
void matvec(MatrixType const& A, Vector<host_space, typename MatrixType::data_type> const& x, Vector<host_space, typename MatrixType::data_type>& y, typename MatrixType::data_type alpha = 1, typename MatrixType::data_type beta = 0)
{
  A.matvec(x, y, alpha, beta);
}

template <typename data_t>
COOMatrix<data_t> from_dense_coo(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  COOMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}

template <typename data_t>
CSRMatrix<data_t> from_dense_csr(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  CSRMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}
template <typename data_t>
ELLPACKMatrix<data_t> from_dense_ellpack(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  index_t w = 0;
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
  {
    index_t c = 0;
    for (index_t j = 0; j < dense.cols(); ++j)
      c += sparse_detail::keep(dv({i, j}), tolerance);
    w = std::max(w, c);
  }
  ELLPACKMatrix<data_t> out(dense.rows(), dense.cols(), w);
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0, k = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
      {
        out.column_indices.view()({i, k}) = static_cast<std::int64_t>(j);
        out.values.view()({i, k++}) = dv({i, j});
      }
  return out;
}
template <typename data_t>
DIAMatrix<data_t> from_dense_dia(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  DIAMatrix<data_t> out(dense.rows(), dense.cols());
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}
template <typename data_t>
BCOOMatrix<data_t> from_dense_bcoo(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  sparse_detail::check_tolerance(tolerance);
  boba_always_assert(br > 0 && bc > 0, "BCOO block dimensions must be positive");
  boba_always_assert(dense.rows() % br == 0 && dense.cols() % bc == 0, "BCOO dimensions must be divisible by block dimensions");
  BCOOMatrix<data_t> out(br, bc, dense.rows() / br, dense.cols() / bc);
  auto dv = dense.const_view();
  for (index_t i = 0; i < dense.rows(); ++i)
    for (index_t j = 0; j < dense.cols(); ++j)
      if (sparse_detail::keep(dv({i, j}), tolerance))
        out.set_element(i, j, dv({i, j}));
  return out;
}

template <typename sparse_t, typename data_t>
sparse_t from_dense(Matrix<host_space, data_t> const& dense, data_t tolerance = 0)
{
  if constexpr (std::is_same_v<sparse_t, COOMatrix<data_t>>)
    return from_dense_coo(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, CSRMatrix<data_t>>)
    return from_dense_csr(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, ELLPACKMatrix<data_t>>)
    return from_dense_ellpack(dense, tolerance);
  else if constexpr (std::is_same_v<sparse_t, DIAMatrix<data_t>>)
    return from_dense_dia(dense, tolerance);
  else
    static_assert(std::is_same_v<sparse_t, void>, "BCOO from_dense requires block dimensions");
}

template <typename sparse_t, typename data_t>
sparse_t from_dense(Matrix<host_space, data_t> const& dense, index_t br, index_t bc, data_t tolerance = 0)
{
  static_assert(std::is_same_v<sparse_t, BCOOMatrix<data_t>>, "Block-parameterized from_dense is only defined for BCOO");
  return from_dense_bcoo(dense, br, bc, tolerance);
}

template <typename MatrixType>
Matrix<host_space, typename MatrixType::data_type> to_dense(MatrixType const& sparse)
{
  using data_t = typename MatrixType::data_type;
  Matrix<host_space, data_t> out({sparse.nrows(), sparse.ncols()});
  out.fill_with_zeros();
  auto ov = out.view();
  for (index_t i = 0; i < sparse.nrows(); ++i)
    for (index_t j = 0; j < sparse.ncols(); ++j)
      ov({i, j}) = sparse.get_element(i, j);
  return out;
}

} // namespace boba
