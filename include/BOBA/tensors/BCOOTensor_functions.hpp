// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/BCOOTensor.hpp"
#include "BOBA/tensors/SparseTensor.hpp"

#include <map>
#include <vector>

namespace boba
{

namespace detail
{

/** \brief Splits a column-major linear index into (first mode, interior coordinates, last mode). */
template <size_t dimension>
void split_bcoo_index(Array<index_t, dimension> const& sizes,
                      index_t linear,
                      index_t& first,
                      Array<index_t, dimension - 2>& interior,
                      index_t& last)
{
  first = linear % sizes[0];
  linear /= sizes[0];
  for (size_t j = 0; j < dimension - 2; ++j)
  {
    interior[j] = linear % sizes[j + 1];
    linear /= sizes[j + 1];
  }
  last = linear;
}

/** \brief Column-major linear index of an interior coordinate (mode 1 fastest). */
template <size_t dimension>
index_t interior_linear_index(Array<index_t, dimension> const& sizes, Array<index_t, dimension - 2> const& interior)
{
  index_t linear = 0;
  index_t stride = 1;
  for (size_t j = 0; j < dimension - 2; ++j)
  {
    linear += interior[j] * stride;
    stride *= sizes[j + 1];
  }
  return linear;
}

} // namespace detail

/**
 * \brief Returns the dense tensor with the same extents and stored blocks; unstored blocks are zero.
 *
 * Computed on the host; the result lives in the space of \p bcoo.
 */
template <size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> to_dense(BCOOTensor<dimension, space, data_t> const& bcoo)
{
  BCOOTensor<dimension, host_space, data_t> const host = bcoo;
  Tensor<dimension, host_space, data_t> dense(host.sizes());
  dense.fill_with_zeros();
  const auto& sizes = host.sizes();
  const index_t first = sizes[0];
  const index_t last = sizes[dimension - 1];
  index_t interior_size = 1;
  for (size_t j = 1; j + 1 < dimension; ++j)
  {
    interior_size *= sizes[j];
  }
  auto indices_view = host.m_indices.const_view();
  const data_t* values = host.m_values.data();
  data_t* out = dense.data();
  for (index_t k = 0; k < host.nonzero_blocks(); ++k)
  {
    Array<index_t, dimension - 2> interior{};
    for (size_t j = 0; j < dimension - 2; ++j)
    {
      interior[j] = indices_view({k, static_cast<index_t>(j)});
    }
    const index_t mid = detail::interior_linear_index<dimension>(sizes, interior);
    for (index_t b = 0; b < last; ++b)
    {
      for (index_t a = 0; a < first; ++a)
      {
        out[a + first * (mid + interior_size * b)] = values[a + first * (b + last * k)];
      }
    }
  }
  return Tensor<dimension, space, data_t>(dense);
}

/**
 * \brief Imports a dense tensor into block coordinate format.
 *
 * A block `T(:, i_1, ..., i_{d-2}, :)` is stored if any of its entries has `abs(value) > tolerance`; stored blocks keep
 * all their entries exactly. Runs on host.
 */
template <size_t dimension, execution_space space, typename data_t>
  requires(dimension >= 3)
BCOOTensor<dimension, space, data_t> from_dense_bcoo(Tensor<dimension, space, data_t> const& dense,
                                                     real_type_t<data_t> tolerance = 0)
{
  using real_data_t = real_type_t<data_t>;
  boba_always_assert(tolerance >= real_data_t{0}, "BCOO tolerance must be nonnegative");
  boba_always_assert(std::isfinite(static_cast<double>(tolerance)), "BCOO tolerance must be finite");
  Tensor<dimension, host_space, data_t> const host = dense;
  const auto sizes = host.sizes();
  const index_t first = sizes[0];
  const index_t last = sizes[dimension - 1];
  index_t interior_size = 1;
  for (size_t j = 1; j + 1 < dimension; ++j)
  {
    interior_size *= sizes[j];
  }
  const data_t* in = host.data();
  std::vector<index_t> kept;
  for (index_t mid = 0; mid < interior_size; ++mid)
  {
    bool keep = false;
    for (index_t b = 0; b < last && !keep; ++b)
    {
      for (index_t a = 0; a < first; ++a)
      {
        if (::boba::abs(in[a + first * (mid + interior_size * b)]) > tolerance)
        {
          keep = true;
          break;
        }
      }
    }
    if (keep)
    {
      kept.push_back(mid);
    }
  }
  const auto blocks = static_cast<index_t>(kept.size());
  Matrix<host_space, index_t> indices({blocks, dimension - 2});
  Tensor<3, host_space, data_t> values({first, last, blocks});
  auto indices_view = indices.view();
  data_t* out = values.data();
  for (index_t k = 0; k < blocks; ++k)
  {
    index_t mid = kept[static_cast<size_t>(k)];
    for (size_t j = 0; j < dimension - 2; ++j)
    {
      indices_view({k, static_cast<index_t>(j)}) = mid % sizes[j + 1];
      mid /= sizes[j + 1];
    }
    const index_t source = kept[static_cast<size_t>(k)];
    for (index_t b = 0; b < last; ++b)
    {
      for (index_t a = 0; a < first; ++a)
      {
        out[a + first * (b + last * k)] = in[a + first * (source + interior_size * b)];
      }
    }
  }
  return BCOOTensor<dimension, space, data_t>(
    BCOOTensor<dimension, host_space, data_t>(sizes, indices, values, dense.name()));
}

/**
 * \brief Converts a sparse (COO) tensor to block coordinate format.
 *
 * Every interior coordinate with at least one stored entry becomes a dense block; entries absent from the COO tensor
 * are zero in the block. Runs on host.
 */
template <size_t dimension, execution_space space, typename data_t>
  requires(dimension >= 3)
BCOOTensor<dimension, space, data_t> convert_to_BCOOTensor(SparseTensor<dimension, space, data_t> const& sparse)
{
  SparseTensor<dimension, host_space, data_t> const host = sparse;
  const auto sizes = host.sizes();
  const index_t first = sizes[0];
  const index_t last = sizes[dimension - 1];
  const auto linear_view = host.linear_indices().const_view();
  const auto values_view = host.values().const_view();
  std::map<index_t, index_t> block_of_mid;
  std::vector<Array<index_t, dimension - 2>> coordinates;
  for (index_t entry = 0; entry < host.nnz(); ++entry)
  {
    index_t a = 0;
    index_t b = 0;
    Array<index_t, dimension - 2> interior{};
    detail::split_bcoo_index<dimension>(sizes, linear_view({entry}), a, interior, b);
    const index_t mid = detail::interior_linear_index<dimension>(sizes, interior);
    if (block_of_mid.emplace(mid, static_cast<index_t>(coordinates.size())).second)
    {
      coordinates.push_back(interior);
    }
  }
  const auto blocks = static_cast<index_t>(coordinates.size());
  Matrix<host_space, index_t> indices({blocks, dimension - 2});
  Tensor<3, host_space, data_t> values({first, last, blocks});
  values.fill_with_zeros();
  auto indices_view = indices.view();
  for (index_t k = 0; k < blocks; ++k)
  {
    for (size_t j = 0; j < dimension - 2; ++j)
    {
      indices_view({k, static_cast<index_t>(j)}) = coordinates[static_cast<size_t>(k)][j];
    }
  }
  data_t* out = values.data();
  for (index_t entry = 0; entry < host.nnz(); ++entry)
  {
    index_t a = 0;
    index_t b = 0;
    Array<index_t, dimension - 2> interior{};
    detail::split_bcoo_index<dimension>(sizes, linear_view({entry}), a, interior, b);
    const index_t k = block_of_mid[detail::interior_linear_index<dimension>(sizes, interior)];
    out[a + first * (b + last * k)] += values_view({entry});
  }
  return BCOOTensor<dimension, space, data_t>(
    BCOOTensor<dimension, host_space, data_t>(sizes, indices, values, host.name()));
}

/**
 * \brief Converts a block coordinate tensor to a sparse (COO) tensor storing every entry of every stored block.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> convert_to_SparseTensor(BCOOTensor<dimension, space, data_t> const& bcoo)
{
  BCOOTensor<dimension, host_space, data_t> const host = bcoo;
  const auto& sizes = host.sizes();
  const index_t first = sizes[0];
  const index_t last = sizes[dimension - 1];
  index_t interior_size = 1;
  for (size_t j = 1; j + 1 < dimension; ++j)
  {
    interior_size *= sizes[j];
  }
  Vector<host_space, index_t> linear({host.nnz()});
  Vector<host_space, data_t> values({host.nnz()});
  auto linear_view = linear.view();
  auto values_view = values.view();
  auto indices_view = host.m_indices.const_view();
  const data_t* blocks = host.m_values.data();
  index_t entry = 0;
  for (index_t k = 0; k < host.nonzero_blocks(); ++k)
  {
    Array<index_t, dimension - 2> interior{};
    for (size_t j = 0; j < dimension - 2; ++j)
    {
      interior[j] = indices_view({k, static_cast<index_t>(j)});
    }
    const index_t mid = detail::interior_linear_index<dimension>(sizes, interior);
    for (index_t b = 0; b < last; ++b)
    {
      for (index_t a = 0; a < first; ++a, ++entry)
      {
        linear_view({entry}) = a + first * (mid + interior_size * b);
        values_view({entry}) = blocks[a + first * (b + last * k)];
      }
    }
  }
  SparseTensor<dimension, host_space, data_t> host_sparse(sizes, std::move(linear), std::move(values), host.name());
  return SparseTensor<dimension, space, data_t>(host_sparse);
}

/**
 * \brief Permutes the interior modes of a block coordinate tensor in place.
 *
 * \param permutation Mode permutation, as for SparseTensor: output mode `i` is input mode `permutation[i]`.
 *        The first and last modes (the block modes) must stay in place.
 */
template <execution_space space, size_t dimension, typename data_t>
void permute(BCOOTensor<dimension, space, data_t>& bcoo, Array<index_t, dimension> const& permutation)
{
  boba_always_assert(permutation[0] == 0 && permutation[dimension - 1] == dimension - 1,
                     "BCOOTensor permute: the first and last (block) modes must stay in place");
  BCOOTensor<dimension, host_space, data_t> host = bcoo;
  Array<index_t, dimension> sizes = host.sizes();
  for (size_t i = 0; i < dimension; ++i)
  {
    sizes[i] = host.sizes(static_cast<size_t>(permutation[i]));
  }
  Matrix<host_space, index_t> indices({host.nonzero_blocks(), dimension - 2});
  auto in_view = host.m_indices.const_view();
  auto out_view = indices.view();
  for (index_t k = 0; k < host.nonzero_blocks(); ++k)
  {
    for (size_t i = 1; i + 1 < dimension; ++i)
    {
      out_view({k, static_cast<index_t>(i - 1)}) = in_view({k, permutation[i] - 1});
    }
  }
  bcoo = BCOOTensor<dimension, space, data_t>(
    BCOOTensor<dimension, host_space, data_t>(sizes, indices, host.m_values, host.name()));
}

/**
 * \brief Concept satisfied by the sparse (non-dense) operator-core formats: SparseTensor (COO) and BCOOTensor.
 */
template <typename tensor_t>
concept IsSparseCoreType = IsSparseTensorType<tensor_t> or IsBCOOTensorType<tensor_t>;

/**
 * \brief One core of a TT-matrix/TT-vector product for a block coordinate matrix core.
 *
 * out((ra1, rx1), n, (ra2, rx2)) = sum_m A(ra1, n, m, ra2) x(rx1, m, rx2), with (ra1, rx1) and (ra2, rx2) flattened
 * first index fastest, as in SparseTensorTrainMatrix's COO product.
 */
template <execution_space space, typename data_t>
Tensor<3, space, data_t> matrix_core_vector_core_product(const BCOOTensor<4, space, data_t>& A,
                                                          const Tensor<3, space, data_t>& x)
{
  static_assert(space == host_space, "BCOOTensor tensor-train kernels are host-only");
  const index_t R1 = A.sizes(0);
  const index_t N = A.sizes(1);
  const index_t M = A.sizes(2);
  const index_t R2 = A.sizes(3);
  const index_t X1 = x.sizes(0);
  const index_t X2 = x.sizes(2);
  boba_always_assert_equal(x.sizes(1), M, "BCOO TT matvec: core columns must match the vector mode");
  Tensor<3, space, data_t> out({R1 * X1, N, R2 * X2});
  out.fill_with_zeros();
  data_t* o = out.data();
  const data_t* xp = x.data();
  const data_t* block_data = A.m_values.data();
  const auto indices_view = A.m_indices.const_view();
  for (index_t q = 0; q < A.nonzero_blocks(); ++q)
  {
    const index_t n = indices_view({q, 0});
    const index_t m = indices_view({q, 1});
    const data_t* a = block_data + q * R1 * R2;
    for (index_t rx2 = 0; rx2 < X2; ++rx2)
    {
      for (index_t ra2 = 0; ra2 < R2; ++ra2)
      {
        for (index_t rx1 = 0; rx1 < X1; ++rx1)
        {
          const data_t xv = xp[rx1 + X1 * (m + M * rx2)];
          data_t* column = o + R1 * rx1 + R1 * X1 * (n + N * (ra2 + R2 * rx2));
          const data_t* a_column = a + R1 * ra2;
          for (index_t ra1 = 0; ra1 < R1; ++ra1)
          {
            column[ra1] += a_column[ra1] * xv;
          }
        }
      }
    }
  }
  return out;
}

} // namespace boba
