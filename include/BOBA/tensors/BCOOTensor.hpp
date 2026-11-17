// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/Matrix.hpp"
#include "BOBA/tensors/Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace boba
{

/**
 * \brief Block coordinate sparse tensor: dense blocks over the first and last modes, stored at coordinates of the
 * interior modes.
 *
 * For a tensor `T(i_0, i_1, ..., i_{d-2}, i_{d-1})`, each stored interior coordinate `(i_1, ..., i_{d-2})` carries the
 * dense block `T(:, i_1, ..., i_{d-2}, :)` of shape `(sizes(0), sizes(d-1))`; missing coordinates are zero blocks.
 * This is the sparsity of tensor-train cores whose physical modes are sparse while the rank modes are dense: a
 * TT-matrix core `A(ra1, n, m, ra2)` stores one dense `ra1 x ra2` block per nonzero `(n, m)` pair, so contractions
 * over the blocks run as dense matrix products.
 *
 * `m_values` has shape `(sizes(0), sizes(d-1), nonzero_blocks())` and `m_indices` has shape
 * `(nonzero_blocks(), d-2)`; block `k` sits at interior coordinate `m_indices(k, :)`.
 *
 * Invariant: blocks are unique and sorted column-major in the interior coordinates (the last interior mode varies
 * slowest), the order in which the tensor-train kernels traverse them.
 *
 * Construction from coordinate lists sorts and coalesces on the host; the contraction kernels are host-only.
 */
template <size_t _dimension, execution_space space, typename _data_t>
struct BCOOTensor
{
  static_assert(_dimension >= 3, "BCOOTensor needs a first mode, at least one interior mode, and a last mode");

  static constexpr size_t dimension = _dimension;
  static constexpr size_t interior_dimension = _dimension - 2;
  static constexpr std::string_view object_type_name = "BCOOTensor";

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;
  using index_array = Array<index_t, dimension>;

  index_array m_sizes{};
  Tensor<3, space, data_t> m_values;
  Matrix<space, index_t> m_indices;
  std::string m_name{object_type_name};

  BCOOTensor() = default;
  BCOOTensor(BCOOTensor const&) = default;
  BCOOTensor(BCOOTensor&&) = default;
  BCOOTensor& operator=(BCOOTensor const&) = default;
  BCOOTensor& operator=(BCOOTensor&&) = default;
  ~BCOOTensor() = default;

  /**
   * \brief Constructs an empty (all-zero) tensor with the requested extents.
   */
  explicit BCOOTensor(index_array sizes, std::string_view name = object_type_name)
      : m_sizes(sizes),
        m_values({sizes[0], sizes[dimension - 1], 0}),
        m_indices({0, interior_dimension}),
        m_name(name)
  {
  }

  /**
   * \brief Constructs a tensor from block coordinates and dense blocks.
   *
   * \param sizes Tensor extents.
   * \param indices Interior coordinates, shape `(blocks, d-2)`, in any order.
   * \param values Dense blocks, shape `(sizes(0), sizes(d-1), blocks)`, matching `indices`.
   *
   * Blocks with equal coordinates are summed.
   */
  BCOOTensor(index_array sizes,
             Matrix<space, index_t> const& indices,
             Tensor<3, space, data_t> const& values,
             std::string_view name = object_type_name)
      : m_sizes(sizes),
        m_name(name)
  {
    boba_always_assert_equal(indices.sizes(1), interior_dimension, "BCOOTensor: one index column per interior mode");
    boba_always_assert_equal(values.sizes(0), sizes[0], "BCOOTensor: block rows must match the first mode");
    boba_always_assert_equal(values.sizes(1), sizes[dimension - 1], "BCOOTensor: block columns must match the last mode");
    boba_always_assert_equal(values.sizes(2), indices.sizes(0), "BCOOTensor: one block per coordinate");
    Matrix<host_space, index_t> host_indices = indices;
    Tensor<3, host_space, data_t> host_values = values;
    sort_and_coalesce(host_indices, host_values);
    m_indices = host_indices;
    m_values = host_values;
  }

  /**
   * \brief Copies a block coordinate tensor from another execution space.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  BCOOTensor(BCOOTensor<dimension, rhs_space, data_t> const& rhs)
      : m_sizes(rhs.m_sizes),
        m_values(rhs.m_values),
        m_indices(rhs.m_indices),
        m_name(rhs.m_name)
  {
  }

  /**
   * \brief Assigns a block coordinate tensor from another execution space.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  BCOOTensor& operator=(BCOOTensor<dimension, rhs_space, data_t> const& rhs)
  {
    m_sizes = rhs.m_sizes;
    m_values = rhs.m_values;
    m_indices = rhs.m_indices;
    m_name = rhs.m_name;
    return *this;
  }

  /** \brief Returns the execution space of the block and coordinate storage. */
  static constexpr execution_space get_space() noexcept
  {
    return space;
  }

  /** \brief Returns the tensor extents. */
  index_array const& sizes() const noexcept
  {
    return m_sizes;
  }

  /** \brief Returns the extent of mode \p d. */
  index_t sizes(size_t d) const
  {
    boba_always_assert_lt(d, dimension, "BCOOTensor: invalid mode");
    return m_sizes[d];
  }

  /** \brief Returns the logical number of entries. */
  index_t size() const noexcept
  {
    return product(m_sizes);
  }

  /** \brief Returns the number of stored blocks. */
  index_t nonzero_blocks() const noexcept
  {
    return m_indices.sizes(0);
  }

  /** \brief Returns the number of stored scalar entries (blocks times block size). */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /** \brief Alias of nnz() matching SparseTensor. */
  index_t number_nonzeros() const noexcept
  {
    return nnz();
  }

  /**
   * \brief Returns the ratio of logical size to storage (values plus coordinates); zero when nothing is stored.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    const double stored = static_cast<double>(m_values.size()) + static_cast<double>(m_indices.size());
    if (stored <= 0.0)
    {
      return 0.0F;
    }
    return static_cast<float>(std::floor(static_cast<double>(size()) / stored * 100.0) / 100.0);
  }

  /** \brief Returns the tensor name. */
  std::string const& name() const noexcept
  {
    return m_name;
  }

  /** \brief Renames the tensor. */
  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  /**
   * \brief Sorts blocks column-major in their interior coordinates and sums blocks with equal coordinates.
   */
  static void sort_and_coalesce(Matrix<host_space, index_t>& indices, Tensor<3, host_space, data_t>& values)
  {
    const index_t blocks = indices.sizes(0);
    const index_t block_size = values.sizes(0) * values.sizes(1);
    auto indices_view = indices.const_view();
    std::vector<index_t> order(static_cast<size_t>(blocks));
    std::iota(order.begin(), order.end(), index_t{0});
    auto column_major_less = [&](index_t a, index_t b)
    {
      for (size_t j = interior_dimension; j-- > 0;)
      {
        const index_t ia = indices_view({a, static_cast<index_t>(j)});
        const index_t ib = indices_view({b, static_cast<index_t>(j)});
        if (ia != ib)
        {
          return ia < ib;
        }
      }
      return false;
    };
    std::stable_sort(order.begin(), order.end(), column_major_less);

    index_t unique = 0;
    for (index_t q = 0; q < blocks; ++q)
    {
      if (q == 0 || column_major_less(order[static_cast<size_t>(q - 1)], order[static_cast<size_t>(q)]))
      {
        ++unique;
      }
    }

    Matrix<host_space, index_t> sorted_indices({unique, interior_dimension});
    Tensor<3, host_space, data_t> sorted_values({values.sizes(0), values.sizes(1), unique});
    sorted_values.fill_with_zeros();
    auto sorted_indices_view = sorted_indices.view();
    const data_t* source = values.data();
    data_t* target = sorted_values.data();
    index_t k = 0;
    for (index_t q = 0; q < blocks; ++q)
    {
      const index_t block = order[static_cast<size_t>(q)];
      if (q > 0 && column_major_less(order[static_cast<size_t>(q - 1)], block))
      {
        ++k;
      }
      for (size_t j = 0; j < interior_dimension; ++j)
      {
        sorted_indices_view({k, static_cast<index_t>(j)}) = indices_view({block, static_cast<index_t>(j)});
      }
      for (index_t entry = 0; entry < block_size; ++entry)
      {
        target[k * block_size + entry] += source[block * block_size + entry];
      }
    }
    indices = std::move(sorted_indices);
    values = std::move(sorted_values);
  }
};

template <typename tensor_t>
struct is_bcoo_tensor : std::false_type
{
};

template <size_t dimension, execution_space space, typename data_t>
struct is_bcoo_tensor<BCOOTensor<dimension, space, data_t>> : std::true_type
{
};

/**
 * \brief Concept satisfied by BCOOTensor specializations, including cv/ref-qualified types.
 */
template <typename tensor_t>
concept IsBCOOTensorType = is_bcoo_tensor<std::remove_cvref_t<tensor_t>>::value;

} // namespace boba
