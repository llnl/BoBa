// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Tensor-train matrix whose cores store only explicitly present entries.
 *
 * Each core has shape `(r_left, rows, cols, r_right)` and is represented by a
 * `SparseTensor<4, space, data_t>`. Sparse entries not present in a core are
 * interpreted as zero. Operations that require dense tensor-train algorithms,
 * such as rounding and matvec, are intentionally not part of this type yet.
 */
template <size_t dimension, execution_space space, typename _data_t>
struct SparseTensorTrainMatrix
{
  using data_t = _data_t;
  using core_t = SparseTensor<4, space, data_t>;

  std::string m_name = "SparseTensorTrainMatrix";
  Array<core_t, dimension> cores;

  /**
   * \brief Constructs a zero sparse tensor-train matrix with the given mode sizes.
   *
   * \param input_rows Row extent of each matrix mode.
   * \param input_cols Column extent of each matrix mode.
   */
  SparseTensorTrainMatrix(
    Array<size_t, dimension> input_rows,
    Array<size_t, dimension> input_cols)
  {
    resize(input_rows, input_cols);
  }

  /**
   * \brief Constructs an empty sparse tensor-train matrix.
   */
  SparseTensorTrainMatrix() = default;

  /**
   * \brief Copies a sparse tensor-train matrix in the same execution space.
   */
  SparseTensorTrainMatrix(SparseTensorTrainMatrix const&) = default;

  /**
   * \brief Moves a sparse tensor-train matrix.
   */
  SparseTensorTrainMatrix(SparseTensorTrainMatrix&&) = default;

  /**
   * \brief Assigns a sparse tensor-train matrix in the same execution space.
   */
  SparseTensorTrainMatrix& operator=(SparseTensorTrainMatrix const&) = default;

  /**
   * \brief Moves a sparse tensor-train matrix into this object.
   */
  SparseTensorTrainMatrix& operator=(SparseTensorTrainMatrix&&) noexcept = default;

  /**
   * \brief Copies a sparse tensor-train matrix from another execution space.
   *
   * \param rhs Sparse tensor-train matrix to copy.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  SparseTensorTrainMatrix(SparseTensorTrainMatrix<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name)
  {
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d] = rhs.cores[d];
    }
  }

  /**
   * \brief Assigns a sparse tensor-train matrix from another execution space.
   *
   * \param rhs Sparse tensor-train matrix to copy.
   * \return This sparse tensor-train matrix.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  SparseTensorTrainMatrix& operator=(SparseTensorTrainMatrix<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d] = rhs.cores[d];
    }
    return *this;
  }

  /**
   * \brief Resizes the matrix and clears every sparse core.
   *
   * Existing entries are discarded and every core is reset to rank one.
   *
   * \param input_rows Row extent of each matrix mode.
   * \param input_cols Column extent of each matrix mode.
   */
  void resize(Array<index_t, dimension> input_rows, Array<index_t, dimension> input_cols)
  {
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d] = core_t({1, input_rows[d], input_cols[d], 1});
    }
  }

  /**
   * \brief Renames the matrix and its cores.
   *
   * \param new_name New matrix name.
   */
  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
  }

  /**
   * \brief Returns the matrix name.
   */
  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Returns the number of cores.
   */
  static constexpr size_t get_number_cores() noexcept
  {
    return dimension;
  }

  /**
   * \brief Returns the tensor-train dimension.
   */
  static constexpr size_t get_dimension() noexcept
  {
    return dimension;
  }

  /**
   * \brief Returns the execution space used by the sparse cores.
   */
  static constexpr execution_space get_space() noexcept
  {
    return space;
  }

  /**
   * \brief Returns the row extent of core \p d.
   */
  size_t core_rows(size_t d) const
  {
    boba_always_assert_lt(d, dimension, "invalid core number");
    return static_cast<size_t>(cores[d].sizes(1));
  }

  /**
   * \brief Returns the column extent of core \p d.
   */
  size_t core_cols(size_t d) const
  {
    boba_always_assert_lt(d, dimension, "invalid core number");
    return static_cast<size_t>(cores[d].sizes(2));
  }

  /**
   * \brief Returns all core row extents.
   */
  Array<size_t, dimension> core_rows() const
  {
    Array<size_t, dimension> output = filled_array<dimension>(0_z);
    for (size_t d = 0; d < dimension; ++d)
    {
      output[d] = core_rows(d);
    }
    return output;
  }

  /**
   * \brief Returns all core column extents.
   */
  Array<size_t, dimension> core_cols() const
  {
    Array<size_t, dimension> output = filled_array<dimension>(0_z);
    for (size_t d = 0; d < dimension; ++d)
    {
      output[d] = core_cols(d);
    }
    return output;
  }

  /**
   * \brief Returns the number of scalar entries in one dense core matrix.
   */
  size_t get_number_matrix_elements(size_t d) const
  {
    return core_rows(d) * core_cols(d);
  }

  /**
   * \brief Returns the number of explicitly stored entries in all cores.
   */
  index_t get_number_elements() const noexcept
  {
    index_t output = 0;
    for (size_t d = 0; d < dimension; ++d)
    {
      output += cores[d].number_nonzeros();
    }
    return output;
  }

  /**
   * \brief Returns the product of the row-mode extents.
   */
  size_t rows() const
  {
    return product(core_rows());
  }

  /**
   * \brief Returns the product of the column-mode extents.
   */
  size_t cols() const
  {
    return product(core_cols());
  }

  /**
   * \brief Returns the logical matrix size.
   */
  size_t get_full_size() const
  {
    return rows() * cols();
  }

  /**
   * \brief Returns the row/column product for each tensor-train mode.
   */
  Array<size_t, dimension> sizes() const
  {
    Array<size_t, dimension> output = filled_array<dimension>(0_z);
    for (size_t d = 0; d < dimension; ++d)
    {
      output[d] = get_number_matrix_elements(d);
    }
    return output;
  }

  /**
   * \brief Returns the row/column product for tensor-train mode \p d.
   */
  size_t sizes(size_t d) const
  {
    return get_number_matrix_elements(d);
  }

  /**
   * \brief Returns the ratio of logical scalar size to sparse storage size.
   *
   * An empty sparse train has compression rate zero because it has no stored
   * entries from which to form a storage ratio.
   */
  [[nodiscard]]
  float compression_rate() const noexcept
  {
    const auto stored_size = get_number_elements();
    if (stored_size == 0)
    {
      return 0.0F;
    }
    const double ratio = static_cast<double>(get_full_size()) / static_cast<double>(stored_size);
    return static_cast<float>(std::floor(ratio * 100.0) / 100.0);
  }

  /**
   * \brief Returns the left rank of core \p d.
   */
  size_t get_ranks_left(size_t d) const
  {
    boba_always_assert_lt(d, dimension, "invalid core number");
    return static_cast<size_t>(cores[d].sizes(0));
  }

  /**
   * \brief Returns the right rank of core \p d.
   */
  size_t get_ranks_right(size_t d) const
  {
    boba_always_assert_lt(d, dimension, "invalid core number");
    return static_cast<size_t>(cores[d].sizes(3));
  }

  /**
   * \brief Returns all tensor-train ranks.
   */
  Array<size_t, dimension + 1> ranks() const
  {
    Array<size_t, dimension + 1> output = filled_array<dimension + 1>(0_z);
    for (size_t d = 0; d < dimension; ++d)
    {
      output[d] = get_ranks_left(d);
    }
    output[dimension] = get_ranks_right(dimension - 1);
    return output;
  }

  /**
   * \brief Returns all tensor-train ranks.
   */
  Array<size_t, dimension + 1> get_ranks() const
  {
    return ranks();
  }

  /**
   * \brief Returns the rank at a tensor-train boundary.
   */
  size_t ranks(size_t d) const
  {
    return d < dimension ? get_ranks_left(d) : get_ranks_right(dimension - 1);
  }

  /**
   * \brief Returns a string representation of the tensor-train ranks.
   */
  std::string ranks_string(size_t /*indent*/ = 1) const
  {
    return " ( " + make_delimited_string(ranks()) + " ) ";
  }

  /**
   * \brief Returns the flattened row index for a row-mode multi-index.
   */
  size_t row_index(Array<size_t, dimension> const& multiindex) const
  {
    return index(multiindex, core_rows());
  }

  /**
   * \brief Returns the flattened column index for a column-mode multi-index.
   */
  size_t col_index(Array<size_t, dimension> const& multiindex) const
  {
    return index(multiindex, core_cols());
  }

  /**
   * \brief Flattens a multi-index using the supplied mode sizes.
   */
  size_t index(Array<size_t, dimension> const& multiindex, Array<size_t, dimension> const& mode_sizes) const
  {
    return Multiindexer<dimension>(mode_sizes).index(multiindex);
  }

  /**
   * \brief Replaces this train with a rank-one train representing a constant.
   *
   * The first core stores the supplied scalar and subsequent cores store one.
   * Only nonzero entries are explicitly stored.
   */
  void fill_with(data_t value)
  {
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d] = make_constant_core(core_rows(d), core_cols(d), d == 0 ? value : data_t(1));
    }
  }

  /**
   * \brief Replaces this train with a rank-one train representing zero.
   */
  void fill_with_zeros()
  {
    for (size_t d = 0; d < dimension; ++d)
    {
      cores[d] = core_t({1, static_cast<index_t>(core_rows(d)), static_cast<index_t>(core_cols(d)), 1});
    }
  }

  /**
   * \brief Applies this sparse matrix to a tensor-train vector.
   *
   * This is the host-side reference implementation. Each sparse matrix core
   * is contracted with the corresponding dense vector core, and the resulting
   * dense core is reshaped into the output tensor train. The low-memory
   * implementation used by the dense matrix class is intentionally omitted.
   *
   * \param input Tensor-train vector to multiply.
   * \return The resulting tensor-train vector.
   */
  TensorTrain<dimension, space, data_t> TensorTrainMatrix_vector_multiply(
    TensorTrain<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    TensorTrain<dimension, space, data_t> output(input.sizes());
    output.svd_tolerance_relative = input.svd_tolerance_relative;
    output.svd_tolerance_absolute = input.svd_tolerance_absolute;

    if constexpr (space != host_space)
    {
      boba_error("not yet implemented");
      return output;
    }

    for (size_t d = 0; d < dimension; ++d)
    {
      boba_always_assert_equal(core_cols(d), input.sizes(d), "invalid size and cols for tensor-train matvec");

      const Array<size_t, 2> new_rank_left{get_ranks_left(d), input.get_ranks_left(d)};
      const Array<size_t, 2> new_rank_right{get_ranks_right(d), input.get_ranks_right(d)};
      const auto ranks_left_view = Multiindexer<2>(new_rank_left);
      const auto ranks_right_view = Multiindexer<2>(new_rank_right);
      const size_t new_ranks_left = ranks_left_view.size();
      const size_t new_ranks_right = ranks_right_view.size();

      if ((new_ranks_left == 0) || (new_ranks_right == 0))
      {
        output.fill_with_zeros();
        return output;
      }

      auto temporary = tensor_contraction<1>(
        {"ml", "row", "col", "mr"},
        cores[d],
        {"l", "col", "r"},
        input.cores[d],
        {"ml", "l", "row", "mr", "r"});
      output.cores[d] = reshape<3>(temporary, {new_ranks_left, core_rows(d), new_ranks_right});
    }

    checkpoint();
    return output;
  }

  /**
   * \brief Applies this sparse matrix to a tensor-train vector.
   *
   * \param input Tensor-train vector to multiply.
   * \return The resulting tensor-train vector when implemented.
   */
  TensorTrain<dimension, space, data_t> apply(
    TensorTrain<dimension, space, data_t> const& input) const
  {
    return TensorTrainMatrix_vector_multiply(input);
  }

  /**
   * \brief Returns a copy with every core's row and column modes transposed.
   */
  SparseTensorTrainMatrix transpose() const
  {
    SparseTensorTrainMatrix output = *this;
    for (size_t d = 0; d < dimension; ++d)
    {
      permute<space, 4, data_t>(output.cores[d], Array<index_t, 4>{0, 2, 1, 3});
    }
    return output;
  }

};

/**
 * \brief Converts a sparse tensor-train matrix to a dense tensor-train matrix.
 *
 * The sparse cores are expanded without changing their tensor-train ranks.
 * Sparse entries not present in a core remain zero in the dense core.
 *
 * \param sparse Sparse tensor-train matrix to convert.
 * \return Dense tensor-train matrix with the same cores and name.
 */
template <size_t dimension, execution_space output_space, execution_space input_space, typename data_t>
TensorTrainMatrix<dimension, output_space, data_t>
to_dense(SparseTensorTrainMatrix<dimension, input_space, data_t> const& sparse)
{
  TensorTrainMatrix<dimension, output_space, data_t> output(sparse.core_rows(), sparse.core_cols());
  output.rename(sparse.name());

  for (size_t d = 0; d < dimension; ++d)
  {
    output.cores[d] = to_dense(sparse.cores[d]);
  }

  return output;
}

/**
 * \brief Converts a sparse tensor-train matrix to a dense matrix in its input space.
 */
template <size_t dimension, execution_space input_space, typename data_t>
TensorTrainMatrix<dimension, input_space, data_t>
to_dense(SparseTensorTrainMatrix<dimension, input_space, data_t> const& sparse)
{
  return to_dense<dimension, input_space, input_space, data_t>(sparse);
}

/**
 * \brief Converts a dense tensor-train matrix to a sparse tensor-train matrix.
 *
 * Every dense core entry with magnitude greater than \p tolerance is retained.
 * The input and output execution spaces may differ; sparse core copy
 * construction performs the required transfer.
 *
 * \param ttm Dense tensor-train matrix to convert.
 * \param tolerance Magnitude threshold for retaining core entries.
 * \return Sparse tensor-train matrix containing the retained core entries.
 */
template <size_t dimension, execution_space output_space, execution_space input_space, typename data_t>
SparseTensorTrainMatrix<dimension, output_space, data_t>
from_dense(
  TensorTrainMatrix<dimension, input_space, data_t> const& ttm,
  real_type_t<data_t> tolerance = static_cast<real_type_t<data_t>>(0))
{
  SparseTensorTrainMatrix<dimension, output_space, data_t> output(ttm.core_rows(), ttm.core_cols());
  output.rename(ttm.m_name);

  for (size_t d = 0; d < dimension; ++d)
  {
    auto filter = [tolerance](data_t value, auto /*indices*/)
    {
      return ::boba::abs(value) > tolerance;
    };
    auto sparse_core = convert_to_SparseTensor(ttm.cores[d], filter);
    output.cores[d] = sparse_core;
  }
  return output;
}

/**
 * \brief Converts a dense tensor-train matrix to a sparse matrix in its input space.
 *
 * This overload allows the output execution space to be deduced from the dense
 * input. Use the fully specified overload when a different output space is
 * required.
 *
 * \param ttm Dense tensor-train matrix to convert.
 * \param tolerance Magnitude threshold for retaining core entries.
 * \return Sparse tensor-train matrix containing the retained core entries.
 */
template <size_t dimension, execution_space input_space, typename data_t>
SparseTensorTrainMatrix<dimension, input_space, data_t>
from_dense(
  TensorTrainMatrix<dimension, input_space, data_t> const& ttm,
  real_type_t<data_t> tolerance = static_cast<real_type_t<data_t>>(0))
{
  return from_dense<dimension, input_space, input_space, data_t>(ttm, tolerance);
}

} // namespace boba
