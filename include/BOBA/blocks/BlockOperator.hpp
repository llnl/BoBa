// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "BOBA/blocks/BlockVector.hpp"

namespace boba
{
/**
 *  Support structure for representing block matrices easily, e.g.
 *  A = [A_11 A_12
 *       A_21 A_22]
 */

template <typename operator_t>
struct BlockOperator
{
  using data_t = typename operator_t::data_t;
  using real_data_t = typename operator_t::real_data_t;

  size_t block_rows = 0;
  size_t block_cols = 0;
  std::vector<operator_t> operator_blocks;
  std::string m_name;

  /// \brief default constructor
  BlockOperator() = default;

  /**
   * \brief Construct a block operator with fixed block dimensions.
   *
   * \param[in] _block_rows number of block rows
   * \param[in] _block_cols number of block columns
   */
  BlockOperator(size_t _block_rows, size_t _block_cols)
  {
    block_rows = _block_rows;
    block_cols = _block_cols;
    operator_blocks.resize(block_rows * block_cols);
  }

  /**
   *  Access an element of the BlockOperator
   *  Example:  example_matrix({2, 3})
   */

  /**
   * \brief Access a block by row and column.
   *
   * \param[in] rc block coordinates
   * \return reference to the requested block
   */
  operator_t& operator()(::boba::Array<size_t, 2> rc)
  {
    auto row = rc[0];
    auto col = rc[1];
    return operator_blocks.at(row + block_rows * col);
  }

  /**
   *  const accessor
   */

  /**
   * \brief Access a block by row and column, const.
   *
   * \param[in] rc block coordinates
   * \return const reference to the requested block
   */
  operator_t const& operator()(::boba::Array<size_t, 2> rc) const
  {
    auto row = rc[0];
    auto col = rc[1];
    return operator_blocks.at(row + block_rows * col);
  }

  /**
   *  Action of the BlockOperator on a BlockVector
   */

  /**
   * \brief Apply the block operator to a block vector.
   *
   * \param[in] vec input block vector
   * \return resulting block vector
   */
  template <typename vector_t>
  [[nodiscard]]
  BlockVector<vector_t> operator*(const BlockVector<vector_t>& vec) const
  {
    BOBA_CALI_MARK
    BlockVector<vector_t> output_vector(vec.block_size);
    for (size_t row = 0; row < block_rows; row++)
    {
      output_vector(row) = this->operator()({row, 0}) * vec(0);
      for (size_t col = 1; col < block_cols; col++)
      {
        output_vector(row) += this->operator()({row, col}) * vec(col);
      }
      output_vector(row).round();
    }
    return output_vector;
  }

  void savetxt(std::string name = "matrix", std::string ext = "") const
  {
    for (size_t row = 0; row < block_rows; row++)
    {
      for (size_t col = 0; col < block_cols; col++)
      {
        boba::savetxt(operator_blocks.at(row + block_rows * col), name + "_" + std::to_string(row) + "_" + std::to_string(col) + ext);
      }
    }
  }

  /**
   * \brief Get the block operator name.
   *
   * \return current name
   */
  std::string const& name() const noexcept
  {
    return m_name;
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  /**
   * \brief Get the total compressed element count across all blocks.
   *
   * \return total number of stored elements
   */
  [[nodiscard]]
  size_t get_number_elements() const
  {
    size_t count = 0;

    for (size_t row = 0; row < block_rows; row++)
    {
      for (size_t col = 0; col < block_cols; col++)
      {
        count += operator_blocks.at(row + block_rows * col).get_number_elements();
      }
    }
    return count;
  }

  /**
   * \brief Get the total full size across all blocks.
   *
   * \return total full size
   */
  [[nodiscard]]
  size_t get_full_size() const
  {
    size_t count = 0;
    for (size_t row = 0; row < block_rows; row++)
    {
      for (size_t col = 0; col < block_cols; col++)
      {
        count += operator_blocks.at(row + block_rows * col).get_full_size();
      }
    }
    return count;
  }

  /**
   * \brief Get the total row dimension represented by the block operator.
   *
   * \return total row count
   */
  [[nodiscard]]
  size_t rows() const
  {
    size_t count = 0;

    for (size_t row = 0; row < block_rows; row++)
    {
      count += operator_blocks.at(row + block_rows * 0).rows();
    }
    return count;
  }

  /**
   * \brief Get the total column dimension represented by the block operator.
   *
   * \return total column count
   */
  [[nodiscard]]
  size_t cols() const
  {
    size_t count = 0;

    for (size_t col = 0; col < block_cols; col++)
    {
      count += operator_blocks.at(0 + block_rows * col).cols();
    }
    return count;
  }

  /**
   * \brief Returns the ratio of the full size over the compressed size, truncated to two digits
   */

  /**
   * \brief Return the ratio of full size to compressed size, truncated to two digits.
   *
   * \return truncated compression rate
   */
  [[nodiscard]]
  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  /**
   * Counts the number of nonzero elements based on a tolerance, like MATLAB's nnz
   */

  /**
   * \brief Count blocks with entries above a tolerance.
   *
   * \param[in] tolerance threshold used to treat entries as zero
   * \return number of nonzero elements
   */
  [[nodiscard]]
  index_t number_nonzeros(const data_t tolerance = 1.0e-15) const
  {
    size_t count = 0;
    for (size_t row = 0; row < block_rows; row++)
    {
      for (size_t col = 0; col < block_cols; col++)
      {
        count += operator_blocks.at(row + block_rows * col).number_nonzeros(tolerance);
      }
    }
    return count;
  }

  /**
   * Computes the sparsity ratio consistent with matrix sparsity
   */

  /**
   * \brief Compute the sparsity ratio consistent with matrix sparsity.
   *
   * \param[in] tolerance threshold used to treat entries as zero
   * \return full size divided by the number of nonzero elements
   */
  [[nodiscard]]
  data_t sparsity(const data_t tolerance = 1.0e-15) const
  {
    data_t full_size = this->get_full_size();
    data_t nnz = this->number_nonzeros(tolerance);
    if (nnz == 0)
    {
      return data_t(0);
    }
    return full_size / nnz;
  }
};

template <typename operator_t>
typename operator_t::data_t norm_frobenius(
  const BlockOperator<operator_t>& block_operator)
{
  BOBA_CALI_MARK
  typename operator_t::data_t output = 0.0;
  for (size_t row = 0; row < block_operator.block_rows; row++)
  {
    for (size_t col = 0; col < block_operator.block_cols; col++)
    {
      auto norm_segment = ::boba::norm_frobenius(block_operator({row, col}));
      output += ::boba::pow(norm_segment, 2.0);
    }
  }
  return ::boba::sqrt(output);
}

template <typename operator_t>
typename operator_t::data_t norm_difference_frobenius(
  const BlockOperator<operator_t>& block_vec_a,
  const BlockOperator<operator_t>& block_vec_b)
{
  BOBA_CALI_MARK
  typename operator_t::data_t result = 0.0;
  for (size_t row = 0; row < block_vec_a.block_rows; row++)
  {
    for (size_t col = 0; col < block_vec_a.block_cols; col++)
    {
      result += norm_difference_frobenius(block_vec_a({row, col}), block_vec_b({row, col}));
    }
  }
  return boba::sqrt(result);
}

/**
 *  Action of the BlockOperator on a BlockOperator
 */

template <typename operator_t>
BlockOperator<operator_t> operator*(
  const BlockOperator<operator_t>& operator_A,
  const BlockOperator<operator_t>& operator_B)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(operator_A.block_cols, operator_B.block_rows, "Incompatible block sizes");
  BlockOperator<operator_t> output_operator(operator_A.block_rows, operator_B.block_cols);
  size_t block_common = operator_A.block_cols;

  for (size_t row = 0; row < output_operator.block_rows; row++)
  {
    for (size_t col = 0; col < output_operator.block_cols; col++)
    {
      output_operator({row, col}) = operator_A({row, 0}) * operator_B({0, col});
      for (size_t k = 1; k < block_common; k++)
      {
        output_operator({row, col}) += operator_A({row, k}) * operator_B({k, col});
      }
      output_operator({row, col}).round();
    }
  }
  return output_operator;
}

/**
 * @brief Left vector multiply of a matrix
 *
 * @param[in] vector_A BlockVector
 * @param[in] operator_B BlockOperator
 * @return BlockVector
 */

template <typename operator_t, typename vector_t>
BlockVector<vector_t> operator*(
  const BlockVector<vector_t>& vector_A,
  const BlockOperator<operator_t>& operator_B)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(vector_A.block_size, operator_B.block_rows, "Incompatible block sizes");
  BlockVector<vector_t> output_vector(operator_B.block_cols);
  for (size_t col = 0; col < output_vector.block_size; col++)
  {
    output_vector(col) = vector_A(0) * operator_B({0, col});
    for (size_t row = 1; row < operator_B.block_rows; row++)
    {
      output_vector(col) += vector_A(row) * operator_B({row, col});
    }
    output_vector(col).round();
  }
  return output_vector;
}

/**
 * @brief Checks the Frobenius norm of each of the blocks and generates a mask to be used with apply_masked
 * @return A mask that can be used in apply_masked
 */

template <typename operator_t>
Matrix<host_space, size_t> generate_mask(
  const BlockOperator<operator_t>& operator_A,
  const typename operator_t::data_t tolerance = 1.0e-13)
{
  BOBA_CALI_MARK
  auto block_rows = operator_A.block_rows;
  auto block_cols = operator_A.block_cols;

  Matrix<host_space, size_t> mask_a({block_rows, block_cols});
  auto mask_a_view = mask_a.view();

  for (size_t row = 0; row < block_rows; row++)
  {
    for (size_t col = 0; col < block_cols; col++)
    {
      bool is_nonzero = boba::product(operator_A({row, col}).sizes()) > 0;
      if (is_nonzero)
      {
        is_nonzero = ::boba::norm_frobenius(operator_A({row, col})) > tolerance;
      }
      mask_a_view({row, col}) = is_nonzero ? 1 : 0;
    }
  }
  return mask_a;
}

/**
 *  Action of the BlockOperator on a BlockOperator, with masking (only applies elements such that mask(i, j) > 0)
 */

template <typename operator_t>
BlockOperator<operator_t> apply_masked(
  const BlockOperator<operator_t>& operator_A,
  const BlockOperator<operator_t>& operator_B,
  const Matrix<host_space, size_t> mask_a,
  const Matrix<host_space, size_t> mask_b)
{
  BOBA_CALI_MARK
  BlockOperator<operator_t> output_operator(operator_A.block_rows, operator_B.block_cols);
  size_t block_common = operator_A.block_cols;

  for (size_t row = 0; row < output_operator.block_rows; row++)
  {
    for (size_t col = 0; col < output_operator.block_cols; col++)
    {
      // Compute for k > 0
      bool is_initialized = false;
      for (size_t k = 0; k < block_common; k++)
      {
        bool do_product = mask_a({row, k}) * mask_b({k, col}) > 0;
        if (do_product and is_initialized)
        {
          output_operator({row, col}) += operator_A({row, k}) * operator_B({k, col});
        }
        else if (do_product and not(is_initialized))
        {
          output_operator({row, col}) = operator_A({row, k}) * operator_B({k, col});
          is_initialized = true;
        }
      }
      if (is_initialized)
      {
        output_operator({row, col}).round();
      }
    }
  }
  return output_operator;
}

/**
 *  Action of the BlockOperator on a BlockOperator, with masking (only applies elements such that mask(i, j) > 0)
 */

template <typename operator_t, typename vector_t>
BlockVector<vector_t> apply_masked(
  const BlockOperator<operator_t>& block_sparse_operator,
  const BlockVector<vector_t>& vec,
  const Matrix<host_space, size_t> mask)
{
  BOBA_CALI_MARK
  BlockVector<vector_t> output_vector(block_sparse_operator.block_rows);

  for (size_t row = 0; row < block_sparse_operator.block_rows; row++)
  {
    bool is_initialized = false;
    for (size_t col = 0; col < block_sparse_operator.block_cols; col++)
    {
      bool do_product = mask({row, col}) > 0;
      if (do_product and is_initialized)
      {
        output_vector(row) += block_sparse_operator({row, col}) * vec(col);
      }
      else if (do_product and not(is_initialized))
      {
        output_vector(row) = block_sparse_operator({row, col}) * vec(col);
        is_initialized = true;
      }
    }

    if (is_initialized)
    {
      output_vector(row).round();
    }
    else
    {
      // TODO, this should be sized based on block_sparse_operator rows
      output_vector(row) = 0.0 * vec(0);
    }
  }

  return output_vector;
}

/**
 * @brief Convert a block matrix into an equivalent matrix
 *
 * @param[in] block_matrix BlockOperator<matrix>
 * @return matrix with values filled to be algebraically equivalent to the block matrix
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> unblock(const BlockOperator<Matrix<space, data_t>>& block_matrix)
{
  auto rows = 0_z;
  auto cols = rows;
  std::vector<size_t> rows_offets;
  std::vector<size_t> cols_offets;
  auto block_rows = block_matrix.block_rows;
  auto block_cols = block_matrix.block_cols;

  rows_offets.resize(block_rows + 1);
  cols_offets.resize(block_cols + 1);

  for (size_t r = 0; r < block_rows; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_matrix({r, 0}).rows();
  }
  for (size_t c = 0; c < block_cols; c++)
  {
    cols_offets.at(c) = cols;
    cols += block_matrix({0, c}).cols();
  }

  rows_offets.at(block_rows) = rows;
  cols_offets.at(block_cols) = cols;

  Matrix<space, data_t> unblocked({rows, cols});
  unblocked.fill_with_zeros();

  //
  // Consider all combinations of blocks
  //
  for (size_t r = 0; r < block_rows; r++)
  {
    for (size_t c = 0; c < block_cols; c++)
    {
      Array<index_t, 2> replacement_rows{rows_offets.at(r), rows_offets.at(r + 1)};
      Array<index_t, 2> replacement_cols{cols_offets.at(c), cols_offets.at(c + 1)};

      unblocked.replace_submatrix(replacement_rows, replacement_cols, block_matrix({r, c}));
    }
  }

  return unblocked;
}

/**
 * @brief Convert a block vector into an equivalent tt
 *
 * @param[in] block_vec BlockVector<vector>
 * @return vector with values filled to be algebraically equivalent to the block vector
 */

template <execution_space space, typename data_t>
Vector<space, data_t> unblock(const BlockVector<Vector<space, data_t>>& block_vec)
{
  auto rows = 0_z;
  std::vector<size_t> rows_offets;
  auto block_size = block_vec.block_size;

  rows_offets.resize(block_size + 1);

  for (size_t r = 0; r < block_size; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_vec(r).size();
  }

  rows_offets.at(block_size) = rows;

  Vector<space, data_t> unblocked({rows});
  unblocked.fill_with_zeros();

  //
  // Consider all combinations of blocks
  //
  for (size_t r = 0; r < block_size; r++)
  {
    Array<index_t, 2> replacement_rows{rows_offets.at(r), rows_offets.at(r + 1)};
    unblocked.replace_subvector(replacement_rows, block_vec(r));
  }

  return unblocked;
}

/**
 * @brief Convert a matrix into an equivalent BlockVector<ttm>, using offsets collected from block_ttm
 *
 * @param[in,out] block_ttm BlockVector<ttm> with core rows and columns set so that they add up to the sizes in full_tt, as these will be used to determine offsets from which to copy data
 * @param[in] full_ttm TTM from which core data will be copied
 */

template <execution_space space, typename data_t>
void copy_operator_into_block_form(
  BlockOperator<Matrix<space, data_t>>& block_matrix,
  const Matrix<space, data_t>& full_matrix)
{
  auto rows = 0_z;
  auto cols = rows;
  std::vector<size_t> rows_offets;
  std::vector<size_t> cols_offets;
  auto block_rows = block_matrix.block_rows;
  auto block_cols = block_matrix.block_cols;
  rows_offets.resize(block_rows + 1);
  cols_offets.resize(block_cols + 1);

  for (size_t r = 0; r < block_rows; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_matrix({r, 0}).rows();
  }
  for (size_t c = 0; c < block_cols; c++)
  {
    cols_offets.at(c) = cols;
    cols += block_matrix({0, c}).cols();
  }
  rows_offets.at(block_rows) = rows;
  cols_offets.at(block_cols) = cols;

  boba_always_assert_equal(rows, full_matrix.rows(), "Block matrix is not consistent with full matrix");
  boba_always_assert_equal(cols, full_matrix.cols(), "Block matrix is not consistent with full matrix");

  for (size_t d_row = 0; d_row < block_rows; d_row++)
  {
    for (size_t d_col = 0; d_col < block_cols; d_col++)
    {
      Array<index_t, 2> _rows{rows_offets.at(d_row), rows_offets.at(d_row + 1)};
      Array<index_t, 2> _cols{cols_offets.at(d_col), cols_offets.at(d_col + 1)};

      block_matrix({d_row, d_col}) = full_matrix.get_submatrix(_rows, _cols);
    }
  }
}

/**
 * @brief Convert a vector into an equivalent BlockVector<tt>, using offsets collected from block_tt
 * WARNING - this function should be used only if you really know what you are doing (see test_block_operator for example)
 *
 * @param[in,out] block_tt BlockVector<tt> with sizes set so that they add up to the sizes in full_tt, as these will be used to determine offsets from which to copy data
 * @param[in] full_tt TT from which core data will be copied
 */

template <execution_space space, typename data_t>
void copy_vector_into_block_form(
  BlockVector<Vector<space, data_t>>& block_vec,
  const Vector<space, data_t>& full_vector)
{
  auto rows = 0_z;
  std::vector<size_t> rows_offets;
  auto blocks = block_vec.block_size;
  rows_offets.resize(blocks + 1);

  for (size_t r = 0; r < blocks; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_vec(r).size();
  }
  rows_offets.at(blocks) = rows;

  boba_always_assert_equal(rows, full_vector.size(), "Block vector is not consistent with full vector");

  for (size_t d_row = 0; d_row < blocks; d_row++)
  {
    Array<index_t, 2> _rows{rows_offets.at(d_row), rows_offets.at(d_row + 1)};
    block_vec(d_row) = full_vector.get_subvector(_rows);
  }
}

/**
 * @brief Convert a block TensorTrainMatrix into an equivalent ttm
 *
 *
 * @param[in] block_ttm BlockOperator<ttm>
 * @return TensorTrainMatrix with values filled to be algebraically equivalent to the block ttm
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrainMatrix<dimension, space, data_t> unblock(const BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm_in)
{
  auto rows = filled_array<dimension>(0_z);
  auto cols = rows;
  std::vector<Array<size_t, dimension>> rows_offets;
  std::vector<Array<size_t, dimension>> cols_offets;
  auto block_rows = block_ttm_in.block_rows;
  auto block_cols = block_ttm_in.block_cols;
  rows_offets.resize(block_rows + 1);
  cols_offets.resize(block_cols + 1);

  for (size_t r = 0; r < block_rows; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_ttm_in({r, 0}).core_rows();
  }
  for (size_t c = 0; c < block_cols; c++)
  {
    cols_offets.at(c) = cols;
    cols += block_ttm_in({0, c}).core_cols();
  }

  rows_offets.at(block_rows) = rows;
  cols_offets.at(block_cols) = cols;

  TensorTrainMatrix<dimension, space, data_t> unblocked(rows, cols);
  unblocked.fill_with_zeros();

  //
  // Normalize the ranks
  //
  Array<index_t, dimension + 1> target_ranks = filled_array<dimension + 1>(1_z);
  for (size_t r = 0; r < block_rows; r++)
  {
    for (size_t c = 0; c < block_cols; c++)
    {
      target_ranks[0] = std::lcm(target_ranks[0], block_ttm_in({r, c}).get_ranks_left(0));
      for (size_t dim = 0; dim < dimension; dim++)
      {
        target_ranks[dim + 1] = std::lcm(target_ranks[dim + 1], block_ttm_in({r, c}).get_ranks_right(dim));
      }
    }
  }

  auto block_ttm_in_normalized = block_ttm_in;

  for (size_t r = 0; r < block_rows; r++)
  {
    for (size_t c = 0; c < block_cols; c++)
    {
      for (size_t dim = 0; dim < dimension; dim++)
      {
        auto target_ranks_left = target_ranks[dim];
        auto target_ranks_right = target_ranks[dim + 1];

        auto current_ranks_left = block_ttm_in({r, c}).get_ranks_left(dim);
        auto current_ranks_right = block_ttm_in({r, c}).get_ranks_right(dim);

        if ((target_ranks_left == current_ranks_left) and (target_ranks_right == current_ranks_right))
        {
          continue;
        }

        index_t left_multiple = target_ranks_left / current_ranks_left;
        index_t right_multiple = target_ranks_right / current_ranks_right;

        boba_always_assert_equal(current_ranks_left * left_multiple, target_ranks_left, "Error in rank normalization");
        boba_always_assert_equal(current_ranks_right * right_multiple, target_ranks_right, "Error in rank normalization");

        Tensor<4, space, data_t> rank_normalizer({left_multiple, 1, 1, right_multiple});
        rank_normalizer.fill_with(1.0 / static_cast<data_t>(rank_normalizer.size()));

        block_ttm_in_normalized({r, c}).cores[dim] = tensor_product(rank_normalizer, block_ttm_in({r, c}).cores[dim]);
      }
    }
  }

  //
  // Consider all combinations of blocks
  //
  Multiindexer<dimension> block_row_ids(filled_array<dimension>(block_rows));
  Multiindexer<dimension> block_col_ids(filled_array<dimension>(block_cols));
  Multiindexer<2> block_ids({block_row_ids.size(), block_col_ids.size()});

  for (size_t id = 0; id < block_ids.size(); id++)
  {
    auto [row_id, col_id] = block_ids.multiindex(id);
    auto block_row_mid = block_row_ids.multiindex(row_id);
    auto block_col_mid = block_row_ids.multiindex(col_id);

    auto block_rows_offets = filled_array<dimension>(0_z);
    auto block_cols_offets = filled_array<dimension>(0_z);

    for (size_t dim = 0; dim < dimension; dim++)
    {
      block_rows_offets[dim] = rows_offets.at(block_row_mid[dim])[dim];
      block_cols_offets[dim] = cols_offets.at(block_col_mid[dim])[dim];
    }

    TensorTrainMatrix<dimension, space, data_t> temp;
    for (size_t dim = 0; dim < dimension; dim++)
    {
      temp.cores[dim] = block_ttm_in_normalized({block_row_mid[dim], block_col_mid[dim]}).cores[dim];
    }

    unblocked.TensorTrainMatrix_add(temp, block_rows_offets, block_cols_offets);
  }

  return unblocked;
}

/**
 * @brief Convert a block TensorTrain into an equivalent tt
 *
 *
 * @param[in] block_tt BlockVector<tt>
 * @return TensorTrain with values filled to be algebraically equivalent to the block tt
 */

template <size_t dimension, execution_space space, typename data_t>
TensorTrain<dimension, space, data_t> unblock(const BlockVector<TensorTrain<dimension, space, data_t>>& block_tt_in)
{
  auto sizes = filled_array<dimension>(0_z);
  std::vector<Array<size_t, dimension>> sizes_offets;
  auto block_size = block_tt_in.block_size;
  sizes_offets.resize(block_size + 1);

  for (size_t r = 0; r < block_size; r++)
  {
    sizes_offets.at(r) = sizes;
    sizes += block_tt_in({r}).sizes();
  }
  sizes_offets.at(block_size) = sizes;

  TensorTrain<dimension, space, data_t> unblocked(sizes);
  unblocked.fill_with_zeros();

  //
  // Normalize the ranks
  //
  Array<index_t, dimension + 1> target_ranks = filled_array<dimension + 1>(1_z);
  for (size_t r = 0; r < block_size; r++)
  {
    target_ranks[0] = std::lcm(target_ranks[0], block_tt_in(r).get_ranks_left(0));
    for (size_t dim = 0; dim < dimension; dim++)
    {
      target_ranks[dim + 1] = std::lcm(target_ranks[dim + 1], block_tt_in(r).get_ranks_right(dim));
    }
  }

  auto block_tt_in_normalized = block_tt_in;

  for (size_t r = 0; r < block_size; r++)
  {
    for (size_t dim = 0; dim < dimension; dim++)
    {
      auto target_ranks_left = target_ranks[dim];
      auto target_ranks_right = target_ranks[dim + 1];

      auto current_ranks_left = block_tt_in(r).get_ranks_left(dim);
      auto current_ranks_right = block_tt_in(r).get_ranks_right(dim);

      if ((target_ranks_left == current_ranks_left) and (target_ranks_right == current_ranks_right))
      {
        continue;
      }

      index_t left_multiple = target_ranks_left / current_ranks_left;
      index_t right_multiple = target_ranks_right / current_ranks_right;
      boba_always_assert_equal(current_ranks_left * left_multiple, target_ranks_left, "Error in rank normalization");
      boba_always_assert_equal(current_ranks_right * right_multiple, target_ranks_right, "Error in rank normalization");

      Tensor<3, space, data_t> rank_normalizer({left_multiple, 1, right_multiple});
      rank_normalizer.fill_with(1.0 / static_cast<data_t>(rank_normalizer.size()));

      block_tt_in_normalized(r).cores[dim] = tensor_product(rank_normalizer, block_tt_in(r).cores[dim]);
    }
  }

  //
  // Consider all combinations of blocks
  //
  Multiindexer<dimension> block_ids(filled_array<dimension>(block_size));

  for (size_t id = 0; id < block_ids.size(); id++)
  {
    auto block_mid = block_ids.multiindex(id);
    auto block_sizes_offets = filled_array<dimension>(0_z);

    for (size_t dim = 0; dim < dimension; dim++)
    {
      block_sizes_offets[dim] = sizes_offets.at(block_mid[dim])[dim];
    }

    TensorTrain<dimension, space, data_t> temp;
    for (size_t dim = 0; dim < dimension; dim++)
    {
      temp.cores[dim] = block_tt_in_normalized(block_mid[dim]).cores[dim];
    }

    unblocked.TensorTrain_add(temp, block_sizes_offets);
  }

  return unblocked;
}

/**
 * @brief Convert a TensorTrainMatrix into an equivalent BlockVector<ttm>, using offsets collected from block_ttm
 * WARNING - this function should be used only if you really know what you are doing (see test_block_operator for example)
 *
 * @param[in,out] block_ttm BlockVector<ttm> with core rows and columns set so that they add up to the sizes in full_tt, as these will be used to determine offsets from which to copy data
 * @param[in] full_ttm TTM from which core data will be copied
 */

template <size_t dimension, execution_space space, typename data_t>
void copy_operator_into_block_form(
  BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  const TensorTrainMatrix<dimension, space, data_t>& full_ttm)
{
  auto rows = filled_array<dimension>(0_z);
  auto cols = rows;
  std::vector<Array<size_t, dimension>> rows_offets;
  std::vector<Array<size_t, dimension>> cols_offets;
  auto block_rows = block_ttm.block_rows;
  auto block_cols = block_ttm.block_cols;
  rows_offets.resize(block_rows + 1);
  cols_offets.resize(block_cols + 1);

  for (size_t r = 0; r < block_rows; r++)
  {
    rows_offets.at(r) = rows;
    rows += block_ttm({r, 0}).core_rows();
  }
  for (size_t c = 0; c < block_cols; c++)
  {
    cols_offets.at(c) = cols;
    cols += block_ttm({0, c}).core_cols();
  }
  rows_offets.at(block_rows) = rows;
  cols_offets.at(block_cols) = cols;

  boba_always_assert_equal(rows, full_ttm.core_rows(), "Block TTM is not consistent with full TTM");
  boba_always_assert_equal(cols, full_ttm.core_cols(), "Block TTM is not consistent with full TTM");

  for (size_t d_row = 0; d_row < block_rows; d_row++)
  {
    for (size_t d_col = 0; d_col < block_cols; d_col++)
    {
      auto core_rows = rows_offets.at(d_row + 1) - rows_offets.at(d_row);
      auto core_cols = cols_offets.at(d_col + 1) - cols_offets.at(d_col);

      block_ttm({d_row, d_col}).resize(core_rows, core_cols);

      for (size_t d_core = 0; d_core < dimension; d_core++)
      {
        auto ranks_left = full_ttm.get_ranks_left(d_core);
        auto ranks_right = full_ttm.get_ranks_right(d_core);

        block_ttm({d_row, d_col}).cores[d_core].resize({ranks_left, core_rows[d_core], core_cols[d_core], ranks_right});

        auto block_view = block_ttm({d_row, d_col}).cores[d_core].view();
        auto full_view = full_ttm.cores[d_core].const_view();
        auto core_row_offsets = rows_offets.at(d_row);
        auto core_col_offsets = cols_offets.at(d_col);

        ::boba::loop<space, 1>(block_view.size(),
                               [=] __boba_host_device__(size_t i)
        {
          auto [left_rank, row, col, right_rank] = block_view.multiindex(i);
          auto full_row = row + core_row_offsets[d_core];
          auto full_col = col + core_col_offsets[d_core];
          block_view(i) = full_view({left_rank, full_row, full_col, right_rank});
        });
      }
    }
  }

  for (size_t r = 0; r < block_rows; r++)
  {
    for (size_t c = 0; c < block_cols; c++)
    {
      for (size_t d_core = 0; d_core < dimension - 1; d_core++)
      {
        boba_always_assert_equal(block_ttm({r, c}).get_ranks_right(d_core), block_ttm({r, c}).get_ranks_left(d_core + 1), "Inconsistent cores");
      }
    }
  }
}

/**
 * @brief Convert a TensorTrain into an equivalent BlockVector<tt>, using offsets collected from block_tt
 * WARNING - this function should be used only if you really know what you are doing (see test_block_operator for example)
 *
 * @param[in,out] block_tt BlockVector<tt> with sizes set so that they add up to the sizes in full_tt, as these will be used to determine offsets from which to copy data
 * @param[in] full_tt TT from which core data will be copied
 */

template <size_t dimension, execution_space space, typename data_t>
void copy_vector_into_block_form(
  BlockVector<TensorTrain<dimension, space, data_t>>& block_tt,
  const TensorTrain<dimension, space, data_t>& full_tt)
{
  checkpoint();
  auto sizes = filled_array<dimension>(0_z);
  std::vector<Array<size_t, dimension>> sizes_offets;
  auto blocks = block_tt.block_size;
  sizes_offets.resize(blocks + 1);

  for (size_t r = 0; r < blocks; r++)
  {
    sizes_offets.at(r) = sizes;
    sizes += block_tt(r).sizes();
  }
  sizes_offets.at(blocks) = sizes;

  boba_always_assert_equal(sizes, full_tt.sizes(), "Block TT is not consistent with full TT");

  for (size_t d_row = 0; d_row < blocks; d_row++)
  {
    auto core_sizes = sizes_offets.at(d_row + 1) - sizes_offets.at(d_row);
    block_tt(d_row).resize(core_sizes);

    for (size_t d_core = 0; d_core < dimension; d_core++)
    {
      auto ranks_left = full_tt.get_ranks_left(d_core);
      auto ranks_right = full_tt.get_ranks_right(d_core);

      block_tt(d_row).cores[d_core].resize({ranks_left, core_sizes[d_core], ranks_right});

      auto block_view = block_tt({d_row}).cores[d_core].view();
      auto full_view = full_tt.cores[d_core].const_view();
      auto core_row_offsets = sizes_offets.at(d_row);
      ::boba::loop<space, 1>(block_view.size(),
                             [=] __boba_host_device__(size_t i)
      {
        auto [left_rank, row, right_rank] = block_view.multiindex(i);
        auto full_row = row + core_row_offsets[d_core];
        block_view(i) = full_view({left_rank, full_row, right_rank});
      });
    }
  }

  for (size_t r = 0; r < blocks; r++)
  {
    for (size_t d_core = 0; d_core < dimension - 1; d_core++)
    {
      boba_always_assert_equal(block_tt(r).get_ranks_right(d_core), block_tt(r).get_ranks_left(d_core + 1), "Inconsistent cores");
    }
  }
}

/**
 * @brief khatri_rao product of two block_operators
 *
 * @param[in] operator_A BlockOperator
 * @param[in] operator_B BlockOperator
 * @return The khatri_rao of operator_A and operator_B
 */

template <typename operator_t>
BlockOperator<operator_t> khatri_rao(
  const BlockOperator<operator_t>& operator_A,
  const BlockOperator<operator_t>& operator_B)
{
  BOBA_CALI_MARK

  auto block_rows = operator_A.block_rows;
  auto block_cols = operator_A.block_cols;

  BlockOperator<operator_t> output(block_rows, block_cols);

  //
  // Each block operator is the tensor product of the corresponding inputs
  //
  for (size_t row = 0; row < block_rows; row++)
  {
    for (size_t col = 0; col < block_cols; col++)
    {
      output({row, col}) = tensor_product(operator_A({row, col}), operator_B({row, col}));
    }
  }

  return output;
}

/**
 * @brief khatri_rao product of two block_vectors
 *
 * @param[in] vector_A block_vectors
 * @param[in] vector_B block_vectors
 * @return The khatri_rao of vector_A and vector_B
 */

template <typename vector_t>
BlockVector<vector_t> khatri_rao(
  const BlockVector<vector_t>& vector_A,
  const BlockVector<vector_t>& vector_B)
{
  BOBA_CALI_MARK
  auto block_size = vector_A.block_size;
  BlockVector<vector_t> output(block_size);

  //
  // Each block vector is the tensor product of the corresponding inputs
  //
  for (size_t row = 0; row < block_size; row++)
  {
    output(row) = tensor_product(vector_A(row), vector_B(row));
  }

  return output;
}

template <typename operator_t>
BlockOperator<operator_t> operator*(
  const BlockOperator<operator_t>& input,
  const PermutationMatrix<host_space, size_t>& permutation)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(input.block_cols, permutation.rows(), "Incompatible block sizes");
  BlockOperator<operator_t> output(input.block_rows, input.block_cols);
  auto permutation_view = permutation.view();

  for (size_t row = 0; row < input.block_rows; row++)
  {
    for (size_t col = 0; col < input.block_cols; col++)
    {
      auto permuted_col = permutation_view(col);
      output({row, col}) = input({row, permuted_col});
    }
  }

  return output;
}

template <typename operator_t>
BlockVector<operator_t> operator*(
  const BlockVector<operator_t>& input,
  const PermutationMatrix<host_space, size_t>& permutation)
{
  BOBA_CALI_MARK
  BlockVector<operator_t> output(input.block_size);
  auto permutation_view = permutation.view();

  for (size_t col = 0; col < output.block_size; col++)
  {
    auto permuted_col = permutation_view(col);
    output(col) = input(permuted_col);
  }

  return output;
}

template <typename operator_t>
BlockOperator<operator_t> operator*(
  const PermutationMatrix<host_space, size_t>& permutation,
  const BlockOperator<operator_t>& input)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(permutation.rows(), input.block_rows, "Incompatible block sizes");
  BlockOperator<operator_t> output(input.block_rows, input.block_cols);
  auto permutation_view = permutation.view();

  for (size_t row = 0; row < output.block_rows; row++)
  {
    for (size_t col = 0; col < output.block_cols; col++)
    {
      auto permuted_row = permutation_view(row);
      output({permuted_row, col}) = input({row, col});
    }
  }

  return output;
}

template <typename operator_t>
BlockVector<operator_t> operator*(
  const PermutationMatrix<host_space, size_t>& permutation,
  const BlockVector<operator_t>& input)
{
  BOBA_CALI_MARK
  BlockVector<operator_t> output(input.block_size);
  auto permutation_view = permutation.view();

  for (size_t col = 0; col < input.block_size; col++)
  {
    auto permuted_row = permutation_view(col);
    output(permuted_row) = input(col);
  }

  return output;
}

// -------------------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Write to file in a way consistent with Tensor::write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(const BlockVector<TensorTrain<dimension, space, data_t>>& block_tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  auto block_size = block_tt.block_size;
  ::boba::Tensor<1, host_space, size_t> temp({1});
  temp.fill_with(block_size);
  boba::write_to_file(temp, print_filename + "_block_size");

  for (size_t element = 0; element < block_size; element++)
  {
    auto row_name = print_filename + "_" + std::to_string(element);
    for (size_t d = 0; d < dimension; d++)
    {
      boba::write_to_file(block_tt(element).cores[d], row_name + "_core_" + std::to_string(d));
    }
  }
}

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(const BlockVector<Tensor<dimension, space, data_t>>& block_tensor, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_tensor.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  auto block_size = block_tensor.block_size;
  ::boba::Tensor<1, host_space, size_t> temp({1});
  temp.fill_with(block_size);
  boba::write_to_file(temp, print_filename + "_block_size");

  for (size_t element = 0; element < block_size; element++)
  {
    auto row_name = print_filename + "_" + std::to_string(element);
    boba::write_to_file(block_tensor(element), row_name);
  }
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(BlockVector<TensorTrain<dimension, space, data_t>>& block_tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  ::boba::Tensor<1, host_space, size_t> temp({1});
  boba::read_from_file(temp, print_filename + "_block_size");
  auto block_size = temp({0});

  block_tt.vector_blocks.resize(block_size);

  for (size_t element = 0; element < block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    read_from_file(block_tt({element}), element_str);
  }
}

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(BlockVector<Tensor<dimension, space, data_t>>& block_tensor, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_tensor.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  ::boba::Tensor<1, host_space, size_t> temp({1});
  boba::read_from_file(temp, print_filename + "_block_size");
  auto block_size = temp({0});

  block_tensor.vector_blocks.resize(block_size);

  for (size_t element = 0; element < block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    read_from_file(block_tensor({element}), element_str);
  }
}

/**
 * \brief
 * Write to file in a way consistent with Tensor::write_to_file
 */

template <typename vector_t>
void write_to_file(const BlockVector<vector_t>& block_vec, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_vec.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  for (size_t element = 0; element < block_vec.block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    write_to_file(block_vec({element}), element_str);
  }
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <typename vector_t>
void read_from_file(BlockVector<vector_t>& block_vec, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_vec.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  block_vec.vector_blocks.resize(block_vec.block_size);

  for (size_t element = 0; element < block_vec.block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    read_from_file(block_vec({element}), element_str);
  }
}

/**
 * \brief
 * Write the tt to a HDF5 file in a way consistent with tt::write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(const BlockVector<TensorTrain<dimension, space, data_t>>& block_tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_tt.name();
  }

  detail::HDF5File h5_file(filename, "w");

  auto blocks = block_tt.block_size;

  h5_file.write_int(object_name + "_block_size", blocks);

  for (size_t block = 0; block < blocks; block++)
  {
    std::string block_name = object_name + "_" + std::to_string(block);
    for (size_t d = 0; d < dimension; d++)
    {
      h5_file.write_array(block_name + "_core_" + std::to_string(d), block_tt(block).cores[d]);
    }
  }
}

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(const BlockVector<Tensor<dimension, space, data_t>>& block_tensor, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_tensor.name();
  }

  detail::HDF5File h5_file(filename, "w");

  auto blocks = block_tensor.block_size;

  h5_file.write_int(object_name + "_block_size", blocks);

  for (size_t block = 0; block < blocks; block++)
  {
    std::string block_name = object_name + "_" + std::to_string(block);
    h5_file.write_array(block_name, block_tensor(block));
  }
}

/**
 * \brief
 * Read from a file generated from write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(BlockVector<TensorTrain<dimension, space, data_t>>& block_tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_tt.name();
  }

  detail::HDF5File h5_file(filename, "r");

  auto blocks = static_cast<index_t>(h5_file.read_int(object_name + "_block_size"));

  block_tt.vector_blocks.resize(blocks);

  block_tt.block_size = blocks;
  for (size_t block = 0; block < blocks; block++)
  {
    std::string block_name = object_name + "_" + std::to_string(block);
    for (size_t d = 0; d < dimension; d++)
    {
      h5_file.read_array(block_name + "_core_" + std::to_string(d), block_tt(block).cores[d]);
    }
  }
}

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(BlockVector<Tensor<dimension, space, data_t>>& block_tensor, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_tensor.name();
  }

  detail::HDF5File h5_file(filename, "r");

  auto blocks = static_cast<index_t>(h5_file.read_int(object_name + "_block_size"));

  block_tensor.vector_blocks.resize(blocks);

  block_tensor.block_size = blocks;
  for (size_t block = 0; block < blocks; block++)
  {
    std::string block_name = object_name + "_" + std::to_string(block);
    h5_file.read_array(block_name, block_tensor(block));
  }
}

/**
 * \brief
 * Write to file in a way consistent with Tensor::write_to_file
 * for BlockOperator of TensorTrainMatrix.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(
  const BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename = "")
{
  std::string print_filename = filename.empty()
                                 ? block_ttm.name()
                                 : std::string(filename);

  const size_t rows = block_ttm.block_rows;
  const size_t cols = block_ttm.block_cols;

  boba::Tensor<1, host_space, size_t> temp_rows({1});
  boba::Tensor<1, host_space, size_t> temp_cols({1});
  temp_rows.fill_with(rows);
  temp_cols.fill_with(cols);
  boba::write_to_file(temp_rows, print_filename + "_block_rows");
  boba::write_to_file(temp_cols, print_filename + "_block_cols");

  for (size_t row = 0; row < rows; ++row)
  {
    const std::string row_str = print_filename + "_" + std::to_string(row);
    for (size_t col = 0; col < cols; ++col)
    {
      boba::write_to_file(block_ttm({row, col}), row_str + "_" + std::to_string(col));
    }
  }
}

/**
 * \brief
 * Read from files generated from write_to_file
 * for BlockOperator of TensorTrainMatrix.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(
  BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename = "")
{
  std::string print_filename = filename.empty()
                                 ? block_ttm.name()
                                 : std::string(filename);

  boba::Tensor<1, host_space, size_t> temp_rows({1});
  boba::Tensor<1, host_space, size_t> temp_cols({1});
  boba::read_from_file(temp_rows, print_filename + "_block_rows");
  boba::read_from_file(temp_cols, print_filename + "_block_cols");
  const size_t rows = temp_rows({0});
  const size_t cols = temp_cols({0});

  block_ttm.block_rows = rows;
  block_ttm.block_cols = cols;
  block_ttm.operator_blocks.resize(rows * cols);

  for (size_t row = 0; row < rows; ++row)
  {
    const std::string row_str = print_filename + "_" + std::to_string(row);
    for (size_t col = 0; col < cols; ++col)
    {
      boba::read_from_file(block_ttm({row, col}), row_str + "_" + std::to_string(col));
    }
  }
}

/**
 * \brief
 * Write  to a MATLAB mat-file in a way consistent with Tensor::write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_mat_file(const BlockVector<TensorTrain<dimension, space, data_t>>& block_vec, std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_vec.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "w");

  auto block_size = block_vec.block_size;
  ::boba::Tensor<1, host_space, size_t> temp({1});
  temp.fill_with(block_size);
  matlab_file.write_array(print_filename + "_block_size", temp);

  for (size_t element = 0; element < block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    for (size_t d = 0; d < dimension; d++)
    {
      auto core_str = element_str + "_core_" + std::to_string(d);
      matlab_file.write_array(core_str, block_vec({element}).cores[d]);
    }
  }
}

/**
 * \brief
 * Read from a file generated from write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_mat_file(BlockVector<TensorTrain<dimension, space, data_t>>& block_tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = block_tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "r");
  ::boba::Tensor<2, host_space, size_t> temp({1, 1}); // In matlab, scalars are 1x1
  matlab_file.read_array(print_filename + "_block_size", temp);
  auto block_size = temp({0, 0});

  block_tt.block_size = block_size;
  block_tt.vector_blocks.resize(block_size);

  for (size_t element = 0; element < block_tt.block_size; element++)
  {
    auto element_str = print_filename + "_" + std::to_string(element);
    for (size_t d = 0; d < dimension; d++)
    {
      auto core_str = element_str + "_core_" + std::to_string(d);
      matlab_file.read_array(core_str, block_tt({element}).cores[d]);
    }
  }
}

/**
 * \brief
 * Write to a MATLAB mat-file in a way consistent with Tensor::write_to_mat_file
 * for BlockOperator of TensorTrainMatrix.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_mat_file(
  const BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename = "")
{
  std::string print_filename = filename.empty()
                                 ? block_ttm.name()
                                 : std::string(filename);

  const size_t rows = block_ttm.block_rows;
  const size_t cols = block_ttm.block_cols;

  detail::MatFile matlab_file(print_filename, "w");

  ::boba::Tensor<1, host_space, size_t> temp({1});
  temp.fill_with(rows);
  matlab_file.write_array(print_filename + "_block_rows", temp);
  temp.fill_with(cols);
  matlab_file.write_array(print_filename + "_block_cols", temp);

  for (size_t row = 0; row < rows; ++row)
  {
    const std::string row_str = print_filename + "_" + std::to_string(row);
    for (size_t col = 0; col < cols; ++col)
    {
      const auto col_str = row_str + "_" + std::to_string(col);
      for (size_t core = 0; core < dimension; core++)
      {
        auto core_str = col_str + "_core_" + std::to_string(core);
        matlab_file.write_array(core_str, block_ttm({row, col}).cores[core]);
      }
    }
  }
}

/**
 * \brief
 * Read from a file generated from write_to_mat_file
 * for BlockOperator of TensorTrainMatrix.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_mat_file(
  BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename = "")
{
  std::string print_filename = filename.empty()
                                 ? block_ttm.name()
                                 : std::string(filename);

  detail::MatFile matlab_file(print_filename, "r");

  ::boba::Tensor<2, host_space, size_t> temp({1, 1}); // In matlab, scalars are 1x1
  matlab_file.read_array(print_filename + "_block_rows", temp);
  auto block_rows = temp({0, 0});
  matlab_file.read_array(print_filename + "_block_cols", temp);
  auto block_cols = temp({0, 0});

  block_ttm.block_rows = block_rows;
  block_ttm.block_cols = block_cols;
  block_ttm.operator_blocks.resize(block_rows * block_cols);

  for (size_t row = 0; row < block_rows; ++row)
  {
    const std::string row_str = print_filename + "_" + std::to_string(row);
    for (size_t col = 0; col < block_cols; ++col)
    {
      const auto col_str = row_str + "_" + std::to_string(col);
      for (size_t core = 0; core < dimension; core++)
      {
        auto core_str = col_str + "_core_" + std::to_string(core);
        matlab_file.read_array(core_str, block_ttm({row, col}).cores[core]);
      }
    }
  }
}

/**
 * \brief
 * Write the block operator (TT-matrix blocks) to a HDF5 file.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(
  const BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename,
  std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_ttm.name();
  }

  detail::HDF5File h5_file(filename, "w");

  const size_t rows = block_ttm.block_rows;
  const size_t cols = block_ttm.block_cols;

  h5_file.write_int(object_name + "_block_rows", static_cast<long long>(rows));
  h5_file.write_int(object_name + "_block_cols", static_cast<long long>(cols));

  for (size_t row = 0; row < rows; ++row)
  {
    const std::string block_row_name = object_name + "_" + std::to_string(row);
    for (size_t col = 0; col < cols; ++col)
    {
      const std::string block_name = block_row_name + "_" + std::to_string(col);
      for (size_t d = 0; d < dimension; ++d)
      {
        h5_file.write_array(block_name + "_core_" + std::to_string(d), block_ttm({row, col}).cores[d]);
      }
    }
  }
}

/**
 * \brief
 * Read the block operator (TT-matrix blocks) from a HDF5 file generated above.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(
  BlockOperator<TensorTrainMatrix<dimension, space, data_t>>& block_ttm,
  std::string_view filename,
  std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = block_ttm.name();
  }

  detail::HDF5File h5_file(filename, "r");

  const size_t rows = static_cast<size_t>(h5_file.read_int(object_name + "_block_rows"));
  const size_t cols = static_cast<size_t>(h5_file.read_int(object_name + "_block_cols"));

  block_ttm.block_rows = rows;
  block_ttm.block_cols = cols;
  block_ttm.operator_blocks.resize(rows * cols);

  for (size_t row = 0; row < rows; ++row)
  {
    const std::string block_row_name = object_name + "_" + std::to_string(row);
    for (size_t col = 0; col < cols; ++col)
    {
      const std::string block_name = block_row_name + "_" + std::to_string(col);

      for (size_t d = 0; d < dimension; ++d)
      {
        h5_file.read_array(block_name + "_core_" + std::to_string(d), block_ttm({row, col}).cores[d]);
      }
    }
  }
}
} // namespace boba
