// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Quantized tensor train matrix.
 */

template <::boba::execution_space space, typename _data_t>
struct QuantizedTensorTrainMatrix
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "QuantizedTensorTrainMatrix";

  size_t rows_base = 2;
  size_t cols_base = 2;
  size_t exponent = 1;

  real_data_t svd_tolerance_relative = 1.0e-12;
  real_data_t svd_tolerance_absolute = 1.0e-12;
  size_t svd_max_kept_values = highest_value<size_t>();

  std::vector<Tensor<4, space, data_t>> cores;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize. Instantiated object is the zero tensor train matrix.
   */

  QuantizedTensorTrainMatrix(
    const size_t _rows_base,
    const size_t _cols_base,
    const size_t _exponent)
  {
    this->resize(_rows_base, _cols_base, _exponent);
  }

  // Default constructor
  QuantizedTensorTrainMatrix() = default;

  // Copy constructor
  QuantizedTensorTrainMatrix(QuantizedTensorTrainMatrix const&) = default;

  // Move constructor
  QuantizedTensorTrainMatrix(QuantizedTensorTrainMatrix&&) = default;

  // Copy assignment
  QuantizedTensorTrainMatrix& operator=(QuantizedTensorTrainMatrix const&) = default;

  // Move assignment
  QuantizedTensorTrainMatrix& operator=(QuantizedTensorTrainMatrix&&) = default;

  /**
   * \brief
   * Host/device copy constructor
   */

  template <execution_space rhs_space>
  QuantizedTensorTrainMatrix(QuantizedTensorTrainMatrix<rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        rows_base(rhs.rows_base),
        cols_base(rhs.cols_base),
        exponent(rhs.exponent),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        svd_max_kept_values(rhs.svd_max_kept_values),
        cores(rhs.cores)
  {
  }

  /**
   * \brief
   * Host/device copy assignment
   */

  template <execution_space rhs_space>
  QuantizedTensorTrainMatrix& operator=(QuantizedTensorTrain<rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    rows_base = rhs.rows_base;
    cols_base = rhs.cols_base;
    exponent = rhs.exponent;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    svd_max_kept_values = rhs.svd_max_kept_values;
    cores = rhs.cores;
  }

  ~QuantizedTensorTrainMatrix() = default;

  /**
   * \brief
   * Resizes the quantized tensor train matrix logical shape and reinitializes every core to zero.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   */

  void resize(
    const size_t _rows_base,
    const size_t _cols_base,
    const size_t _exponent)
  {
    boba_always_assert_positive(rows_base, "rows_base must be positive");
    boba_always_assert_positive(cols_base, "cols_base must be positive");
    boba_always_assert_positive(exponent, "exponent must be positive");

    rows_base = _rows_base;
    cols_base = _cols_base;
    exponent = _exponent;

    cores.resize(exponent);
    for (auto& core : cores)
    {
      core.resize({1_z, rows_base, cols_base, 1_z});
      core.fill_with_zeros();
    }
  }

  /**
   * \brief
   * Renames this quantized tensor train matrix. Each core will have the
   * new name format: new_name + "_core_" + core_number.
   */
  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < exponent; d++)
    {
      this->cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Printing
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Prints out the information of the this quantized tensor train matrix.
   */
  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    for (size_t d = 0; d < exponent; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      std::cout << write_indent(indent + 2) << this->m_name << std::endl
                << write_indent(indent + 3) << "ranks_left = " << this->cores[d].sizes(0)
                << std::endl
                << write_indent(indent + 3) << "ranks_right = " << this->cores[d].sizes(3)
                << std::endl
                << write_indent(indent + 3) << "size = " << this->cores[d].sizes(1) << " x " << this->cores[d].sizes(2) << std::endl;

      this->cores[d].print(indent + 1);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Decompress
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Decompresses this quantized tensor train matrix back into a dense Matrix.
   */
  Matrix<space, data_t> decompress() const
  {
    // Choose an upper bound for 2*exponent, so that we can recast this qttm as a ttm
    // then use TensorTrainMatrix::decompress
    constexpr size_t dimension_upper_bound = 12;
    boba_always_assert_le(2 * exponent, dimension_upper_bound, "dimension_upper_bound too small, please adjust for your needs - contact a developer if this is a longer term issue");

    // Pad ficticious dimensions with 1s for the extend dimension, which does not increase the size
    ::boba::Array<size_t, dimension_upper_bound> rows = filled_array<dimension_upper_bound>(1_z);
    ::boba::Array<size_t, dimension_upper_bound> cols = filled_array<dimension_upper_bound>(1_z);
    for (size_t d = 0; d < exponent; d++)
    {
      rows[d] = rows_base;
      cols[d] = cols_base;
    }
    ::boba::TensorTrainMatrix<dimension_upper_bound, space, data_t> ttm(rows, cols);
    for (size_t d = 0; d < dimension_upper_bound; d++)
    {
      if (d < exponent)
      {
        ttm.cores[d] = cores[d];
      }
      else
      {
        // Set each core to a scalar value of 1
        ttm.cores[d].resize(filled_array<4>(1_z));
        ttm.cores[d].fill_with(1);
      }
    }

    return ttm.decompress();
  }

  // -------------------------------------------------------------------------------------
  // Section: Getters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Returns number of cores
   */

  constexpr size_t get_number_cores()
  {
    return exponent;
  }

  /**
   * \brief
   * Returns number of matrix elements per core.
   */
  size_t constexpr get_number_matrix_elements(size_t d) const
  {
    detail::ignore(d);
    return rows_base * cols_base;
  }

  /**
   * \brief
   * Returns the number of elements in the physical bonds of
   * this quantized tensor train matrix.
   */
  size_t get_number_elements() const
  {
    size_t sum_elements = 0;
    for (size_t d = 0; d < exponent; d++)
    {
      sum_elements += this->cores[d].get_number_elements();
    }
    return sum_elements;
  }

  /**
   * \brief
   * Returns the size of the rows of the i'th core.
   */
  size_t core_rows(size_t i) const
  {
    detail::ignore(i);
    return rows_base;
  }

  /**
   * \brief
   * Returns the size of the columns of the i'th core
   */
  size_t core_cols(size_t i) const
  {
    detail::ignore(i);
    return cols_base;
  }

  /**
   * \brief
   * Returns the number of matrix elements represented from the rows
   * of this quantized tensor train matrix.
   */
  size_t rows() const
  {
    return pow(rows_base, exponent);
  }

  /**
   * \brief
   * Returns the number of matrix elements represented from the columns
   * of this quantized tensor train matrix.
   */
  size_t cols() const
  {
    return pow(cols_base, exponent);
  }

  /**
   * \brief
   * Returns the full number of matrix elements this quantized
   * tensor train matrix represents.
   */
  size_t get_full_size() const
  {
    return rows() * cols();
  }

  /**
   * \brief Returns the ratio of the full size over the compressed size, truncated to two digits
   */

  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  // ranks and max ranks

  /**
   * \brief
   * Returns the left rank of the specified core of this
   * quantized tensor train matrix.
   */
  size_t get_ranks_left(size_t i) const
  {
    boba_always_assert(i < exponent, "invalid core number");
    return this->cores[i].sizes(0);
  }

  /**
   * \brief
   * Returns the right rank of the specified core of this
   * quantized tensor train matrix.
   */
  size_t get_ranks_right(size_t i) const
  {
    boba_always_assert(i < exponent, "invalid core number");
    return this->cores[i].sizes(3);
  }

  /**
   * \brief
   * Returns the number of nonzeros in this quantized tensor train matrix.
   */
  index_t number_nonzeros(const data_t tolerance = 1.0e-15) const
  {
    index_t sum = 0;
    for (size_t d = 0; d < exponent; d++)
    {
      sum += this->cores[d].number_nonzeros(tolerance);
    }
    return sum;
  }

  /**
   * \brief
   * Computes the sparsity ratio consistent with matrix sparsity
   */
  data_t sparsity(const data_t tolerance = 1.0e-15) const
  {
    data_t full_size = this->get_full_size();
    data_t nnz = this->number_nonzeros(tolerance);
    return full_size / nnz;
  }

  // -------------------------------------------------------------------------------------
  // Section: Debugging
  // -------------------------------------------------------------------------------------

  // -------------------------------------------------------------------------------------
  // Section: Write
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Sets this into a rank-one train filled with x, equivalent to a tensor filled with x
   */

  void fill_with(data_t x)
  {
    BOBA_CALI_MARK
    set_to_rank_one_scalar_core(this->cores[0], x);
    for (size_t d = 1; d < exponent; d++)
    {
      set_to_rank_one_scalar_core(this->cores[d], 1.0);
    }
  }

  /**
   * \brief
   * Sets this into a rank-one train filled with zeros
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK
    for (size_t d = 0; d < exponent; d++)
    {
      set_to_rank_one_scalar_core(this->cores[d], 0.0);
    }
  }

  /**
   * \brief
   * Sets this into a rank-one train of identity matrices
   */

  void set_to_identity_train()
  {
    BOBA_CALI_MARK
    for (size_t d = 0; d < exponent; d++)
    {
      set_to_identity_core(this->cores[d]);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Adds an input quantized tensor train matrix to this quantized tensor train matrix.
   */
  void QuantizedTensorTrainMatrix_add(QuantizedTensorTrainMatrix<space, data_t> const& input)
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < exponent; d++)
    {
      boba_assert_equal(input.core_rows(d), core_rows(d), "incompatible rows");
      boba_assert_equal(input.core_cols(d), core_cols(d), "incompatible cols");
    }
    checkpoint();
    this->add_train(input);
    checkpoint();
  }

  /**
   * \brief
   * Tensor train addition of this and subtrain
   */

  void add_train(QuantizedTensorTrainMatrix const& subtrain)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {
      boba_assert_ge(core_rows(d), subtrain.core_rows(d), "Subtrain size exceeds parent");
      boba_assert_ge(core_cols(d), subtrain.core_cols(d), "Subtrain size exceeds parent");
    }

    for (size_t d = 0; d < exponent; d++)
    {
      add_subcore(this->cores[d], subtrain.cores[d], d, exponent, 0, 0);
    }

    checkpoint();
  }

  /**
   * \brief
   * this += rhs
   */

  QuantizedTensorTrainMatrix& operator+=(QuantizedTensorTrainMatrix const& rhs)
  {
    BOBA_CALI_MARK
    this->QuantizedTensorTrainMatrix_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  QuantizedTensorTrainMatrix operator+(QuantizedTensorTrainMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this -= rhs
   */

  QuantizedTensorTrainMatrix& operator-=(QuantizedTensorTrainMatrix const& rhs)
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix rhs_minus{rhs};
    rhs_minus *= -1.0;
    this->QuantizedTensorTrainMatrix_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * negation operator
   */

  QuantizedTensorTrainMatrix operator-()
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief
   * output = this - rhs
   */

  QuantizedTensorTrainMatrix operator-(QuantizedTensorTrainMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * output = this * input
   */

  QuantizedTensorTrainMatrix operator*(QuantizedTensorTrainMatrix const& input) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix output{*this};
    output *= input;
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  QuantizedTensorTrainMatrix& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    this->multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  QuantizedTensorTrainMatrix operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrainMatrix output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Compress a matrix into this quantized tensor train matrix.
   * Note, the number of elements of the matrix must be consistent
   * with (row_base*col_base)^exponent from the defined quantized
   * tensor train matrix.
   */
  void compress(const boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();

    boba_always_assert_equal(input.sizes(0), rows(), "input has incompatible number of rows");
    boba_always_assert_equal(input.sizes(1), cols(), "input has incompatible number of cols");

    const size_t base_m = rows_base * cols_base;
    boba_always_assert_equal(input.size(), boba::pow(base_m, exponent), "QTT-matrix not correctly set up to compress this matrix");

    // Special case for rank 1
    if (exponent == 1_z)
    {
      cores.resize(1_z);
      cores[0].resize({1_z, rows_base, cols_base, 1_z});
      cores[0].reshape(input);
      return;
    }

    // Padding dimensions
    constexpr size_t dimension_upper_bound = 12;
    boba_always_assert_le(exponent, dimension_upper_bound, "dimension_upper_bound too small");

    ::boba::Array<size_t, dimension_upper_bound> rowsA = filled_array<dimension_upper_bound>(1_z);
    ::boba::Array<size_t, dimension_upper_bound> colsA = filled_array<dimension_upper_bound>(1_z);
    for (size_t d = 0; d < exponent; ++d)
    {
      rowsA[d] = rows_base;
      colsA[d] = cols_base;
    }
    // Construct the big ttm and tols
    ::boba::TensorTrainMatrix<dimension_upper_bound, space, data_t> ttm(rowsA, colsA);
    ttm.svd_tolerance_relative = svd_tolerance_relative;
    ttm.svd_tolerance_absolute = svd_tolerance_absolute;

    // Compress via ttm
    ttm.compress(input);

    // Copy to the QTT cores
    cores.resize(exponent);
    for (size_t d = 0; d < exponent; ++d)
    {
      cores[d] = ttm.cores[d];
    }
    checkpoint();
  }

  /**
   * \brief
   * Left-orthogonalize this quatized tensor train matrix by a left-to-right QR sweep.
   *
   * After this operation, all cores except the last are left-orthogonal.
   * The triangular factor from each QR factorization is absorbed into the left rank
   * index of the next core, so the represented tensor train matrix is unchanged
   * up to floating-point roundoff.
   */
  void orthogonalize()
  {
    BOBA_CALI_MARK
    checkpoint();

    if (get_number_cores() == 1_z)
    {
      bool left_rank_one = this->get_ranks_left(0_z) == 1_z;
      bool right_rank_one = this->get_ranks_right(0_z) == 1_z;

      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core =
        tensor_reduction_double_index(
          {"l", "row", "col", "r"},
          this->cores[0_z],
          {"row", "col"});

      this->cores[0_z] =
        reshape<4>(new_core, {1_z, core_rows(0_z), core_cols(0_z), 1_z});

      return;
    }

    boba::QR<space, data_t> qr;

    for (size_t d = 0_z; d + 1_z < get_number_cores(); d++)
    {
      checkpoint();

      {
        auto unfold_left = compute_unfold_left(this->cores[d]);
        qr(unfold_left);
      }

      if (qr.ranks == 0_z)
      {
        this->fill_with_zeros();
        return;
      }

      checkpoint();
      this->cores[d] =
        write_to_core_from_left_fold(qr.Q, core_rows(d), core_cols(d));

      checkpoint();
      this->cores[d + 1_z] =
        boba::tensor_contraction_single_index(
          {"k", "l"}, qr.R, {"l", "row", "col", "r"}, this->cores[d + 1_z], {"k", "row", "col", "r"});
    }

    checkpoint();
  }

  /**
   * \brief
   * Round this quatized tensor train matrix by left-orthogonalization followed by a
   * right-to-left truncated SVD sweep.
   *
   * The method first left-orthogonalizes the quantized tensor train matrix using a
   * QR sweep. It then performs a right-to-left SVD sweep, truncating each
   * interface rank according to svd_tolerance_relative, svd_tolerance_absolute, and
   * max_ranks. The resulting quantized tensor train matrix is right-orthogonal, with
   * the first core as the orthogonality center.
   */
  void round()
  {
    BOBA_CALI_MARK
    checkpoint();

    if (get_number_cores() == 1_z)
    {
      bool left_rank_one = this->get_ranks_left(0_z) == 1_z;
      bool right_rank_one = this->get_ranks_right(0_z) == 1_z;

      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core =
        tensor_reduction_double_index(
          {"l", "row", "col", "r"},
          this->cores[0_z],
          {"row", "col"});

      this->cores[0_z] =
        reshape<4>(new_core, {1_z, core_rows(0_z), core_cols(0_z), 1_z});

      return;
    }

    this->orthogonalize();

    boba::SVD<space, data_t> svd;
    svd.tolerance_relative = svd_tolerance_relative;
    svd.tolerance_absolute = svd_tolerance_absolute;

    for (size_t d = get_number_cores() - 1_z; d > 0_z; d--)
    {
      checkpoint();

      // This SVD truncates the interface rank r_d.
      svd.max_kept_singular_values = svd_max_kept_values;

      {
        auto unfold_right = compute_unfold_right(this->cores[d]);
        svd(unfold_right);
      }

      if (svd.significant_singular_values == 0_z)
      {
        this->fill_with_zeros();
        return;
      }

      {
        auto unfold_right = svd.V.transpose();

        this->cores[d] =
          write_to_core_from_right_fold(
            unfold_right,
            core_rows(d),
            core_cols(d));
      }

      checkpoint();
      apply_as_diagonal_right_in_place(svd.S, svd.U);

      checkpoint();
      this->cores[d - 1_z] =
        boba::tensor_contraction_single_index(
          {"l", "row", "col", "k"}, this->cores[d - 1_z], {"k", "r"}, svd.U, {"l", "row", "col", "r"});
    }

    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Multiply scalar
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Multiplies the tensor train by x, such that A = x*A
   */

  void multiply_scalar(data_t x)
  {
    checkpoint();
    this->cores[0] *= x;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matrix-vector Multiply
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * See QuantizedTensorTrainMatrix_vector_multiply.
   */

  QuantizedTensorTrain<space, data_t> apply(
    QuantizedTensorTrain<space, data_t> const& input) const
  {
    return this->QuantizedTensorTrainMatrix_vector_multiply_vanilla(input);
  }

  /**
   * \brief
   * Takes an input tensor train and performs a matrix-vector multiply in the sense of tensor trains.
   * At runtime, dispatches the basic or inline-rounding version of the underlying routine.
   */

  QuantizedTensorTrain<space, data_t> QuantizedTensorTrainMatrix_vector_multiply(
    QuantizedTensorTrain<space, data_t> const& input) const
  {
    return QuantizedTensorTrainMatrix_vector_multiply_vanilla(input);
  }

  /**
   * \brief
   * QuantizedTensorTrainMatrix_vector_multiply
   */

  QuantizedTensorTrain<space, data_t> QuantizedTensorTrainMatrix_vector_multiply_vanilla(
    QuantizedTensorTrain<space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    QuantizedTensorTrain<space, data_t> output(rows_base, exponent);
    output.svd_tolerance_relative = input.svd_tolerance_relative;
    output.svd_tolerance_absolute = input.svd_tolerance_absolute;

    for (size_t d = 0; d < exponent; d++)
    {
      boba_always_assert_equal(this->core_cols(d), input.sizes(d), "invalid size and cols for tensor-train matvec");

      size_t new_rows = this->core_rows(d);

      Array<size_t, 2> new_rank_left{input.get_ranks_left(d), get_ranks_left(d)};
      Array<size_t, 2> new_rank_right{input.get_ranks_right(d), get_ranks_right(d)};
      auto ranks_left_view = ::boba::Multiindexer<2>(new_rank_left);
      auto ranks_right_view = ::boba::Multiindexer<2>(new_rank_right);
      size_t new_ranks_left = ranks_left_view.size();
      size_t new_ranks_right = ranks_right_view.size();

      if ((new_ranks_left == 0) || (new_ranks_right == 0))
      {
        output.fill_with_zeros();
        return output;
      }

      auto temporary = tensor_contraction_single_index(
        {"ml", "row", "col", "mr"}, this->cores[d], {"l", "col", "r"}, input.cores[d], {"ml", "l", "row", "mr", "r"});

      output.cores[d] = reshape<3>(temporary, {new_ranks_left, new_rows, new_ranks_right});
    }
    checkpoint();
    return output;
  }
};

template <::boba::execution_space space, typename _data_t>
QuantizedTensorTrainMatrix<space, _data_t> operator*(_data_t scalar, QuantizedTensorTrainMatrix<space, _data_t> const& input)
{
  BOBA_CALI_MARK
  QuantizedTensorTrainMatrix<space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

template <::boba::execution_space space, typename _data_t>
QuantizedTensorTrain<space, _data_t> operator*(QuantizedTensorTrainMatrix<space, _data_t> const& matrix, QuantizedTensorTrain<space, _data_t> const& vector)
{
  BOBA_CALI_MARK
  return matrix.apply(vector);
}

} // namespace boba
