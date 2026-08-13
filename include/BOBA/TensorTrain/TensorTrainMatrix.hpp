// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tensor train matrix.
\verbatim
 Example for dimension = 3

       (1, r_0)                  (r_0, r_1)              (r_1, 1)
 ___  _________             ___  __________              _________
 \    |       |             \    |         |             |       |
  >   | 1,r_0 | rows[0] (x)  >   | r_0,r_1 | rows[1] (x) | r_1,1 | rows[2]
 /__  |_______|             /__  |_________|             |_______|
 r_0   cols[0]              r_1    cols[1]                cols[2]
\endverbatim
  */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
struct TensorTrainMatrix
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;
  using bandwidth_t = boba::TensorTrain<dimension, space, index_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "TensorTrainMatrix";

  Array<size_t, dimension + 1> max_ranks = filled_array<dimension + 1>(highest_value<size_t>());

  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);

  // per rank-multiindex per-core matrix bandwidth
  bandwidth_t bandwidths;

  Array<Tensor<4, space, data_t>, dimension> cores;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize. Instantiated object is the zero tensor train matrix.
   */

  TensorTrainMatrix(
    const Array<size_t, dimension> input_rows,
    const Array<size_t, dimension> input_cols)
  {
    this->resize(input_rows, input_cols);
  }

  /**
   * \brief
   * Resizes the tensor train matrix and reinitializes every core to zero.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   */

  void resize(
    const Array<index_t, dimension> input_rows,
    const Array<index_t, dimension> input_cols)
  {
    for (size_t d = 0; d < dimension; d++)
    {
      cores[d].resize({1, input_rows[d], input_cols[d], 1});
      cores[d].fill_with_zeros();
    }
    bandwidths.resize(filled_array<dimension>(index_t(0)));
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
  }

  // Default constructor
  TensorTrainMatrix() = default;

  // Copy constructor
  TensorTrainMatrix(TensorTrainMatrix const& rhs) = default;

  // Move constructor
  TensorTrainMatrix(TensorTrainMatrix&& rhs) = default;

  // Copy assignment
  TensorTrainMatrix& operator=(TensorTrainMatrix const& rhs) = default;

  // Move assignment
  TensorTrainMatrix& operator=(TensorTrainMatrix&& rhs) noexcept = default;

  // -------------------------------------------------------------------------------------
  // Section: Printing
  // -------------------------------------------------------------------------------------

  // prints the ranks:
  // (1, r_0) (r_0, r_1) (r_1, 1)

  /**
   * \brief
   * A string describing the tensor train ranks
   */

  Array<size_t, dimension + 1> ranks() const
  {
    Array<size_t, dimension + 1> rarr;
    for (size_t d = 0; d < get_number_cores(); d++)
    {
      rarr[d] = get_ranks_left(d);
    }
    rarr[dimension] = get_ranks_right(dimension - 1);
    return rarr;
  }

  std::string ranks_string(
    size_t /*indent*/ = 1) const
  {
    return (" ( " + make_delimited_string(ranks()) + " ) ");
  }

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    for (size_t d = 0; d < dimension; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      std::cout << write_indent(indent + 2) << this->m_name << std::endl
                << write_indent(indent + 3) << "ranks_left = " << this->get_ranks_left(d)
                << std::endl
                << write_indent(indent + 3) << "ranks_right = " << this->get_ranks_right(d)
                << std::endl
                << write_indent(indent + 3) << "size = " << this->core_rows(d) << " x " << this->core_cols(d) << std::endl;

      this->cores[d].print(indent + 1);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Decompress
  // -------------------------------------------------------------------------------------

  Matrix<space, data_t> decompress() const
  {
    checkpoint();
    ::boba::Array<size_t, dimension> sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      sizes[d] = core_rows(d) * core_cols(d);
    }
    checkpoint();
    ::boba::TensorTrain<dimension, space, data_t> tt_of_matrices(sizes);
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      auto left_ranks = this->get_ranks_left(d);
      auto right_ranks = this->get_ranks_right(d);
      tt_of_matrices.cores[d].resize({left_ranks, core_rows()[d] * core_cols()[d], right_ranks});
      tt_of_matrices.cores[d].reshape(this->cores[d]);
    }
    checkpoint();
    auto tt_of_matrices_decompressed = tt_of_matrices.decompress();
    checkpoint();

    ::boba::Array<size_t, 2 * dimension> rows_and_cols;
    for (size_t d = 0; d < dimension; d++)
    {
      rows_and_cols[2 * d + 0] = core_rows()[d];
      rows_and_cols[2 * d + 1] = core_cols()[d];
    }
    checkpoint();
    ::boba::Tensor<2 * dimension, space, data_t> matrix_row_col_tensor(rows_and_cols);
    matrix_row_col_tensor.reshape(tt_of_matrices_decompressed);
    checkpoint();
    ::boba::Array<size_t, 2 * dimension> permutations;
    for (size_t d = 0; d < dimension; d++)
    {
      permutations[d] = 2 * d + 0;
      permutations[dimension + d] = 2 * d + 1;
    }
    checkpoint();
    permute(matrix_row_col_tensor, permutations);
    checkpoint();
    boba::Matrix<space, data_t> full_matrix({rows(), cols()});
    full_matrix.reshape(matrix_row_col_tensor);
    checkpoint();
    return full_matrix;
  }

  // -------------------------------------------------------------------------------------
  // Section: Getters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Returns name of ttm
   */

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Returns number of cores
   */

  static constexpr size_t get_number_cores()
  {
    return get_dimension();
  }

  /**
   * \brief Returns number of cores
   */

  static constexpr size_t get_dimension()
  {
    return dimension;
  }

  size_t constexpr get_number_matrix_elements(size_t d) const
  {
    return core_rows(d) * core_cols(d);
  }

  size_t get_number_elements() const
  {
    size_t sum_elements = 0;
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += this->cores[d].get_number_elements();
    }
    return sum_elements;
  }

  size_t core_rows(size_t i) const
  {
    boba_always_assert_nonnegative(i, "invalid core number");
    boba_always_assert_lt(i, dimension, "invalid core number");
    return this->cores[i].sizes(1);
  }

  size_t core_cols(size_t i) const
  {
    boba_always_assert_nonnegative(i, "invalid core number");
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(2);
  }

  Array<size_t, dimension> core_rows() const
  {
    Array<size_t, dimension> rows = filled_array<dimension>(0_z);
    for (size_t d = 0; d < dimension; d++)
    {
      rows[d] = core_rows(d);
    }
    return rows;
  }

  Array<size_t, dimension> core_cols() const
  {
    Array<size_t, dimension> cols = filled_array<dimension>(0_z);
    for (size_t d = 0; d < dimension; d++)
    {
      cols[d] = core_cols(d);
    }
    return cols;
  }

  size_t rows() const
  {
    return product(core_rows());
  }

  size_t cols() const
  {
    return product(core_cols());
  }

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

  size_t get_ranks_left(size_t i) const
  {
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(0);
  }

  size_t get_ranks_right(size_t i) const
  {
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(3);
  }

  Array<size_t, dimension> sizes() const noexcept
  {
    ::boba::Array<size_t, dimension> this_sizes{0};
    for (size_t d = 0; d < dimension; d++)
    {
      this_sizes[d] = sizes(d);
    }
    return this_sizes;
  }

  size_t sizes(size_t d) const noexcept
  {
    return core_rows()[d] * core_cols()[d];
  }

  /**
   * Counts the sum of the number of nonzero elements of each core
   */

  index_t number_nonzeros(const data_t tolerance = 1.0e-15) const
  {
    index_t sum = 0;
    for (size_t d = 0; d < dimension; d++)
    {
      sum += this->cores[d].number_nonzeros(tolerance);
    }
    return sum;
  }

  /**
   * Computes the sparsity ratio consistent with matrix sparsity
   */

  data_t sparsity(const data_t tolerance = 1.0e-15) const
  {
    data_t full_size = this->get_full_size();
    data_t nnz = this->number_nonzeros(tolerance);
    return full_size / nnz;
  }

  /**
   * \brief
   * A string describing the tensor train ranks
   */

  Array<size_t, dimension + 1> get_ranks() const
  {
    Array<size_t, dimension + 1> rarr;
    for (size_t d = 0; d < get_number_cores(); d++)
    {
      rarr[d] = get_ranks_left(d);
    }
    rarr[dimension] = get_ranks_right(dimension - 1);
    return rarr;
  }

  // -------------------------------------------------------------------------------------
  // Section: Debugging
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Returns true if the largest element outside of the bandwidth is smaller than tol.
   */

  void determine_bandwidth(data_t tol = 1.0e-13)
  {
    for (size_t d = 0; d < dimension; d++)
    {
      auto this_core_rows = this->core_rows(d);
      auto this_core_cols = this->core_cols(d);
      auto core_ranks_left = get_ranks_left(d);
      auto core_ranks_right = get_ranks_right(d);

      boba_always_assert_equal(this_core_rows, this_core_cols, "Additional work required for rectangular operators");

      Multiindexer<2> matrix_mider({this_core_rows, this_core_cols});

      auto core_view = this->cores[d].const_view();

      bandwidths.cores[d].resize({core_ranks_left, 1, core_ranks_right});
      auto bandwidths_view = bandwidths.cores[d].view();
      ::boba::loop<space, 2>({core_ranks_left, core_ranks_right},
                             [=] __boba_host_device__(Array<index_t, 2> rank_pair)
      {
        auto [ranks_left, ranks_right] = rank_pair;
        index_t value = 0;

        for (index_t i = 0; i < matrix_mider.size(); i++)
        {
          auto [row, col] = matrix_mider.multiindex(i);
          auto diagonal_value = core_view({ranks_left, row, row, ranks_right});
          auto core_value = core_view({ranks_left, row, col, ranks_right});

          if (abs(core_value) > abs(diagonal_value) * tol)
          {
            if (col > row)
            {
              value = max(value, col - row);
            }
            else if (col < row)
            {
              value = max(value, row - col);
            }
          }
        }

        bandwidths_view({ranks_left, 0, ranks_right}) = value;
      });
    }
  }

  /**
   * @return true if the bandwidths for this TTM have been determined or defined
   */

  bool is_bandwidths_defined() const
  {
    return product(bandwidths.sizes()) > 0;
  }

  // -------------------------------------------------------------------------------------
  // Section: Copy
  // -------------------------------------------------------------------------------------

  template <boba::execution_space input_space>
  TensorTrainMatrix& operator=(TensorTrainMatrix<dimension, input_space, data_t> const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    this->max_ranks = input.max_ranks;

    checkpoint();
    this->resize(input.core_rows(), input.core_cols());
    checkpoint();
    this->cores = input.cores;
    this->svd_tolerance_absolute = input.svd_tolerance_absolute;
    this->svd_tolerance_relative = input.svd_tolerance_relative;
    this->bandwidths = input.bandwidths;
    checkpoint();
    return *this;
  }

  TensorTrainMatrix transpose() const
  {
    BOBA_CALI_MARK
    checkpoint();
    TensorTrainMatrix output = *this;

    for (size_t d = 0; d < dimension; d++)
    {
      permute(
        {"l", "i", "j", "r"},
        output.cores[d],
        {"l", "j", "i", "r"});
    }
    checkpoint();
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Indexing
  // -------------------------------------------------------------------------------------

  size_t row_index(::boba::Array<size_t, dimension>& multiindex) const
  {
    return index(multiindex, core_rows());
  }

  size_t col_index(::boba::Array<size_t, dimension>& multiindex) const
  {
    return index(multiindex, core_cols());
  }

  size_t index(::boba::Array<size_t, dimension>& multiindex, ::boba::Array<size_t, dimension>& sizes) const
  {
    auto this_view = ::boba::Multiindexer<dimension>(sizes);
    return this_view.index(multiindex);
  }

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
    for (size_t d = 1; d < dimension; d++)
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
    for (size_t d = 0; d < dimension; d++)
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
    for (size_t d = 0; d < dimension; d++)
    {
      set_to_identity_core(this->cores[d]);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tensor train addition of this and subtrain, where subtrain is smaller than this.
   * The subtrain must have an offset and size for each dimension such that offset+size fits into this.
   * Zeros are padded to subtrain so that addition is valid.
   */

  void add_subtrain(
    TensorTrainMatrix const& subtrain,
    const ::boba::Array<size_t, dimension>& initial_row,
    const ::boba::Array<size_t, dimension>& initial_col)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_assert_ge(core_rows()[d], (initial_row[d] + subtrain.core_rows()[d]), "Subtrain size exceeds parent");
      boba_assert_ge(core_cols()[d], (initial_col[d] + subtrain.core_cols()[d]), "Subtrain size exceeds parent");
    }

    for (size_t d = 0; d < dimension; d++)
    {
      add_subcore(this->cores[d], subtrain.cores[d], d, dimension, initial_row[d], initial_col[d]);
    }

    checkpoint();
  }

  void TensorTrainMatrix_add(
    TensorTrainMatrix<dimension, space, data_t> const& input,
    ::boba::Array<size_t, dimension>& initial_row,
    ::boba::Array<size_t, dimension>& initial_col)
  {
    BOBA_CALI_MARK
    checkpoint();
    this->add_subtrain(input, initial_row, initial_col);
    checkpoint();
  }

  /**
   * \brief
   * Tensor train addition.
   */

  void TensorTrainMatrix_add(TensorTrainMatrix const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_assert_equal(input.core_rows()[d], this->core_rows()[d], "incompatible rows");
      boba_assert_equal(input.core_cols()[d], this->core_cols()[d], "incompatible cols");
    }
    checkpoint();

    ::boba::Array<size_t, dimension> initial_row = ::boba::filled_array<dimension>(0_z);
    ::boba::Array<size_t, dimension> initial_col = ::boba::filled_array<dimension>(0_z);

    checkpoint();
    this->add_subtrain(input, initial_row, initial_col);
    checkpoint();
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrainMatrix& operator+=(TensorTrainMatrix const& rhs)
  {
    BOBA_CALI_MARK
    this->TensorTrainMatrix_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TensorTrainMatrix operator+(TensorTrainMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrainMatrix output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrainMatrix& operator-=(TensorTrainMatrix const& rhs)
  {
    BOBA_CALI_MARK
    TensorTrainMatrix rhs_minus{rhs};
    rhs_minus *= static_cast<real_data_t>(-1);
    this->TensorTrainMatrix_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * negation operator
   */

  TensorTrainMatrix operator-()
  {
    BOBA_CALI_MARK
    TensorTrainMatrix output{*this};
    output *= static_cast<real_data_t>(-1);
    return output;
  }

  /**
   * \brief
   * output = this - rhs
   */

  TensorTrainMatrix operator-(TensorTrainMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrainMatrix output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * this *= input, using default methods
   */

  TensorTrainMatrix& operator*=(TensorTrainMatrix const& input)
  {
    BOBA_CALI_MARK
    this->TensorTrainMatrix_right_multiply(input);
    return *this;
  }

  /**
   * \brief
   * output = this * input
   */

  TensorTrainMatrix operator*(TensorTrainMatrix const& input) const
  {
    BOBA_CALI_MARK
    TensorTrainMatrix output{*this};
    output *= input;
    return output;
  }

  /**
   * \brief
   * output = this * input
   */

  TensorTrain<dimension, space, data_t> operator*(TensorTrain<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    return this->TensorTrainMatrix_vector_multiply(input);
  }

  /**
   * \brief
   * output = this * input vector
   */

  Vector<space, data_t> operator*(Vector<space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();
    Tensor<dimension, space, data_t> tensor_input(this->core_cols());
    tensor_input.reshape(input);
    auto output_tensor = this->TensorTrainMatrix_vector_multiply(tensor_input);
    Vector<space, data_t> output_vector({output_tensor.size()});
    output_vector.reshape(output_tensor);
    return output_vector;
  }

  /**
   * \brief
   * output = this * input tensor
   */

  Tensor<dimension, space, data_t> operator*(Tensor<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    return this->TensorTrainMatrix_vector_multiply(input);
  }

  /**
   * \brief
   * this *= scalar
   */

  TensorTrainMatrix& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    this->multiply_scalar(scalar);
    return *this;
  }

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  TensorTrainMatrix& operator*=(_real_data_t const scalar)
  {
    BOBA_CALI_MARK
    data_t complex_scalar{scalar, static_cast<_real_data_t>(0)};
    *this *= complex_scalar;
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  TensorTrainMatrix operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    TensorTrainMatrix output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matrix-Matrix Multiply
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Matrix-matrix multiply of two tensor train matrices.
   * this = this*input
   */

  void TensorTrainMatrix_right_multiply(
    TensorTrainMatrix const& input)
  {
    BOBA_CALI_MARK
    checkpoint();

    TensorTrainMatrix_right_multiply_vanilla(input);

    bool bandwidths_defined = input.is_bandwidths_defined() or this->is_bandwidths_defined();
    if (bandwidths_defined)
    {
      determine_bandwidth();
    }
  }

  /**
   * \brief
   * Matrix-matrix multiply of two tensor train matrices.
   * See TensorTrainMatrix_right_multiply
   * Basic algorithm using tensor contractions
   */

  void TensorTrainMatrix_right_multiply_vanilla(TensorTrainMatrix const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      Tensor<4, space, data_t> this_copy = this->cores[d];

      boba_always_assert_equal(
        input.core_rows(d),
        this->core_cols(d),
        "invalid size and cols for tensor-train mat-mat");

      auto temporary = tensor_contraction<1>(
        {"l1", "row", "col", "r1"}, this_copy, {"l2", "col", "new_col", "r2"}, input.cores[d], {"l1", "l2", "row", "new_col", "r1", "r2"});

      checkpoint();
      // temporary([l1, l2], row, new_col, [r1, r2]) = temporary(l1, l2, row, new_col, r1, r2)
      auto new_rows = this->core_rows(d);
      auto new_cols = input.core_cols(d);
      auto new_ranks_left = input.get_ranks_left(d) * this->get_ranks_left(d);
      auto new_ranks_right = input.get_ranks_right(d) * this->get_ranks_right(d);
      if ((new_ranks_left == 0) || (new_ranks_right == 0))
      {
        this->cores[d].fill_with_zeros();
        return;
      }
      this->cores[d].resize({new_ranks_left, new_rows, new_cols, new_ranks_right});
      this->cores[d].reshape(temporary);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Compress an arbitrarily sized matrix into this tensor train matrix via the tensor train algorithm.
   * Note the sizes must be consistent.
   */

  void compress_via_tt(boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK

    ::boba::Array<size_t, dimension> sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      sizes[d] = core_rows(d) * core_cols(d);
    }

    //
    // Recast matrix to tensor
    //
    boba::Tensor<dimension, space, data_t> tensor_recast(sizes);

    auto rows_view = ::boba::Multiindexer<dimension>(core_rows());
    auto cols_view = ::boba::Multiindexer<dimension>(core_cols());
    auto input_view = input.const_view();
    auto tensor_recast_view = tensor_recast.view();
    auto local_rows = core_rows();

    ::boba::loop<space, 2>(input.sizes(),
                           [=] __boba_host_device__(Array<size_t, 2> rc)
    {
      size_t r = rc[0];
      size_t c = rc[1];
      auto rows_multiindex = rows_view.multiindex(r);
      auto cols_multiindex = cols_view.multiindex(c);
      ::boba::Array<size_t, dimension> tensor_multiindex;
      for (size_t d = 0; d < dimension; d++)
      {
        tensor_multiindex[d] = rows_multiindex[d] + local_rows[d] * cols_multiindex[d];
      }
      tensor_recast_view(tensor_multiindex) = input_view({r, c});
    });

    TensorTrain<dimension, space, data_t> train_ttm(sizes);
    train_ttm.max_ranks = max_ranks;

    //
    // Compress tensor
    //
    train_ttm.compress(tensor_recast);

    //
    // Write to TTM
    //
    for (size_t d = 0; d < dimension; d++)
    {
      auto new_rank_left = train_ttm.get_ranks_left(d);
      auto new_rank_right = train_ttm.get_ranks_right(d);
      this->cores[d].resize({new_rank_left, core_rows()[d], core_cols()[d], new_rank_right});

      auto submatrix_view = ::boba::Multiindexer<dimension>({core_rows()[d], core_cols()[d]});
      auto tt_core_view = train_ttm.cores[d].const_view();
      auto ttm_core_view = this->cores[d].view();

      ::boba::loop<space, 3>({new_rank_left, train_ttm.sizes(d), new_rank_right},
                             [=] __boba_host_device__(Array<index_t, 3> ijkl)
      {
        auto [rl, i, rr] = ijkl;
        auto submatrix_mid = submatrix_view.multiindex({i});
        ttm_core_view({rl, submatrix_mid[0], submatrix_mid[1], rr}) = tt_core_view({rl, i, rr});
      });
    }
  }

  /**
   * \brief Reshape tt cores to ttm cores
   * Example - given tt cores (l, k, r), where k = i + I*j, we reshape (l, [i j], r) ->  (l, i, j, r)
   */

  void reshape_from_tt(const TensorTrain<dimension, space, data_t>& tt)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      auto left_ranks = tt.get_ranks_left(d);
      auto right_ranks = tt.get_ranks_right(d);
      this->cores[d].resize({left_ranks, core_rows()[d], core_cols()[d], right_ranks});
      this->cores[d].reshape(tt.cores[d]);
    }
  }

  /**
   * \brief
   * Compress an arbitrarily sized matrix into this tensor train matrix.
   * Note the sizes must be consistent.
   */

  void compress(const boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    if (dimension == 1)
    {
      cores[0].reshape(input);
      return;
    }

    // first, recast
    //   matrix([i0 i1 ...], [j0 j1 ...])
    // into a tensor:
    //  tensor(i0, i1, ..., j0, j1, ...)

    auto rows_and_cols = ::boba::concatenate(core_rows(), core_cols());

    checkpoint();
    ::boba::Tensor<2 * dimension, space, data_t> matrix_row_col_tensor(rows_and_cols);
    matrix_row_col_tensor.reshape(input);

    // permute tensor:
    //  tensor(i0, j0, i1, j1, ...)

    checkpoint();
    ::boba::Array<size_t, 2 * dimension> permutations;
    for (size_t d = 0; d < dimension; d++)
    {
      permutations[2 * d + 0] = d;
      permutations[2 * d + 1] = dimension + d;
    }

    checkpoint();
    permute(matrix_row_col_tensor, permutations);

    // reshape tensor so corresponding row/col indices are together
    //  tensor([i0 j0], [i1 j1], ...)

    checkpoint();
    ::boba::Array<size_t, dimension> sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      sizes[d] = core_rows(d) * core_cols(d);
    }

    // reshape tensor so corresponding row/col indices are together
    //  tensor([i0 j0], [i1 j1], ...)

    checkpoint();
    ::boba::Tensor<dimension, space, data_t> matrix_tensor(sizes);
    matrix_tensor.reshape(matrix_row_col_tensor);

    // use tt compression
    //  tt([i0 j0], [i1 j1], ...)

    checkpoint();
    auto tt_of_matrices = compress_to_TensorTrain(matrix_tensor, svd_tolerance_relative, svd_tolerance_absolute);
    tt_of_matrices.max_ranks = max_ranks;
    tt_of_matrices.round();

    reshape_from_tt(tt_of_matrices);

    checkpoint();
  }

  /**
   * \brief
   * Left-orthogonalize this tensor train matrix by a left-to-right QR sweep.
   *
   * After this operation, all cores except the last are left-orthogonal. The
   * triangular factor from each QR factorization is absorbed into the left rank
   * index of the next core, so the represented tensor train matrix is unchanged
   * up to floating-point roundoff.
   */
  void orthogonalize()
  {
    BOBA_CALI_MARK
    checkpoint();

    if constexpr (dimension == 1_z)
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

    for (size_t d = 0_z; d + 1_z < dimension; d++)
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
        boba::tensor_contraction<1>(
          {"k", "l"}, qr.R, {"l", "row", "col", "r"}, this->cores[d + 1_z], {"k", "row", "col", "r"});
    }

    checkpoint();
  }

  /**
   * \brief
   * Round this tensor train matrix by left-orthogonalization followed by a
   * right-to-left truncated SVD sweep.
   *
   * The method first left-orthogonalizes the tensor train matrix using a QR
   * sweep. It then performs a right-to-left SVD sweep, truncating each interface
   * rank according to svd_tolerance_relative, svd_tolerance_absolute, and
   * max_ranks. The resulting tensor train matrix is right-orthogonal, with the
   * first core as the orthogonality center.
   */
  void round()
  {
    BOBA_CALI_MARK
    checkpoint();

    if constexpr (dimension == 1_z)
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

    for (size_t d = dimension - 1_z; d > 0_z; d--)
    {
      checkpoint();

      // This SVD truncates the interface rank r_d.
      svd.max_kept_singular_values = max_ranks[d];

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
        boba::tensor_contraction<1>(
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

  /**
   * \brief
   * Multiplies each core of the tensor train by its own scalar, such that A = product(scalars)*A
   */

  void multiply_scalars(::boba::Array<data_t, dimension>& scalars)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d] *= scalars[d];
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Quadratic terms
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Implemented a quadratic-like term in terms of tensor train matrices.
   * output = variable*this*variable, where * is the tensor-train matrix-matrix product.
   */

  void apply_quadratic(
    const TensorTrainMatrix& variable,
    TensorTrainMatrix& output)
  {
    BOBA_CALI_MARK

    apply_quadratic_vanilla(variable, output);
  }

  /**
   * \brief
   * Implemented a quadratic-like term in terms of tensor train matrices.
   * output = variable^T*this*variable, where * is the tensor-train matrix-matrix product.
   */

  void apply_quadratic_vanilla(
    const TensorTrainMatrix& variable,
    TensorTrainMatrix& output)
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < dimension; d++)
    {
      checkpoint();
      boba_always_assert_equal(
        this->core_rows(d),
        variable.core_cols(d),
        "invalid size and cols for apply_quadratic left product");

      boba_always_assert_equal(
        this->core_cols(d),
        variable.core_rows(d),
        "invalid size and cols for apply_quadratic right product");

      // variable^T*this*variable
      auto intermediate_value = tensor_contraction<1>(
        {"vT_l", "row", "new_row", "vT_r"}, variable.cores[d], {"ml", "row", "col", "mr"}, this->cores[d], {"vT_l", "new_row", "vT_r", "ml", "col", "mr"});

      auto output_temporary = tensor_contraction<1>(
        {"vT_l", "new_row", "vT_r", "ml", "col", "mr"}, intermediate_value, {"v_l", "col", "new_col", "v_r"}, variable.cores[d], {"vT_l", "ml", "v_l", "new_row", "new_col", "vT_r", "mr", "v_r"});

      auto new_rows = variable.core_rows(d);
      auto new_cols = variable.core_cols(d);
      auto this_ranks_left = this->get_ranks_left(d);
      auto this_ranks_right = this->get_ranks_right(d);
      auto variable_ranks_left = variable.get_ranks_left(d);
      auto variable_ranks_right = variable.get_ranks_right(d);

      auto new_ranks_left = variable_ranks_left * this_ranks_left * variable_ranks_left;
      auto new_ranks_right = variable_ranks_right * this_ranks_right * variable_ranks_right;

      output.cores[d].resize({new_ranks_left, new_rows, new_cols, new_ranks_right});
      output.cores[d].reshape(output_temporary);
    }
    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Inverse
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Takes an initial guess inverse and computes the tensor train matrix inverse of this.
   * If oseledets_scheme is true, Oseledet's algorithm is used.
   * If false, an alternative two-step algorithm is provided.
   */

  bool compute_inverse(
    TensorTrainMatrix& inverse,
    data_t& residual_final,
    data_t tolerance,
    data_t maximum_iterations,
    size_t scheme_option,
    bool verbose)
  {
    BOBA_CALI_MARK
    bool success = false;
    checkpoint();
    BOBA_CALI_BEGIN("setup");
    boba_always_assert_equal(core_rows(), core_cols(), "nonsquare inverse not yet implemented");

    this->round();
    inverse.round();
    checkpoint();

    checkpoint();
    TensorTrainMatrix inverse_temp(core_rows(), core_cols());
    inverse_temp.rename("inverse_temp");
    checkpoint();
    TensorTrainMatrix identity(core_rows(), core_cols());
    identity.rename("ttm_identity");
    identity.set_to_identity_train();
    checkpoint();

    data_t initial_convergence = 0.0;
    {
      TensorTrainMatrix convergence_check(core_rows(), core_cols());
      convergence_check.rename("convergence_check");
      convergence_check = inverse;
      convergence_check.TensorTrainMatrix_right_multiply(
        *this);
      checkpoint();
      initial_convergence = norm_difference_frobenius(convergence_check, identity);
    }

    if (verbose)
    {
      std::cout << write_indent(1)
                << "% iterations,||R|| residual metric, time, compression rate, ranks"
                << std::endl;
    }

    BOBA_CALI_SWITCH("setup", "iterating");

    size_t iterations_used = 0;
    bool iterating = true;
    data_t convergence = -1.0;
    while (iterating)
    {
      boba::TicToc<boba::tictoc_units::milliseconds> iterate;
      if (scheme_option == 0)
      {
        // scheme is X_new = 2*X - X*A*X
        checkpoint();
        this->apply_quadratic(
          inverse,
          inverse_temp);
        checkpoint();

        inverse_temp.round();

        inverse_temp *= -1.0;
        checkpoint();
        inverse *= 2.0;
        checkpoint();
        inverse += inverse_temp;
        inverse_temp.fill_with_zeros();
      }
      else if (scheme_option == 1)
      {
        checkpoint();
        /*
          From:  https://jmm.guilan.ac.ir/article_4759.html
          Q = Ax. P = Q*(2I - Q).
          X_new = X(2I - Q)*(3I - P*(3I - P)).
        */
        // this is A
        // inverse is X

        TensorTrainMatrix inverse_temp_I2(core_rows(), core_cols());

        inverse_temp_I2.rename("Identity_x2");
        inverse_temp_I2.set_to_identity_train();
        inverse_temp_I2 *= 2.0;

        // Q = Ax^n
        inverse_temp = *this;
        inverse_temp.TensorTrainMatrix_right_multiply(
          inverse);

        inverse_temp.round();

        TensorTrainMatrix inverse_temp_Q(core_rows(), core_cols());
        inverse_temp_Q.rename("inverse_temp_Q");
        inverse_temp_Q = inverse_temp;

        // Y = 2I - Q, P = QY
        inverse_temp_Q *= -1.0;
        inverse_temp_I2 += inverse_temp_Q;
        inverse_temp_Q *= -1.0;
        inverse_temp_Q.TensorTrainMatrix_right_multiply(
          inverse_temp_I2);

        inverse_temp_Q.round();

        TensorTrainMatrix inverse_temp_P = -inverse_temp_Q;
        inverse_temp_P.rename("inverse_temp_P");

        // M = 3I - P

        TensorTrainMatrix inverse_temp_I3(core_rows(), core_cols());
        inverse_temp_I3.rename("Identity_x3");
        inverse_temp_I3.set_to_identity_train();
        inverse_temp_I3 *= 3.0;
        inverse_temp_I3 += inverse_temp_P;

        inverse_temp_I3.round();

        TensorTrainMatrix inverse_temp_M(core_rows(), core_cols());
        inverse_temp_M.rename("inverse_temp_M");
        inverse_temp_M = inverse_temp_I3;

        inverse_temp_I3.set_to_identity_train();
        inverse_temp_I3 *= 3.0;

        inverse_temp_P.TensorTrainMatrix_right_multiply(
          inverse_temp_M);

        inverse_temp_P.round();

        inverse_temp_I3 += inverse_temp_P;

        TensorTrainMatrix inverse_temp_N(core_rows(), core_cols());
        inverse_temp_N.rename("inverse_temp_N");
        inverse_temp_N = inverse_temp_I3;

        inverse_temp_I2.TensorTrainMatrix_right_multiply(
          inverse_temp_N);

        inverse_temp_I2.round();

        inverse.TensorTrainMatrix_right_multiply(
          inverse_temp_I2);

        inverse.round();
      }
      else
      {
        // scheme is X_new = X*2*( I - 0.5*A*X )
        checkpoint();
        inverse_temp = *this;
        checkpoint();

        inverse_temp.TensorTrainMatrix_right_multiply(
          inverse);

        inverse_temp.round();

        inverse_temp *= -0.5;
        checkpoint();

        inverse_temp += identity;
        inverse_temp *= 2.0;

        inverse.TensorTrainMatrix_right_multiply(
          inverse_temp);

        inverse.round();

        inverse_temp.fill_with_zeros();
      }

      inverse.round();

      {
        TensorTrainMatrix convergence_check(core_rows(), core_cols());
        convergence_check.rename("convergence_check");
        convergence_check = inverse;
        convergence_check.TensorTrainMatrix_right_multiply(
          *this);
        convergence = norm_difference_frobenius(convergence_check, identity);
        convergence /= initial_convergence;
      }
      checkpoint();
      size_t iteration_time = iterate.timing();
      checkpoint();
      if (verbose)
      {
        std::cout << write_indent(1)
                  << "% " << iterations_used
                  << "  ||R|| " << convergence
                  << "  " << iterate.units_string << " " << iteration_time
                  << "  CR " << inverse.compression_rate() << "x  "
                  << inverse.ranks_string()
                  << std::endl;
      }
      iterations_used++;
      size_t iteration_timeout = iterate.convert<boba::tictoc_units::seconds>(100 * 60);
      residual_final = convergence;
      if constexpr (boba::is_boba_debug_mode())
      {
        iteration_timeout *= 100;
      }
      if (iteration_time > iteration_timeout)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << " iteration timeout failure, "
                    << iteration_time << " > " << iteration_timeout
                    << " " << iterate.units_string << std::endl;
        }
        iterating = false;
        residual_final = 1;
        success = true;
      }
      if (convergence < tolerance)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << " convergence achieved = " << convergence << std::endl;
        }
        iterating = false;
        success = true;
      }
      if (iterations_used >= maximum_iterations)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << "% maximum iterations reached " << std::endl;
        }
        if (!boba::isnan(residual_final))
        {
          success = true;
        }
        iterating = false;
        success = true;
      }
      checkpoint();
    }
    checkpoint();
    if (verbose)
    {
      //::boba::detail::print_umpire_stats(stdout);
      boba_print(residual_final);
    }
    return success;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matrix-vector Multiply
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * See TensorTrainMatrix_vector_multiply.
   */

  TensorTrain<dimension, space, data_t> apply(
    TensorTrain<dimension, space, data_t> const& input)
  {
    return this->TensorTrainMatrix_vector_multiply(input);
  }

  /**
   * \brief
   * Takes an input tensor train and performs a matrix-vector multiply in the sense of tensor trains.
   * At runtime, dispatches the basic or inline-rounding version of the underlying routine.
   */

  TensorTrain<dimension, space, data_t> TensorTrainMatrix_vector_multiply(
    TensorTrain<dimension, space, data_t> const& input) const
  {
    return TensorTrainMatrix_vector_multiply_vanilla(input);
  }

  /**
   * \brief
   * Takes an input tensor train and performs a matrix-vector multiply in a sum-factorized way
    // Example for 4D: Given input T_nmst
    // T_1nmst = T_nmst
    // A_1nnr B_rmml C_lssp D_ptt1 T_1nmst
    //         contractions     ^^   ^   ^
    // A_1nnr B_rmml C_lssp T_ptnms
    //                   ^^   ^   ^
    // A_1nnr B_rmml T_lstnm
    //            ^^   ^   ^
    // A_1nnr T_rmstn
    //     ^^   ^   ^
    // T_nmst = T_n1mst
    // Note that 1D is a special case
   */

  Tensor<dimension, space, data_t> TensorTrainMatrix_vector_multiply(
    Tensor<dimension, space, data_t> const& input,
    bool use_low_memory_method = true) const
  {
    checkpoint();
    use_low_memory_method = use_low_memory_method and env_match("TTMVEC_METHOD", "LOWMEM");

    Tensor<dimension, space, data_t> reshaped_output(this->core_rows());

    if (use_low_memory_method)
    {
      reshaped_output.fill_with_zeros();

      // Get a multiindexer of all possible rank choices
      Multiindexer<dimension + 1> ranks_mider(this->get_ranks());
      for (size_t r = 0; r < ranks_mider.size(); r++)
      {
        auto rank_one_term = extract_rank_one_ttm(ranks_mider.multiindex(r));
        auto rank_one_result = rank_one_term.TensorTrainMatrix_vector_multiply(input, false);
        reshaped_output += rank_one_result;
      }

      return reshaped_output;
    }

    if constexpr (dimension == 1)
    {
      checkpoint();
      auto output = tensor_contraction<1>(
        {"l", "i", "j", "r"}, this->cores[0], {"j"}, input, {"l", "i", "r"});

      reshaped_output.reshape(output);
      return reshaped_output;
    }
    else
    {
      checkpoint();
      // Move reshape
      Array<size_t, dimension + 1> reshape_sizes;
      reshape_sizes[0] = 1_z;
      for (size_t d = 1; d < dimension + 1; d++)
      {
        reshape_sizes[d] = input.sizes(d - 1);
      }

      checkpoint();
      Tensor<dimension + 1, space, data_t> output(reshape_sizes);
      output.reshape(input);

      checkpoint();
      for (size_t d = dimension; d > 0; d--)
      {
        auto new_output = tensor_contraction<2>(
          this->cores[d - 1],
          output,
          {2, 3},
          {dimension, 0});

        checkpoint();
        output = new_output;
      }

      checkpoint();
      reshaped_output.reshape(output);
      return reshaped_output;
    }
  }

  /**
   * \brief
   * TensorTrainMatrix_vector_multiply
   * Each core performs a contraction according to
   * output_d{i, [ri rk], [rj rl]} = sum_k ttm_d{i, k, ri, rj}*tt_d{k, rk, rl}
   * where "_d", refers to the d'th core.
   */

  TensorTrain<dimension, space, data_t> TensorTrainMatrix_vector_multiply_vanilla(
    TensorTrain<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    TensorTrain<dimension, space, data_t> output(input.sizes());
    output.svd_tolerance_relative = input.svd_tolerance_relative;
    output.svd_tolerance_absolute = input.svd_tolerance_absolute;

    bool use_bandwidth = product(bandwidths.sizes()) > 0;

    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(this->core_cols(d), input.sizes(d), "invalid size and cols for tensor-train matvec");

      size_t new_rows = this->core_rows(d);

      Array<size_t, 2> new_rank_left{get_ranks_left(d), input.get_ranks_left(d)};
      Array<size_t, 2> new_rank_right{get_ranks_right(d), input.get_ranks_right(d)};
      auto ranks_left_view = ::boba::Multiindexer<2>(new_rank_left);
      auto ranks_right_view = ::boba::Multiindexer<2>(new_rank_right);
      size_t new_ranks_left = ranks_left_view.size();
      size_t new_ranks_right = ranks_right_view.size();

      if ((new_ranks_left == 0) || (new_ranks_right == 0))
      {
        output.fill_with_zeros();
        return output;
      }

      output.cores[d].resize({new_ranks_left, new_rows, new_ranks_right});

      if (use_bandwidth)
      {
        auto common = this->core_cols(d);

        auto bandwidth_view = bandwidths.cores[d].view();

        auto input_core_view = input.cores[d].const_view();
        auto this_core_view = this->cores[d].const_view();
        auto output_core_view = output.cores[d].view();

        ::boba::loop<space, 3>(output_core_view.sizes(),
                               [=] __boba_host_device__(Array<index_t, 3> lir)
        {
          auto [rank_left, row, rank_right] = lir;
          auto [rank_this_left, rank_in_left] = ranks_left_view.multiindex(rank_left);
          auto [rank_this_right, rank_in_right] = ranks_right_view.multiindex(rank_right);
          auto local_bandwidth = bandwidth_view({rank_this_left, 0, rank_this_right});

          double sum = 0.0;
          auto kmin = mod(max((common + row) - local_bandwidth, common), common);
          auto kmax = mod(min((common + row) + local_bandwidth, 2 * common - 1), common);
          for (auto kp = kmin; kp <= kmax; kp++)
          {
            const auto value_matrix = this_core_view({rank_this_left, row, kp, rank_this_right});
            const auto value_in = input_core_view({rank_in_left, kp, rank_in_right});
            sum += value_matrix * value_in;
          }
          output_core_view({rank_left, row, rank_right}) = sum;
        });
      }
      else
      {
        auto temporary = tensor_contraction<1>(
          {"ml", "row", "col", "mr"}, this->cores[d], {"l", "col", "r"}, input.cores[d], {"ml", "l", "row", "mr", "r"});

        output.cores[d] = reshape<3>(temporary, {new_ranks_left, new_rows, new_ranks_right});
      }
    }
    checkpoint();
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Extract rank one terms
  // -------------------------------------------------------------------------------------

  TensorTrainMatrix extract_rank_one_ttm(
    const Array<size_t, dimension + 1> rank_indices) const
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_ge(rank_indices[d], 0_z, "Must be positive.");
      boba_always_assert_lt(rank_indices[d], get_ranks_left(d), "Attempting to access an out of bounds rank.");
      boba_always_assert_lt(rank_indices[d + 1], get_ranks_right(d), "Attempting to access an out of bounds rank.");
    }

    checkpoint();
    TensorTrainMatrix rank_one(core_rows(), core_cols());
    rank_one.fill_with_zeros();

    for (size_t d = 0; d < dimension; d++)
    {
      auto core_view = cores[d].const_view();
      auto rank_one_view = rank_one.cores.at(d).view();

      auto rank_index_left = rank_indices[d];
      auto rank_index_right = rank_indices[d + 1];

      checkpoint();
      ::boba::loop<space, 2>({core_rows(d), core_cols(d)},
                             [=] __boba_host_device__(Array<size_t, 2> ij)
      {
        auto x = core_view({rank_index_left, ij[0], ij[1], rank_index_right});
        rank_one_view({0_z, ij[0], ij[1], 0_z}) = x;
      });
    }

    checkpoint();
    return rank_one;
  }

  // -------------------------------------------------------------------------------------
  // Section: Get matrix
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> get_matrix(size_t core_dim, size_t rank_left, size_t rank_right)
  {
    BOBA_CALI_MARK

    checkpoint();
    const size_t submatrix_rows = this->core_rows(core_dim);
    const size_t submatrix_cols = this->core_cols(core_dim);

    boba_always_assert_lt(rank_left, get_ranks_left(core_dim), "Out of bounds on ranks");
    boba_always_assert_lt(rank_right, get_ranks_right(core_dim), "Out of bounds on ranks");

    boba::Matrix<space, data_t> submatrix({submatrix_rows, submatrix_cols});

    auto submatrix_view = submatrix.view();
    auto core_view = this->cores[core_dim].const_view();

    checkpoint();
    ::boba::loop<space, 2>({submatrix_rows, submatrix_cols},
                           [=] __boba_host_device__(Array<size_t, 2> rc)
    {
      size_t r = rc[0];
      size_t c = rc[1];
      submatrix_view({r, c}) = core_view({rank_left, r, c, rank_right});
    });
    checkpoint();
    return submatrix;
  }
};

template <size_t dimension, ::boba::execution_space space, typename _data_t>
TensorTrainMatrix<dimension, space, _data_t> operator*(_data_t scalar, TensorTrainMatrix<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  TensorTrainMatrix<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

template <execution_space space, typename _data_t>
void add_subcore(
  Tensor<4, space, _data_t>& this_core,
  Tensor<4, space, _data_t> const& subcore,
  size_t core_dim,
  size_t train_dimension,
  const size_t initial_index_rows,
  const size_t initial_index_cols)
{
  BOBA_CALI_MARK

  /*
    current    added
     ranks     ranks
  _____________________
  |         |         |
  |  core   |    0    |  current ranks
  |_________|_________|
  |         |         |
  |    0    | subcore |  added ranks
  |_________|_________|

  Leftmost core

    current    added
     ranks     ranks
  _____________________
  |         |         |
  |  core   | subcore |  ranks = 1
  |_________|_________|

  Rightmost core

   ranks = 1
  ___________
  |         |
  |  core   | current ranks
  |_________|
  |         |
  | subcore | added ranks
  |_________|
  */

  // For each block
  /*
  _____________
  |   |   |   | <- 0
  | 0 | 0 | 0 |
  |___|___|___| <- initial_index_rows
  |   |   |   |
  | 0 | X | 0 |
  |___|___|___| <- initial_index_rows + sub_rows
  |   |   |   |
  | 0 | 0 | 0 |
  |___|___|___| <- initial_index_rows + sub_rows + end_rows
  ^   ^   ^   ^
  |   |   |   initial_index_cols + sub_cols + end_cols
  |   |   initial_index_cols + sub_cols
  |   |
  0   initial_index_cols
  */

  if (train_dimension == 1)
  {
    auto subtrain_rows = subcore.sizes(1);
    auto subtrain_cols = subcore.sizes(2);
    boba_always_assert_equal(subcore.sizes(0), 1_z, "Unexpected ranks for core");
    boba_always_assert_equal(subcore.sizes(3), 1_z, "Unexpected ranks for core");
    auto subtrain_view = subcore.const_view();
    auto this_view = this_core.view();
    ::boba::loop<space, 2>({subtrain_rows, subtrain_cols},
                           [=] __boba_host_device__(Array<size_t, 2> ij)
    {
      auto [i, j] = ij;
      this_view({0_z, initial_index_rows + i, initial_index_cols + j, 0_z}) += subtrain_view({0_z, i, j, 0_z});
    });
    return;
  }

  checkpoint();
  const size_t added_ranks_left = subcore.sizes(0);
  const size_t added_ranks_right = subcore.sizes(3);
  const size_t sub_rows = subcore.sizes(1);
  const size_t sub_cols = subcore.sizes(2);
  const size_t this_ranks_left = this_core.sizes(0);
  const size_t this_ranks_right = this_core.sizes(3);

  size_t new_ranks_left = this_ranks_left + added_ranks_left;
  size_t new_ranks_right = this_ranks_right + added_ranks_right;

  size_t this_ranks_left_offset = this_ranks_left;
  size_t this_ranks_right_offset = this_ranks_right;

  if (core_dim == 0)
  {
    new_ranks_left = this_ranks_left;
    this_ranks_left_offset = 0;
  }
  if (core_dim == train_dimension - 1)
  {
    new_ranks_right = this_ranks_right;
    this_ranks_right_offset = 0;
  }

  const size_t rows = this_core.sizes(1);
  const size_t cols = this_core.sizes(2);
  const size_t end_nonzero_cols = initial_index_cols + sub_cols;
  const size_t end_nonzero_rows = initial_index_rows + sub_rows;

  boba_assert_ge(rows, (initial_index_rows + sub_rows), "Subtrain size exceeds parent");
  boba_assert_ge(cols, (initial_index_cols + sub_cols), "Subtrain size exceeds parent");

  checkpoint();
  this_core.resize({new_ranks_left, rows, cols, new_ranks_right});

  checkpoint();
  auto this_view = this_core.view();
  auto subcore_view = subcore.const_view();

  checkpoint();
  ::boba::loop<space, 1>(this_view.size(),
                         [=] __boba_host_device__(size_t I)
  {
    auto [rank_left, row, col, rank_right] = this_view.multiindex(I);
    auto x = _data_t(0.0);

    // are the rank-indices on the support of the core?
    bool is_core_ranks_left = rank_left < this_ranks_left;

    bool is_core_ranks_right = rank_right < this_ranks_right;

    bool is_core = is_core_ranks_right && is_core_ranks_left;

    if (is_core)
    {
      x = this_view({rank_left, row, col, rank_right});
    }

    // are the row/col indices on the support of the subcore?
    bool col_upper_bound = col < end_nonzero_cols;
    bool col_lower_bound = col >= initial_index_cols;
    bool row_upper_bound = row < end_nonzero_rows;
    bool row_lower_bound = row >= initial_index_rows;
    bool is_subcore_row_col = col_upper_bound && col_lower_bound && row_upper_bound && row_lower_bound;

    // are the rank-indices on the support of the subcore?
    bool is_subcore_ranks_left_lower_bound = rank_left >= this_ranks_left_offset;
    bool is_subcore_ranks_left_upper_bound = rank_left < new_ranks_left;
    bool is_subcore_ranks_left = is_subcore_ranks_left_lower_bound && is_subcore_ranks_left_upper_bound;

    bool is_subcore_ranks_right_lower_bound = rank_right >= this_ranks_right_offset;
    bool is_subcore_ranks_right_upper_bound = rank_right < new_ranks_right;
    bool is_subcore_ranks_right = is_subcore_ranks_right_lower_bound && is_subcore_ranks_right_upper_bound;

    bool is_subcore_ranks = is_subcore_ranks_right && is_subcore_ranks_left;

    bool is_subcore = is_subcore_ranks && is_subcore_row_col;
    if (is_subcore)
    {
      const size_t sub_row = row - initial_index_rows;
      const size_t sub_col = col - initial_index_cols;
      const size_t sub_rank_left = rank_left - this_ranks_left_offset;
      const size_t sub_rank_right = rank_right - this_ranks_right_offset;
      x = subcore_view({sub_rank_left, sub_row, sub_col, sub_rank_right});
    }
    this_view({rank_left, row, col, rank_right}) = x;
  });
}

// -------------------------------------------------------------------------------------
// Section: Quadratric operation
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Given an operator A in the ttmatrix format (this, aka "op") and an input ttmatrix x,
 * performs perform x^T*A*x.
 */

template <execution_space space, typename _data_t>
void core_apply_quadratic(
  const Tensor<4, space, _data_t>& this_core,
  const Tensor<4, space, _data_t>& variable,
  Tensor<4, space, _data_t>& output,
  boba::QR<space, _data_t> const& qr)
{
  BOBA_CALI_MARK
  detail::ignore(qr);
  checkpoint();
  size_t this_ranks_left = this_core.sizes(0);
  size_t this_ranks_right = this_core.sizes(3);
  size_t op_ranks_left = variable.sizes(0);
  size_t op_ranks_right = variable.sizes(3);

  Multiindexer<3> ranks_left({op_ranks_left, this_ranks_left, op_ranks_left});
  Multiindexer<3> ranks_right({op_ranks_right, this_ranks_right, op_ranks_right});

  size_t total_ranks_left = ranks_left.size();
  size_t total_ranks_right = ranks_right.size();

  size_t variable_rows = variable.sizes(1);
  size_t variable_cols = variable.sizes(2);

  size_t new_ranks_left = total_ranks_left;

  output.resize({new_ranks_left, this_core.sizes(1), this_core.sizes(2), total_ranks_right});
  checkpoint();

  auto this_view = this_core.const_view();
  auto variable_view = variable.const_view();
  auto output_view = output.view();

  checkpoint();
  ::boba::loop<space, 1>(output.size(),
                         [=] __boba_host_device__(size_t I)
  {
    auto [rank_left, r, c, rank_right] = output.multiindex(I);

    auto left_indices = ranks_left.multiindex(rank_left);
    size_t left_variable_rank_left = left_indices[0];
    size_t this_rank_left = left_indices[1];
    size_t right_variable_rank_left = left_indices[2];

    auto right_indices = ranks_right.multiindex(rank_right);
    size_t left_variable_rank_right = right_indices[0];
    size_t this_rank_right = right_indices[1];
    size_t right_variable_rank_right = right_indices[2];

    auto sum = _data_t(0.0);
    for (size_t k = 0; k < variable_rows; k++)
    {
      auto xT = variable_view({left_variable_rank_left, r, k, left_variable_rank_right});
      for (size_t p = 0; p < variable_cols; p++)
      {
        auto A = this_view({this_rank_left, k, p, this_rank_right});
        auto x = variable_view({right_variable_rank_left, p, c, right_variable_rank_right});
        sum += xT * A * x;
      }
    }
    output_view({rank_left, r, c, rank_right}) = sum;
  });
  checkpoint();
}

// -------------------------------------------------------------------------------------
// Section: Left/right folds
// -------------------------------------------------------------------------------------

// ------------------------------------
// Left fold functions
// ------------------------------------
template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_left(
  ::boba::Tensor<4, space, _data_t>& core)
{
  BOBA_CALI_MARK
  checkpoint();

  const auto ranks_left = core.sizes(0);
  const auto rows = core.sizes(1);
  const auto cols = core.sizes(2);
  const auto ranks_right = core.sizes(3);
  const auto size = rows * cols;

  ::boba::Tensor<3, space, _data_t> tt_core({ranks_left, size, ranks_right});
  tt_core.reshape(core);

  return compute_unfold_left(tt_core);
}

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_right(
  ::boba::Tensor<4, space, _data_t>& core)
{
  BOBA_CALI_MARK

  const auto ranks_left = core.sizes(0);
  const auto rows = core.sizes(1);
  const auto cols = core.sizes(2);
  const auto ranks_right = core.sizes(3);
  const auto size = rows * cols;

  ::boba::Tensor<3, space, _data_t> tt_core({ranks_left, size, ranks_right});
  tt_core.reshape(core);

  return compute_unfold_right(tt_core);
}

template <execution_space space, typename _data_t>
::boba::Tensor<4, space, _data_t> write_to_core_from_left_fold(
  boba::Matrix<space, _data_t> const& unfold_left,
  const index_t rows,
  const index_t cols)
{
  BOBA_CALI_MARK
  checkpoint();

  auto tt_core = write_to_core_from_left_fold(unfold_left, rows * cols);
  auto ranks_left = tt_core.sizes(0);
  auto ranks_right = tt_core.sizes(2);

  ::boba::Tensor<4, space, _data_t> ttm_core({ranks_left, rows, cols, ranks_right});
  ttm_core.reshape(tt_core);

  return ttm_core;
}

template <execution_space space, typename _data_t>
::boba::Tensor<4, space, _data_t> write_to_core_from_right_fold(
  boba::Matrix<space, _data_t> const& unfold_right,
  const size_t rows,
  const size_t cols)
{
  BOBA_CALI_MARK

  auto tt_core = write_to_core_from_right_fold(unfold_right, rows * cols);
  auto ranks_left = tt_core.sizes(0);
  auto ranks_right = tt_core.sizes(2);

  ::boba::Tensor<4, space, _data_t> ttm_core({ranks_left, rows, cols, ranks_right});
  ttm_core.reshape(tt_core);

  return ttm_core;
}

// ------------------------------------
// Right-to-left and vice versa
// ------------------------------------

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_right_from_unfold_left(
  boba::Matrix<space, _data_t> const& unfold_left,
  const size_t rows,
  const size_t cols)
{
  BOBA_CALI_MARK
  checkpoint();

  return compute_unfold_right_from_unfold_left(unfold_left, rows * cols);
}

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_left_from_unfold_right(
  boba::Matrix<space, _data_t> const& unfold_left,
  const size_t rows,
  const size_t cols)
{
  BOBA_CALI_MARK
  checkpoint();

  return compute_unfold_left_from_unfold_right(unfold_left, rows * cols);
}

// -------------------------------------------------------------------------------------
// Section: Diagonalize tensor train
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Lays a tensor-train along the diagonal of a tensor-train-matrix.
 * In matlab, this would be like diag(vector).
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
TensorTrainMatrix<dimension, space, _data_t>
diagonalize(TensorTrain<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  TensorTrainMatrix<dimension, space, _data_t> output(input.sizes(), input.sizes());
  checkpoint();
  for (size_t d = 0; d < dimension; d++)
  {
    size_t new_ranks_left = input.get_ranks_left(d);
    size_t new_ranks_right = input.get_ranks_right(d);
    size_t tt_size = input.sizes(d);
    checkpoint();
    output.cores[d].resize({new_ranks_left, tt_size, tt_size, new_ranks_right});
    output.cores[d].fill_with_zeros();
    checkpoint();
    auto input_core_view = input.cores[d].const_view();
    auto output_core_view = output.cores[d].view();
    checkpoint();
    ::boba::loop<space, 3>({new_ranks_left, new_ranks_right, tt_size},
                           [=] __boba_host_device__(Array<size_t, 3> ijk)
    {
      size_t rank_left = ijk[0];
      size_t rank_right = ijk[1];
      size_t index = ijk[2];
      const auto value_in = input_core_view({rank_left, index, rank_right});
      output_core_view({rank_left, index, index, rank_right}) = value_in;
    });
  }
  checkpoint();
  return output;
}

// -------------------------------------------------------------------------------------
// Section: i/o
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Write the ttm to file in a way consistent with Tensor::write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_file(const TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = ttm.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  for (size_t d = 0; d < dimension; d++)
  {
    boba::write_to_file(ttm.cores[d], print_filename + "_core_" + std::to_string(d));
  }
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_file(TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = ttm.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  for (size_t d = 0; d < dimension; d++)
  {
    boba::read_from_file(ttm.cores[d], print_filename + "_core_" + std::to_string(d));
  }
}

/**
 * \brief
 * Write the tt to a MATLAB mat-file as a cell array of cores.
 *
 * The resulting cell array can be converted to Oseledet's TT-toolbox tt_matrix class object:
 * ```matlab
 * >> load <filename>.mat; % creates a cell array named `filename` in the workspace
 * >> ttm = cell2core(tt_matrix, filename);
 * ```
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_mat_file(const TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = ttm.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "w");
  matlab_file.write_cell_array(print_filename, ttm.cores);
}

/**
 * \brief
 * Read from a file generated by write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_mat_file(TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = ttm.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "r");
  matlab_file.read_cell_array(print_filename, ttm.cores);
}

/**
 * \brief
 * Write the tt to a HDF5 file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_hdf5_file(const TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = ttm.name();
  }

  detail::HDF5File h5_file(filename, "w");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.write_array(object_name + "_core_" + std::to_string(d), ttm.cores[d]);
  }
}

/**
 * \brief
 * Read from a file generated by write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_hdf5_file(TensorTrainMatrix<dimension, space, _data_t>& ttm, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = ttm.name();
  }

  detail::HDF5File h5_file(filename, "r");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.read_array(object_name + "_core_" + std::to_string(d), ttm.cores[d]);
  }
}

} // namespace boba
