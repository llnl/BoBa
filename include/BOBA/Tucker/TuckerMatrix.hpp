// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tucker decomposition of a Matrix
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
struct TuckerMatrix
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "TuckerMatrix";
  ::boba::Array<size_t, dimension + 1> max_ranks = ::boba::filled_array<dimension + 1>(highest_value<size_t>());

  data_t svd_tolerance_relative = 1.0e-12;
  data_t svd_tolerance_absolute = 1.0e-12;

  ::boba::Array<Tensor<3, space, data_t>, dimension> cores;
  Tensor<dimension, space, data_t> R_core;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize.
   * Instantiated object is the zero TuckerMatrix.
   */

  TuckerMatrix(::boba::Array<index_t, dimension> input_rows, ::boba::Array<index_t, dimension> input_cols)
  {
    this->resize(input_rows, input_cols);
  }

  // Default constructor
  TuckerMatrix() = default;

  // Copy constructor
  TuckerMatrix(TuckerMatrix const&) = default;

  // Move constructor
  TuckerMatrix(TuckerMatrix&&) = default;

  // Copy assignment
  TuckerMatrix& operator=(TuckerMatrix const&) = default;

  // Move assignment
  TuckerMatrix& operator=(TuckerMatrix&&) = default;

  /**
   * \brief
   * Copy constructor
   */

  template <execution_space rhs_space>
  TuckerMatrix(TuckerMatrix<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        max_ranks(rhs.max_ranks),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        cores(::boba::typed_array<Matrix<space, data_t>>(rhs.cores)),
        R_core(Tensor<dimension, space, data_t>(rhs.cores))
  {
  }

  template <execution_space rhs_space>
  TuckerMatrix& operator=(TuckerMatrix<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    max_ranks = rhs.max_ranks;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    cores = ::boba::typed_array<Matrix<space, data_t>>(rhs.cores);
    R_core = rhs.R_core;
    return *this;
  }

  ~TuckerMatrix() = default;

  /**
   * \brief
   * Resizes the Tucker matrix decomposition and reinitializes every core to zero.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   */

  void resize(::boba::Array<index_t, dimension> input_rows, ::boba::Array<index_t, dimension> input_cols)
  {
    checkpoint();
    boba_assert_positive(input_rows, "invalid size");
    boba_assert_positive(input_cols, "invalid size");
    for (size_t d = 0; d < dimension; d++)
    {
      cores[d].resize({input_rows[d], input_cols[d], 1});
      cores[d].fill_with_zeros();
    }
    R_core.resize(filled_array<dimension>(1_z));
    R_core.fill_with_zeros();
    checkpoint();
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
    R_core.rename(m_name + "_R_core");
  }

  /**
   * \brief
   * Construct from Tucker
   */

  template <execution_space rhs_space>
  void read_from_Tucker(Tucker<dimension, rhs_space, data_t> const& rhs, Array<size_t, dimension> in_rows, Array<size_t, dimension> in_cols)
  {
    this->rename(rhs.m_name);
    this->max_ranks = rhs.max_ranks;
    this->svd_tolerance_relative = rhs.svd_tolerance_relative;
    this->svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    this->R_core = rhs.R_core;
    for (size_t d = 0; d < dimension; d++)
    {
      cores[d].resize({in_rows[d], in_cols[d], rhs.cores[d].cols()});
      cores[d].reshape(rhs.cores[d]);
    }
  }

  /**
   * \brief
   * Write to Tucker
   */

  Tucker<dimension, space, data_t> write_to_tucker() const
  {
    auto sizes = core_rows() * core_cols();
    Tucker<dimension, space, data_t> output(sizes);
    output.rename(m_name);
    output.max_ranks = this->max_ranks;
    output.svd_tolerance_relative = this->svd_tolerance_relative;
    output.svd_tolerance_absolute = this->svd_tolerance_absolute;

    output.R_core = this->R_core;
    for (size_t d = 0; d < dimension; d++)
    {
      auto ranks = this->cores[d].sizes(2);
      output.cores[d].resize({sizes[d], ranks});
      output.cores[d].reshape(this->cores[d]);
    }

    return output;
  }

  // -------------------------------------------------------------------------------------
  // Printing
  // -------------------------------------------------------------------------------------

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    R_core.print();
    for (size_t d = 0; d < dimension; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      std::cout << write_indent(indent + 2) << cores[d].name() << std::endl
                << write_indent(indent + 3) << "ranks = " << cores[d].sizes(1)
                << std::endl
                << write_indent(indent + 3) << "size = " << cores[d].sizes(0) << std::endl;
      cores[d].print(indent + 1);
    }
  }

  /**
   * \brief
   * The Tucker ranks (extents of the R_core)
   */

  Array<size_t, dimension> ranks() const
  {
    return R_core.sizes();
  }

  /**
   * \brief
   * returns a string describing the Tucker ranks in the format `(r0, r1, ...)`
   */

  std::string ranks_string() const
  {
    return (" ( " + make_delimited_string(ranks()) + " ) ");
  }

  /**
   * \brief
   * Write the Tucker to file in a way consistent with Tensor::write_to_file
   */

  void write_to_file(std::string_view filename = "") const
  {
    std::ofstream file;
    std::string print_filename = "";

    if (filename.empty())
    {
      print_filename = m_name;
    }
    else
    {
      print_filename = std::string(filename);
    }
    for (size_t d = 0; d < dimension; d++)
    {
      std::string print_core = print_filename + "_core_" + std::to_string(d);
      boba::write_to_file(cores[d], print_core);
    }
    boba::write_to_file(R_core, print_filename + "_R_core");
  }

  /**
   * \brief
   * Read from a file generated from write_to_file
   */

  void read_from_file(std::string_view filename = "")
  {
    std::ofstream file;
    std::string print_filename = "";

    if (filename.empty())
    {
      print_filename = m_name;
    }
    else
    {
      print_filename = std::string(filename);
    }
    for (size_t d = 0; d < dimension; d++)
    {
      std::string print_core = print_filename + "_core_" + std::to_string(d);
      boba::read_from_file(cores[d], print_core);
    }
    boba::write_to_file(R_core, print_filename + "_R_core");
  }

  // -------------------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------------------

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

  size_t get_ranks(size_t i) const
  {
    boba_always_assert_lt(i, dimension, "invalid core number");
    return this->cores[i].sizes(2);
  }

  size_t get_number_elements(size_t d) const
  {
    return this->cores[d].size();
  }

  size_t get_number_elements() const
  {
    size_t sum_elements = R_core.size();
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += get_number_elements(d);
    }
    return sum_elements;
  }

  double get_full_size() const
  {
    // returns a double since full_sizes can be huge for high dimensions
    double full_size = 1.0;
    for (size_t d = 0; d < dimension; d++)
    {
      full_size *= core_rows(d) * core_cols(d);
    }
    return full_size;
  }

  /**
   * \brief Returns the ratio of the full size over the compressed size, truncated to two digits
   */

  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  Array<size_t, dimension> core_rows() const noexcept
  {
    ::boba::Array<size_t, dimension> this_sizes{0};
    for (size_t d = 0; d < dimension; d++)
    {
      this_sizes[d] = core_rows(d);
    }
    return this_sizes;
  }

  Array<size_t, dimension> core_cols() const noexcept
  {
    ::boba::Array<size_t, dimension> this_sizes{0};
    for (size_t d = 0; d < dimension; d++)
    {
      this_sizes[d] = core_cols(d);
    }
    return this_sizes;
  }

  size_t core_rows(size_t d) const noexcept
  {
    return this->cores[d].sizes(0);
  }

  size_t core_cols(size_t d) const noexcept
  {
    return this->cores[d].sizes(1);
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

  // -------------------------------------------------------------------------------------
  // Section: Write
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Sets this into a rank-one TuckerMatrix filled with x, equivalent to a tensor filled with x
   */

  void fill_with(data_t x)
  {
    BOBA_CALI_MARK

    R_core.resize(filled_array<dimension>(1_z));
    R_core.fill_with(x);

    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].resize({this->core_rows(d), this->core_cols(d), 1});
      this->cores[d].fill_with(1.0);
    }
  }

  /**
   * \brief
   * Sets this into a rank-one TuckerMatrix filled with zeros
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK

    this->fill_with(0.0);
  }

  // -------------------------------------------------------------------------------------
  // Section: Unroll
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Re-creates the full tensor corresponding to this decomposition
   */

  Matrix<space, data_t> decompress() const
  {
    checkpoint();
    ::boba::Array<size_t, dimension> sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      sizes[d] = core_rows(d) * core_cols(d);
    }
    checkpoint();
    ::boba::Tucker<dimension, space, data_t> Tucker_of_matrices(sizes);
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      auto ranks = this->get_ranks(d);
      Tucker_of_matrices.cores[d].resize({sizes[d], ranks});
      Tucker_of_matrices.cores[d].reshape(this->cores[d]);
    }

    Tucker_of_matrices.R_core = this->R_core;

    checkpoint();
    auto Tucker_of_matrices_decompressed = Tucker_of_matrices.decompress();
    checkpoint();

    ::boba::Array<size_t, 2 * dimension> rows_and_cols;
    for (size_t d = 0; d < dimension; d++)
    {
      rows_and_cols[2 * d + 0] = core_rows(d);
      rows_and_cols[2 * d + 1] = core_cols(d);
    }
    checkpoint();
    ::boba::Tensor<2 * dimension, space, data_t> matrix_row_col_tensor(rows_and_cols);
    matrix_row_col_tensor.reshape(Tucker_of_matrices_decompressed);
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
    boba::Matrix<space, data_t> full_matrix({product(core_rows()), product(core_cols())});
    full_matrix.reshape(matrix_row_col_tensor);
    checkpoint();
    return full_matrix;
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Addition of two TuckerMatrix
   */

  void TuckerMatrix_add(TuckerMatrix const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    auto this_rows = this->core_rows();
    auto this_cols = this->core_cols();
    auto this_Tucker = this->write_to_tucker();
    auto input_Tucker = input.write_to_tucker();
    checkpoint();
    auto new_Tucker = this_Tucker + input_Tucker;
    checkpoint();
    this->read_from_Tucker(new_Tucker, this_rows, this_cols);
    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Matvec
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tucker_matrix_vector_multiply
   * Each core performs a contraction according to
   * output_d{i, [ri rk], [rj rl]} = sum_k TuckerMatrix_d{i, k, ri, rj}*Tucker_d{k, rk, rl}
   * where "_d", refers to the d'th core.
   */

  Tucker<dimension, space, data_t> Tucker_matrix_vector_multiply(
    Tucker<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    Tucker<dimension, space, data_t> output(input.sizes());
    output.svd_tolerance_relative = input.svd_tolerance_relative;
    output.svd_tolerance_absolute = input.svd_tolerance_absolute;

    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_equal(
        input.cores[d].sizes(0),
        this->cores[d].sizes(1),
        "invalid size and cols for matvec");

      size_t new_rows = this->core_rows()[d];
      size_t new_ranks = this->get_ranks(d) * input.get_ranks(d);
      if (new_ranks == 0)
      {
        output.fill_with_zeros();
        return output;
      }

      auto temporary = tensor_contraction_single_index(
        {"new_row", "col", "r1"}, this->cores[d], {"col", "r2"}, input.cores[d], {"new_row", "r1", "r2"});

      checkpoint();
      // temporary(r, [i, n]) = temporary(r, i, n)
      output.cores[d].resize({new_rows, new_ranks});
      output.cores[d].reshape(temporary);
    }

    output.R_core = tensor_product(this->R_core, input.R_core);

    checkpoint();
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  TuckerMatrix& operator+=(TuckerMatrix const& rhs)
  {
    BOBA_CALI_MARK
    this->TuckerMatrix_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TuckerMatrix operator+(TuckerMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TuckerMatrix output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  TuckerMatrix& operator-=(TuckerMatrix const& rhs)
  {
    BOBA_CALI_MARK
    TuckerMatrix rhs_minus{rhs};
    rhs_minus *= static_cast<real_data_t>(-1);
    this->TuckerMatrix_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TuckerMatrix operator-(TuckerMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TuckerMatrix output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * negation operator
   */

  TuckerMatrix operator-()
  {
    BOBA_CALI_MARK
    TuckerMatrix output{*this};
    output *= static_cast<real_data_t>(-1);
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  TuckerMatrix& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    multiply_scalar(scalar);
    return *this;
  }

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  TuckerMatrix& operator*=(_real_data_t const scalar)
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

  TuckerMatrix operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    TuckerMatrix output{*this};
    output *= scalar;
    return output;
  }

  /**
   * \brief
   * output = this * input
   */

  Tucker<dimension, space, data_t> operator*(Tucker<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    return this->Tucker_matrix_vector_multiply(input);
  }

  /**
   * \brief
   * output = this * input
   */

  TuckerMatrix<dimension, space, data_t> operator*(TuckerMatrix<dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    TuckerMatrix<dimension, space, data_t> output{*this};
    output *= input;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Compress an arbitrarily sized matrix into this TuckerMatrix.
   * Note the sizes must be consistent.
   */

  void compress(boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    if (dimension == 1)
    {
      cores[0].reshape(input);
      R_core.fill_with(1.0);
      return;
    }

    auto rows_and_cols = ::boba::concatenate(core_rows(), core_cols());

    checkpoint();
    ::boba::Tensor<2 * dimension, space, data_t> matrix_row_col_tensor(rows_and_cols);
    matrix_row_col_tensor.reshape(input);

    checkpoint();
    ::boba::Array<size_t, 2 * dimension> permutations;
    for (size_t d = 0; d < dimension; d++)
    {
      permutations[2 * d + 0] = d;             // dimension - 1 - d;
      permutations[2 * d + 1] = dimension + d; // 2*dimension - 1 - d;
    }

    checkpoint();
    permute(matrix_row_col_tensor, permutations);

    checkpoint();
    ::boba::Array<size_t, dimension> sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      sizes[d] = core_rows(d) * core_cols(d);
    }

    checkpoint();
    ::boba::Tensor<dimension, space, data_t> matrix_tensor(sizes);
    matrix_tensor.reshape(matrix_row_col_tensor);

    checkpoint();
    Tucker<dimension, space, data_t> Tucker_of_matrices(core_rows() * core_cols());
    Tucker_of_matrices.svd_tolerance_absolute = svd_tolerance_absolute;
    Tucker_of_matrices.svd_tolerance_relative = svd_tolerance_relative;
    Tucker_of_matrices.max_ranks = max_ranks;
    Tucker_of_matrices.compress(matrix_tensor);

    this->read_from_Tucker(Tucker_of_matrices, core_rows(), core_cols());
  }

  /**
   * \brief
   * Uses svd_tolerance, svd_max_kept_values to perform the tensor-round operation on this.
   * This process is broken up in the right-moving QR phase and the left-moving SVD phase.
   */

  void round()
  {
    BOBA_CALI_MARK
    checkpoint();
    auto this_rows = this->core_rows();
    auto this_cols = this->core_cols();
    auto this_Tucker = this->write_to_tucker();
    checkpoint();
    this_Tucker.round();
    this->read_from_Tucker(this_Tucker, this_rows, this_cols);
    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Multiply scalar
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Multiplies the TuckerMatrix by x, such that A = x*A
   */

  void multiply_scalar(data_t x)
  {
    checkpoint();
    this->cores[0] *= x;
  }

  /**
   * \brief
   * Multiplies each core of the TuckerMatrix by its own scalar, such that A = product(scalars)*A
   */

  void multiply_scalars(const ::boba::Array<data_t, dimension> scalars)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d] *= scalars[d];
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Reduction
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Inner product of this and input TuckerMatrix
   */

  data_t inner_product(const TuckerMatrix<dimension, space, data_t>& input) const
  {
    BOBA_CALI_MARK
    checkpoint();
    auto this_Tucker = this->write_to_tucker();
    auto input_Tucker = input.write_to_tucker();
    auto result = this_Tucker.inner_product(input_Tucker);
    return result;
  }
};

template <size_t dimension, ::boba::execution_space space, typename _data_t>
TuckerMatrix<dimension, space, _data_t> operator*(_data_t scalar, TuckerMatrix<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  TuckerMatrix<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

} // namespace boba
