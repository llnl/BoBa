// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace boba
{

/**
 * \brief Matrix specialization of a rank-2 tensor.
 *
 * Rows and columns correspond to `sizes()[0]` and `sizes()[1]` respectively.
 */

// ------------------------------------
template <execution_space space, typename _data_t>
struct Matrix : Tensor<2, space, _data_t>
{
  using base = Tensor<2, space, _data_t>;

  using base::sizes;
  using typename base::data_t;
  using typename base::index_array;
  using typename base::real_data_t;

  static constexpr std::string_view object_type_name = "Matrix";

  /**
   * \brief Constructs a Matrix with the requested shape and optional name.
   * \param new_sizes Matrix extents in row-major order.
   * \param name Object name used in diagnostics and I/O helpers.
   */
  explicit Matrix(
    index_array new_sizes = ::boba::filled_array<2>(static_cast<index_t>(0)),
    std::string_view name = object_type_name)
      : base(std::move(new_sizes), name)
  {
  }

  /**
   * \brief copy constructor
   */
  Matrix(Matrix const&) = default;

  /**
   * \brief move constructor
   */
  Matrix(Matrix&&) = default;

  /**
   * \brief copy assignment operator
   */
  Matrix& operator=(Matrix const&) = default;

  /**
   * \brief move assignment operator
   */
  Matrix& operator=(Matrix&&) = default;

  // Conversion constructor that works with boba::Tensor<2,...>
  // It creates a temporary Matrix that will either be copied (lvalue)
  // or moved (rvalue) depending on the calling code
  /**
   * \brief Constructs a Matrix from a rank-2 tensor.
   * \param input Rank-2 tensor used to initialize this Matrix.
   */
  explicit Matrix(base input) noexcept
      : base(std::move(input))
  {
  }

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Matrix to copy from.
   */
  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  Matrix(Matrix<rhs_space, data_t> const& rhs)
      : base(rhs)
  {
  }

  // -------------------------------------------------------------------------------------
  // Operators
  // -------------------------------------------------------------------------------------

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Matrix to copy from.
   * \return This Matrix after assignment.
   */
  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  Matrix& operator=(Matrix<rhs_space, data_t> const& rhs)
  {
    static_cast<base&>(*this) = static_cast<Tensor<2, rhs_space, data_t> const&>(rhs);
    return *this;
  }

  /**
   * \brief Adds another Matrix elementwise.
   * \param rhs Matrix to add.
   * \return This Matrix after the addition.
   */
  Matrix& operator+=(Matrix const& rhs)
  {
    static_cast<base&>(*this) += static_cast<base const&>(rhs);
    return *this;
  }

  /**
   * \brief Returns the elementwise sum of two matrices.
   * \param rhs Right-hand operand.
   * \return The elementwise sum.
   */
  [[nodiscard]]
  Matrix operator+(Matrix const& rhs)
  {
    BOBA_CALI_OBJECT_MARK
    Matrix output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief Subtracts another Matrix elementwise.
   * \param rhs Matrix to subtract.
   * \return This Matrix after the subtraction.
   */
  Matrix& operator-=(Matrix const& rhs)
  {
    static_cast<base&>(*this) -= static_cast<base const&>(rhs);
    return *this;
  }

  /**
   * \brief Returns the elementwise difference of two matrices.
   * \param rhs Right-hand operand.
   * \return The elementwise difference.
   */
  [[nodiscard]]
  Matrix operator-(Matrix const& rhs)
  {
    Matrix output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief Returns the additive inverse of this Matrix.
   * \return Matrix with every element negated.
   */
  [[nodiscard]]
  Matrix operator-()
  {
    Matrix output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief Multiplies every entry by a scalar.
   * \param scalar Scalar multiplier.
   * \return This Matrix after scaling.
   */
  Matrix& operator*=(data_t const scalar)
  {
    this->multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief Returns a scaled copy of this Matrix.
   * \param scalar Scalar multiplier.
   * \return The scaled Matrix.
   */
  [[nodiscard]]
  Matrix operator*(data_t const scalar) const
  {
    Matrix output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Getters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Returns the number of rows.
   * \return Row count.
   */
  [[nodiscard]]
  index_t rows() const noexcept
  {
    return this->sizes(0);
  }

  /**
   * \brief Returns the number of columns.
   * \return Column count.
   */
  [[nodiscard]]
  index_t cols() const noexcept
  {
    return this->sizes(1);
  }

  // -------------------------------------------------------------------------------------
  // Section: Read/write
  // -------------------------------------------------------------------------------------

  /**
   * \brief Constructs the transpose of this Matrix.
   * \return Transposed copy of this Matrix.
   */
  [[nodiscard]]
  Matrix transpose() const
  {
    checkpoint_objects();
    BOBA_CALI_OBJECT_MARK
    Matrix output{*this};
    permute({"rows", "cols"}, output, {"cols", "rows"});
    return output;
  }

  /**
   * \brief Constructs the conjugate transpose of this Matrix.
   * \return Conjugate-transposed copy of this Matrix.
   */
  [[nodiscard]]
  Matrix conjugate_transpose() const
  {
    auto output = transpose();
    if constexpr (not(std::is_same_v<data_t, real_data_t>))
    {
      complex_conjugate_in_place(output);
    }
    return output;
  }

  /**
   * \brief Constructs the transpose with explicitly requested output dimensions.
   * \param new_sizes_after_transpose Output Matrix dimensions after transposition.
   * \return Transposed copy resized to \p new_sizes_after_transpose.
   */
  [[nodiscard]]
  Matrix transpose_resize(index_array new_sizes_after_transpose) const
  {
    checkpoint_objects();
    BOBA_CALI_OBJECT_MARK
    Matrix output(new_sizes_after_transpose);

    auto this_view = this->const_view();
    auto output_view = output.view();

    ::boba::loop<space, 2>(output.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto i = ij[0];
      auto j = ij[1];
      output_view({i, j}) = this_view({j, i});
    });

    checkpoint_objects();
    return output;
  }

  /**
   * \brief Constructs the conjugate transpose with explicitly requested output dimensions.
   * \param new_sizes_after_transpose Output Matrix dimensions after transposition.
   * \return Conjugate-transposed copy resized to \p new_sizes_after_transpose.
   */
  [[nodiscard]]
  Matrix conjugate_transpose_resize(index_array new_sizes_after_transpose) const
  {
    auto output = transpose_resize(new_sizes_after_transpose);
    if constexpr (not(std::is_same_v<data_t, real_data_t>))
    {
      complex_conjugate_in_place(output);
    }
    return output;
  }

  /**
   * \brief Rewrites the Matrix as an identity Matrix with the same shape.
   */
  void set_to_identity_matrix()
  {
    BOBA_CALI_OBJECT_MARK
    this->fill_with(::boba::PotentiallyComplex<data_t>::value(0.0));
    fill_diagonal(0_z, ::boba::PotentiallyComplex<data_t>::value(1.0));
  }

  /**
   * \brief Sets all strictly lower-triangular entries to zero.
   */
  void erase_lower_triangular()
  {
    BOBA_CALI_OBJECT_MARK
    auto this_view = this->view();
    ::boba::loop<space, 2>({rows(), cols()},
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto i = ij[0];
      auto j = ij[1];
      if (j < i)
      {
        this_view({i, j}) = ::boba::PotentiallyComplex<data_t>::value(0.0);
      }
    });
  }

  /**
   * \brief Fills one row with a constant value.
   * \param row Row index to fill.
   * \param value Value assigned to every entry in the row.
   */
  void fill_row(index_t row, data_t value)
  {
    BOBA_CALI_OBJECT_MARK

    auto this_view = this->view();
    ::boba::loop<space, 1>({cols()},
                           [=] __boba_host_device__(index_t j)
    {
      this_view({row, j}) = value;
    });
  }

  /**
   * \brief Fills one column with a constant value.
   * \param col Column index to fill.
   * \param value Value assigned to every entry in the column.
   */
  void fill_col(index_t col, data_t value)
  {
    BOBA_CALI_OBJECT_MARK
    auto this_view = this->view();
    ::boba::loop<space, 1>({rows()},
                           [=] __boba_host_device__(index_t i)
    {
      this_view({i, col}) = value;
    });
  }

  /**
   * \brief Fills one diagonal of the Matrix with a constant value.
   * \param diagonal Diagonal offset. Positive values select superdiagonals and negative values select subdiagonals.
   * \param value Value assigned to the selected diagonal.
   *
   * Related to the logic in MATLAB's `spdiags`.
   * https://www.mathworks.com/help/matlab/ref/spdiags.html
   */
  void fill_diagonal(int diagonal, data_t value)
  {
    BOBA_CALI_OBJECT_MARK
    auto this_view = this->view();

    auto abs_diagonal = static_cast<index_t>(::boba::abs(diagonal));

    if (diagonal == 0)
    {
      auto length = ::boba::min(rows(), cols());
      ::boba::detail::loop<space>(0_z, length, [=] __boba_host_device__(index_t i)
      {
        this_view({i, i}) = value;
      });
    }
    else if (diagonal > 0)
    {
      boba_always_assert_lt(abs_diagonal, cols(), "diagonal offset exceeds matrix columns");
      auto length = ::boba::min(rows(), cols() - abs_diagonal);
      ::boba::detail::loop<space>(0_z, length, [=] __boba_host_device__(index_t l)
      {
        auto row = l;
        auto col = abs_diagonal + l;
        this_view({row, col}) = value;
      });
    }
    else /* diagonal < 0 */
    {
      boba_always_assert_lt(abs_diagonal, rows(), "diagonal offset exceeds matrix rows");
      auto length = ::boba::min(cols(), rows() - abs_diagonal);
      ::boba::detail::loop<space>(0_z, length, [=] __boba_host_device__(index_t l)
      {
        auto row = abs_diagonal + l;
        auto col = l;
        this_view({row, col}) = value;
      });
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Linear Algebra
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Overwrite this Matrix with its transpose
   */

  void transpose_in_place()
  {
    BOBA_CALI_OBJECT_MARK

    Matrix temp = this->transpose();
    *this = temp;
  }

  /**
   * \brief
   * Overwrite this Matrix with its conjugate transpose
   */

  void conjugate_transpose_in_place()
  {
    transpose_in_place();
    if constexpr (not(std::is_same_v<data_t, real_data_t>))
    {
      complex_conjugate_in_place(*this);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Norm
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Computes the Matrix 1-norm of this Matrix, ||A||_1.
   */

  data_t matrix_one_norm()
  {
    BOBA_CALI_OBJECT_MARK

    auto rows = this->rows();
    auto cols = this->cols();
    auto this_view = this->view();

    if (this->size() == 0)
    {
      return data_t{0};
    }

    ::boba::Vector<space, data_t> column_sums({cols});
    column_sums.fill_with_zeros();
    auto column_sums_view = column_sums.atomic_view();

    ::boba::loop<space, 2>({rows, cols},
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto j = ij[1];
      column_sums_view(j) += ::boba::abs(this_view(ij));
    });

    return column_sums.max_reduce();
  }

  /**
   * \brief
   * Computes the Matrix infinity-norm of this Matrix, ||A||_inf.
   */

  data_t matrix_inf_norm()
  {
    BOBA_CALI_OBJECT_MARK

    auto rows = this->rows();
    auto cols = this->cols();
    auto this_view = this->view();

    if (this->size() == 0)
    {
      return data_t{0};
    }

    ::boba::Vector<space, data_t> row_sums({rows});
    row_sums.fill_with_zeros();
    auto row_sums_view = row_sums.atomic_view();

    ::boba::loop<space, 2>({rows, cols},
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto i = ij[0];
      row_sums_view(i) += ::boba::abs(this_view(ij));
    });

    return row_sums.max_reduce();
  }

  /**
   * Create a new tensor where you apply the function f(x) = max(x, 0) to each entry
   * This would be useful for making A = B - C, where B,C >0
   */

  Matrix nonnegative_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonnegative_part is disabled for complex data type");
    return this->template unary_transform_copy<Matrix>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(x);
    });
  }

  /**
   * Create a new tensor where you apply the function f(x) = max(-x, 0) to each entry.
   * This would be useful for making A = B - C, where B,C >0
   */

  Matrix nonpositive_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonpositive_part is disabled for complex data type");
    return this->template unary_transform_copy<Matrix>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(-x);
    });
  }

  /**
   * @brief Copy the data from this Matrix spanned by [rows[0], rows[1]) by [cols[0], cols[1]) into the output
   *
   * @param[in] rows The span of rows from which output will be copied
   * @param[in] cols The span of columns from which output will be copied
   * @return Matrix formed from the copied data
   */

  // TODO<impl details> rework in terms of extract subtensor
  // TODO<impl details> create specialization for individual 1d subsets / fibers

  Matrix get_submatrix(Array<index_t, 2> rows, Array<index_t, 2> cols) const
  {
    BOBA_CALI_OBJECT_MARK
    boba_assert_lt(rows[0], rows[1], "Incorrect ordering.");
    boba_assert_lt(cols[0], cols[1], "Incorrect ordering.");

    Matrix output({rows[1] - rows[0], cols[1] - cols[0]});

    auto output_view = output.view();
    auto this_view = this->const_view();

    ::boba::loop<space, 2>(output_view.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto [row, col] = ij;
      output_view(ij) = this_view({rows[0] + row, cols[0] + col});
    });
    return output;
  }

  /**
   * \brief
   * Creates a submatrix corresponding to the subset of rows indexed by the input
   */

  Matrix extract_rows(const Vector<space, index_t>& rows) const
  {
    BOBA_CALI_OBJECT_MARK

    Matrix output({rows.size(), cols()});

    auto output_view = output.view();
    auto this_view = this->const_view();
    auto rows_view = rows.const_view();

    ::boba::loop<space, 2>(output_view.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto row_id = rows_view(ij[0]);
      auto col_id = ij[1];
      output_view(ij) = this_view({row_id, col_id});
    });
    return output;
  }

  /**
   * \brief
   * Creates a submatrix corresponding to the row indexed by the input
   */

  Matrix extract_rows(const size_t row) const
  {
    Vector<space, index_t> selection({1});
    selection.fill_with(row);
    return extract_rows(selection);
  }

  /**
   * \brief
   * Creates a submatrix corresponding to the subset of columns indexed by the input
   */

  Matrix extract_columns(const Vector<space, index_t>& cols) const
  {
    BOBA_CALI_OBJECT_MARK

    Matrix output({rows(), cols.size()});

    auto output_view = output.view();
    auto this_view = this->const_view();
    auto cols_view = cols.const_view();

    ::boba::loop<space, 2>(output_view.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto row_id = ij[0];
      auto col_id = cols_view(ij[1]);
      output_view(ij) = this_view({row_id, col_id});
    });
    return output;
  }

  /**
   * \brief
   * Creates a submatrix corresponding to the column indexed by the input
   */

  Matrix extract_columns(const size_t col) const
  {
    Vector<space, index_t> selection({1});
    selection.fill_with(col);
    return extract_columns(selection);
  }

  /**
   * @brief Copy the data from replacement into the submatrix spanned by [rows[0], rows[1]) by [cols[0], cols[1])
   *
   * @param[in] rows The span of rows into which replacement will be copied
   * @param[in] cols The span of columns into which replacement will be copied
   * @param[in] replacement The Matrix containing the data which will be replacing data in this Matrix
   */

  void replace_submatrix(Array<index_t, 2> rows, Array<index_t, 2> cols, const Matrix& replacement)
  {
    BOBA_CALI_OBJECT_MARK
    boba_assert_equal(rows[1] - rows[0], replacement.rows(), "Incorrect ordering.");
    boba_assert_equal(cols[1] - cols[0], replacement.cols(), "Incorrect ordering.");

    auto replacement_view = replacement.const_view();
    auto this_view = this->view();

    ::boba::loop<space, 2>(replacement_view.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      auto [row, col] = ij;
      this_view({rows[0] + row, cols[0] + col}) = replacement_view(ij);
    });
  }
};

/**
 * \brief
 * If the input is a vecotr, lays the vector along the diagonal of a new Matrix, as in MATLAB's diag() command.
 * If the input is a Matrix, grabs the diagonal of the Matrix, as in MATLAB's diag() command.
 */

template <execution_space space, typename data_t>
::boba::Matrix<space, data_t> diagonalize(::boba::Vector<space, data_t> diagonal)
{
  BOBA_CALI_OBJECT_MARK
  ::boba::Matrix<space, data_t> new_matrix({diagonal.size(), diagonal.size()});
  auto diagonal_view = diagonal.view();
  auto new_matrix_view = new_matrix.view();
  new_matrix.fill_with_zeros();
  ::boba::detail::loop<space>(0, new_matrix.rows(), [=] __boba_host_device__(index_t i)
  {
    new_matrix_view({i, i}) = diagonal_view({i});
  });
  return new_matrix;
}

/**
 * \brief
 * If the input is a vecotr, lays the vector along the diagonal of a new Matrix, as in MATLAB's diag() command.
 * If the input is a Matrix, grabs the diagonal of the Matrix, as in MATLAB's diag() command.
 */

template <execution_space space, typename data_t>
::boba::Vector<space, data_t> diagonalize(::boba::Matrix<space, data_t> input)
{
  BOBA_CALI_OBJECT_MARK
  auto length = boba::min(input.rows(), input.cols());
  ::boba::Vector<space, data_t> new_vector({length});
  auto input_view = input.const_view();
  auto new_vector_view = new_vector.view();
  ::boba::detail::loop<space>(0, new_vector.size(), [=] __boba_host_device__(index_t i)
  {
    new_vector_view({i}) = input_view({i, i});
  });
  return new_vector;
}

/**
 * \brief
 * output = a + b
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator+(const Matrix<space, data_t>& a, const Matrix<space, data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Matrix<space, data_t> output = a;
  output += b;
  return output;
}

/**
 * \brief
 * output = a - b
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator-(const Matrix<space, data_t>& a, const Matrix<space, data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Matrix<space, data_t> output = a;
  output -= b;
  return output;
}

/**
 * \brief
 *  output = scalar * matrix_A
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator*(data_t const scalar, Matrix<space, data_t> matrix_A)
{
  BOBA_CALI_OBJECT_MARK
  return matrix_A * scalar;
}

/**
 * \brief
 * Concatenate the columns of two matrices, as in MATLAB C = [A, B]
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> concatenate_columns(
  const Matrix<space, data_t>& A,
  const Matrix<space, data_t>& B)
{
  BOBA_CALI_OBJECT_MARK

  boba_assert_equal(A.rows(), B.rows(), "Incompatible matrices.");

  auto A_cols = A.cols();
  auto C_cols = A_cols + B.cols();
  Matrix<space, data_t> C({A.rows(), C_cols});

  auto A_view = A.const_view();
  auto B_view = B.const_view();
  auto C_view = C.view();

  ::boba::loop<space, 2>(C_view.sizes(),
                         [=] __boba_host_device__(Array<size_t, 2> ij)
  {
    auto [row, col] = ij;
    bool in_A = col < A_cols;
    C_view({row, col}) = in_A ? A_view({row, col}) : B_view({row, col - A_cols});
  });

  return C;
}

template <execution_space space, typename data_t>
Matrix<space, data_t> concatenate_rows(
  const Matrix<space, data_t>& A,
  const Matrix<space, data_t>& B)
{
  BOBA_CALI_OBJECT_MARK

  boba_assert_equal(A.cols(), B.cols(), "Incompatible matrices.");

  auto A_rows = A.rows();
  auto C_rows = A_rows + B.rows();
  Matrix<space, data_t> C({C_rows, A.cols()});

  auto A_view = A.const_view();
  auto B_view = B.const_view();
  auto C_view = C.view();

  ::boba::loop<space, 2>(C_view.sizes(),
                         [=] __boba_host_device__(Array<size_t, 2> ij)
  {
    auto [row, col] = ij;
    bool in_A = row < A_rows;
    C_view({row, col}) = in_A ? A_view({row, col}) : B_view({row - A_rows, col});
  });

  return C;
}

template <execution_space space, typename data_t>
Vector<space, data_t> concatenate_vectors(
  const Vector<space, data_t>& A,
  const Vector<space, data_t>& B)
{
  BOBA_CALI_OBJECT_MARK

  auto A_size = A.size();
  auto C_size = A_size + B.size();
  Vector<space, data_t> C({C_size});

  auto A_view = A.const_view();
  auto B_view = B.const_view();
  auto C_view = C.view();

  ::boba::loop<space, 1>(C_view.sizes(),
                         [=] __boba_host_device__(Array<size_t, 1> i)
  {
    auto idx = i[0];
    bool in_A = idx < A_size;
    C_view(i) = in_A ? A_view(i) : B_view({idx - A_size});
  });

  return C;
}

} // namespace boba
