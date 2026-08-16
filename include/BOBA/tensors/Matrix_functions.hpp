// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef BOBA_EIGEN_TENSOR
#include "unsupported/Eigen/CXX11/Tensor"
#endif

namespace boba
{

/**
 * \brief Solves a linear system with matrix right-hand sides.
 * \param A_matrix Coefficient matrix in `A * X = rhs`.
 * \param rhs Matrix right-hand side.
 * \return Solution matrix `X`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> backsolve(
  Matrix<space, data_t>& A_matrix,
  Matrix<space, data_t>& rhs)
{
  return backsolve(
    static_cast<Matrix<space, data_t> const&>(A_matrix),
    static_cast<Matrix<space, data_t> const&>(rhs));
}

/**
 * \brief Solves a linear system with matrix right-hand sides.
 * \param A_matrix Coefficient matrix in `A * X = rhs`.
 * \param rhs Matrix right-hand side.
 * \return Solution matrix `X`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> backsolve(
  Matrix<space, data_t> const& A_matrix,
  Matrix<space, data_t> const& rhs)
{
  // Solve Ax = rhs
  // output = x
  BOBA_CALI_MARK
  checkpoint_objects();

  boba_always_assert(rhs.rows() == A_matrix.rows(), "incompatible sizes");
  Matrix<space, data_t> output({A_matrix.cols(), rhs.cols()});

  if constexpr (space == execution_space::CPU)
  {
    if (A_matrix.rows() == A_matrix.cols())
    {
      ::boba::detail::ls_solve_lu_eigen(A_matrix, rhs, output);
    }
    else if (A_matrix.rows() > A_matrix.cols())
    {
      ::boba::detail::ls_solve_qr_eigen(A_matrix, rhs, output);
    }
    else
    {
      ::boba::detail::ls_solve_cod_eigen(A_matrix, rhs, output);
    }
  }
  else if constexpr (space == execution_space::CUDA)
  {
    if (A_matrix.rows() < A_matrix.cols())
    {
      Matrix<host_space, data_t> host_A(A_matrix);
      Matrix<host_space, data_t> host_rhs(rhs);
      auto host_output = backsolve(host_A, host_rhs);
      output = host_output;
    }
    else
    {
      auto work = rhs;
      if (A_matrix.rows() == A_matrix.cols())
      {
        ::boba::detail::ls_solve_lu_cusolver(A_matrix, work);
      }
      else
      {
        ::boba::detail::ls_solve_qr_cusolver(A_matrix, work);
        work.resize(output.sizes());
      }
      output = work;
    }
  }
  else if constexpr (space == execution_space::HIP)
  {
    if (A_matrix.rows() < A_matrix.cols())
    {
      Matrix<host_space, data_t> host_A(A_matrix);
      Matrix<host_space, data_t> host_rhs(rhs);
      auto host_output = backsolve(host_A, host_rhs);
      output = host_output;
    }
    else
    {
      auto work = rhs;
      if (A_matrix.rows() == A_matrix.cols())
      {
        ::boba::detail::ls_solve_lu_hipsolver(A_matrix, work);
      }
      else
      {
        ::boba::detail::ls_solve_qr_hipsolver(A_matrix, work);
        work.resize(output.sizes());
      }
      output = work;
    }
  }
  else
  {
    boba_error("Not yet implemented this execution space.");
  }
  checkpoint_objects();
  return output;
}

/**
 * \brief Solves a linear system with a vector right-hand side.
 * \param A_matrix Coefficient matrix in `A * x = rhs`.
 * \param rhs Vector right-hand side.
 * \return Solution vector `x`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> backsolve(
  Matrix<space, data_t>& A_matrix,
  Vector<space, data_t>& rhs)
{
  auto rhs_matrix = reshape_to_matrix(rhs, {rhs.size(), 1});
  auto output_matrix = backsolve(A_matrix, rhs_matrix);
  auto output = flatten(output_matrix);
  return output;
}

/**
 * \brief Solves a linear system with a vector right-hand side.
 * \param A_matrix Coefficient matrix in `A * x = rhs`.
 * \param rhs Vector right-hand side.
 * \return Solution vector `x`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> backsolve(
  Matrix<space, data_t> const& A_matrix,
  Vector<space, data_t> const& rhs)
{
  auto rhs_matrix = reshape_to_matrix(rhs, {rhs.size(), 1});
  auto output_matrix = backsolve(A_matrix, rhs_matrix);
  auto output = flatten(output_matrix);
  return output;
}

/**
 * \brief Solves the right-division system `rhs / A_matrix`.
 * \param A_matrix Right-hand coefficient matrix.
 * \param rhs Left-hand matrix operand.
 * \return Matrix equivalent to `rhs * A_matrix^{-1}` when defined.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> right_backsolve(
  Matrix<space, data_t> const& A_matrix,
  Matrix<space, data_t> const& rhs)
{
  auto rhsT = rhs.transpose();
  auto thisT = A_matrix.transpose();
  return backsolve(thisT, rhsT).transpose();
}

/**
 * \brief Computes a matrix inverse on supported host paths.
 * \param input Square matrix to invert.
 * \return Inverse of \p input.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> compute_inverse(const Matrix<space, data_t>& input)
{
  BOBA_CALI_MARK
  checkpoint();
  // Assume square system
  boba_always_assert(input.rows() == input.cols(), "matrix is nonsquare!");
  Matrix<space, data_t> inverse(input.sizes());

  if constexpr (((space == execution_space::CPU) or not(is_device_context())) and ::boba::boba_eigen_enabled())
  {
    auto this_map = ::boba::get_eigen_map(input);
    auto inverse_map = ::boba::get_eigen_map(inverse);
    // This function has compiler issues if compiled in device code
    inverse_map = this_map.inverse();
  }
  else
  {
    boba_error("Matrix inverse is only supported with eigen builds on the host.");
  }
  checkpoint();
  return inverse;
}

/**
 * \brief Computes a matrix-matrix product.
 * \param matrix_A Left-hand matrix operand.
 * \param matrix_B Right-hand matrix operand.
 * \return Product `matrix_A * matrix_B`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Matrix<space, data_t> general_matrix_matrix_product(
  Matrix<space, data_t> const& matrix_A,
  Matrix<space, data_t> const& matrix_B)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();
  Matrix<space, data_t> matrix_C({matrix_A.rows(), matrix_B.cols()});

  if (matrix_C.size() == 0)
  {
    return matrix_C;
  }
  if ((matrix_A.size() == 0) or (matrix_B.size() == 0))
  {
    matrix_C.fill_with_zeros();
    return matrix_C;
  }

  checkpoint_objects();
  BOBA_CALI_OBJECT_BEGIN("matrix_matrix_product");
  if constexpr (space == execution_space::CPU)
  {
    ::boba::detail::eigen_gemm(matrix_A, matrix_B, matrix_C);
  }
  else if constexpr (space == execution_space::CUDA)
  {
    ::boba::detail::cublas_gemm(matrix_A.view(), matrix_B.view(), matrix_C.view());
  }
  else if constexpr (space == execution_space::HIP)
  {
    ::boba::detail::hipblas_gemm(matrix_A.view(), matrix_B.view(), matrix_C.view());
  }
  else
  {
    boba_error("Unknown execution space!");
  }
  BOBA_CALI_OBJECT_END("matrix_matrix_product");
  checkpoint_objects();
  return matrix_C;
}

/**
 * \brief Computes a matrix-vector product.
 * \param matrix_A Left-hand matrix operand.
 * \param vector_B Right-hand vector operand.
 * \return Product `matrix_A * vector_B`.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Vector<space, data_t> general_matrix_vector_product(
  Matrix<space, data_t> const& matrix_A,
  Vector<space, data_t> const& vector_B)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  Vector<space, data_t> vector_C({matrix_A.rows()});

  if (vector_C.size() == 0)
  {
    return vector_C;
  }
  if ((matrix_A.size() == 0) or (vector_B.size() == 0))
  {
    vector_C.fill_with_zeros();
    return vector_C;
  }
  if ((matrix_A.size() == 1) and (vector_B.size() == 1))
  {
    auto a = matrix_A.sum_reduce();
    auto b = vector_B.sum_reduce();
    vector_C.fill_with(a * b);
    return vector_C;
  }

  checkpoint_objects();
  BOBA_CALI_OBJECT_BEGIN("matrix_matrix_product");
  if constexpr (space == execution_space::CPU)
  {
    ::boba::detail::eigen_gemm(matrix_A, vector_B, vector_C);
  }
  else if constexpr (space == execution_space::CUDA)
  {
    ::boba::detail::cublas_gemm(matrix_A.view(), vector_B.view(), vector_C.view());
  }
  else if constexpr (space == execution_space::HIP)
  {
    ::boba::detail::hipblas_gemm(matrix_A.view(), vector_B.view(), vector_C.view());
  }
  else
  {
    boba_error("Unknown execution space!");
  }
  BOBA_CALI_OBJECT_END("matrix_matrix_product");
  checkpoint_objects();
  return vector_C;
}

/**
 * @brief Right multiply a vector by a matrix
 *
 * @param[in] vector_A vector
 * @param[in] matrix_B matrix
 * @return vector = (vector_A^T * matrix_B)^T
 */

template <execution_space space, typename data_t>
Vector<space, data_t> general_vector_matrix_product(
  Vector<space, data_t> const& vector_A,
  Matrix<space, data_t> const& matrix_B)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  Matrix<space, data_t> vector_C_transpose({1, matrix_B.cols()});

  if (vector_C_transpose.size() == 0)
  {
    return flatten(vector_C_transpose);
  }
  if ((vector_A.size() == 0) or (matrix_B.size() == 0))
  {
    vector_C_transpose.fill_with_zeros();
    return flatten(vector_C_transpose);
  }

  auto vec_transpose = reshape_to_matrix(vector_A, {1, vector_A.size()});

  BOBA_CALI_OBJECT_BEGIN("matrix_matrix_product");
  if constexpr (space == execution_space::CPU)
  {
    ::boba::detail::eigen_gemm(vec_transpose, matrix_B, vector_C_transpose);
  }
  else if constexpr (space == execution_space::CUDA)
  {
    ::boba::detail::cublas_gemm(vec_transpose.view(), matrix_B.view(), vector_C_transpose.view());
  }
  else if constexpr (space == execution_space::HIP)
  {
    ::boba::detail::hipblas_gemm(vec_transpose.view(), matrix_B.view(), vector_C_transpose.view());
  }
  else
  {
    boba_error("Unknown execution space!");
  }
  BOBA_CALI_OBJECT_END("matrix_matrix_product");

  auto vector_C = flatten(vector_C_transpose);
  return vector_C;
}

/**
 * \brief
 * matrix_C = matrix_A*matrix_B
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator*(
  Matrix<space, data_t> const& matrix_A,
  Matrix<space, data_t> const& matrix_B)
{
  return general_matrix_matrix_product(matrix_A, matrix_B);
}

/**
 * \brief
 * output = matrix_A*vector_B
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(
  Matrix<space, data_t> const& matrix_A,
  Vector<space, data_t> const& vector_B)
{
  return general_matrix_vector_product(matrix_A, vector_B);
}

/**
 * \brief
 * output = vector_A*matrix_B
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(
  Vector<space, data_t> const& vector_A,
  Matrix<space, data_t> const& matrix_B)
{
  return general_vector_matrix_product(vector_A, matrix_B);
}

} // namespace boba
