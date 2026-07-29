// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_CUDA_LIBS
#include <cublas_v2.h>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_CUDA_LIBS
extern cublasHandle_t cublas_handle;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

#define cublas_assert(a) cublas_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a cuBLAS error and terminates.
 * @param error cuBLAS status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cublas_assert_(
  cublasStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == CUBLAS_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << " returned an error code: " << std::endl;
  std::cout << cublasGetStatusName(error) << ": " << cublasGetStatusString(error) << std::endl;
  exit(1);
}

/**
 * @brief Multiplies two device matrices with cuBLAS.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @tparam C_view_t Output view type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t>
void cublas_gemm(
  A_view_t const& matrix_A,
  B_view_t const& matrix_B,
  C_view_t const& matrix_C)
{
  cublasOperation_t cublas_noop = CUBLAS_OP_N;
  using data_t = typename C_view_t::data_t;
  using real_data_t = real_type_t<data_t>;
  auto const zero = PotentiallyComplex<data_t>::value(static_cast<real_data_t>(0));
  auto const one = PotentiallyComplex<data_t>::value(static_cast<real_data_t>(1));
  auto const m = static_cast<int>(matrix_A.sizes(0));
  auto const n = static_cast<int>(matrix_B.sizes(1));
  auto const k = static_cast<int>(matrix_B.sizes(0));
  auto const lda = static_cast<int>(matrix_A.sizes(0));
  auto const ldb = static_cast<int>(matrix_B.sizes(0));
  auto const ldc = static_cast<int>(matrix_C.sizes(0));

  // https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
  // C = alpha*op(A)*op(B) + beta*C
  // A is m x k
  // B is k x n
  // C is m x n
  if constexpr (std::is_same_v<data_t, float>)
  {
    cublas_assert(cublasSgemm(
      cublas_handle,
      cublas_noop,
      cublas_noop,
      m,
      n,
      k,
      &one,
      matrix_A.data(),
      lda,
      matrix_B.data(),
      ldb,
      &zero,
      matrix_C.data(),
      ldc));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    cublas_assert(cublasDgemm(
      cublas_handle,
      cublas_noop,
      cublas_noop,
      m,
      n,
      k,
      &one,
      matrix_A.data(),
      lda,
      matrix_B.data(),
      ldb,
      &zero,
      matrix_C.data(),
      ldc));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    cublas_assert(cublasCgemm(
      cublas_handle,
      cublas_noop,
      cublas_noop,
      m,
      n,
      k,
      &one,
      matrix_A.data(),
      lda,
      matrix_B.data(),
      ldb,
      &zero,
      matrix_C.data(),
      ldc));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cublas_assert(cublasZgemm(
      cublas_handle,
      cublas_noop,
      cublas_noop,
      m,
      n,
      k,
      &one,
      matrix_A.data(),
      lda,
      matrix_B.data(),
      ldb,
      &zero,
      matrix_C.data(),
      ldc));
  }
  else
  {
    boba_error("Build error! Unsupported data type for cublas_gemm");
  }
}

/**
 * @brief Computes a device dot product with cuBLAS.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @param vector_A Left operand vector.
 * @param vector_B Right operand vector.
 * @return Dot product value.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto cublas_dot(
  A_view_t const& vector_A,
  B_view_t const& vector_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  using data_t = std::remove_const_t<typename A_view_t::data_t>;

  auto size = static_cast<int>(vector_A.size());
  constexpr int stride = 1;
  data_t result = data_t{};

  if constexpr (std::is_same_v<data_t, float>)
  {
    cublas_assert(cublasSdot(
      cublas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    cublas_assert(cublasDdot(
      cublas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    cublas_assert(cublasCdotu(
      cublas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cublas_assert(cublasZdotu(
      cublas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else
  {
    boba_error("Build error! Unsupported data type for cublas_dot");
  }

  return result;
}

#else

/**
 * @brief Reports that cuBLAS GEMM support is unavailable.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @tparam C_view_t Output view type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t>
void cublas_gemm(
  A_view_t const& matrix_A,
  B_view_t const& matrix_B,
  C_view_t const& matrix_C)
{
  detail::ignore(matrix_A);
  detail::ignore(matrix_B);
  detail::ignore(matrix_C);
  boba_error("Build error! You are trying to run cublas_gemm BOBA_CUDA_LIBS");
}

/**
 * @brief Reports that cuBLAS dot-product support is unavailable.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @param vector_A Left operand vector.
 * @param vector_B Right operand vector.
 * @return This function does not return successfully.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto cublas_dot(
  A_view_t const& vector_A,
  B_view_t const& vector_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  detail::ignore(vector_A);
  detail::ignore(vector_B);
  boba_error("Build error! You are trying to run cublas_dot without BOBA_CUDA_LIBS");
}

#endif

} // namespace detail
} // namespace boba
