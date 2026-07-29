// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_HIP_LIBS
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
#endif

// AMD Documentation
// https://docs.amd.com/bundle/Welcome-to-hipBLAS-s-documentation----hipBLAS-documentation/page/usermanual.html

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_HIP_LIBS

extern hipblasHandle_t hipblas_handle;

extern rocblas_handle _rocblas_handle;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

#define hipblas_assert(a) hipblas_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipBLAS error and terminates.
 * @param error hipBLAS status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hipblas_assert_(
  hipblasStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == HIPBLAS_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  exit(1);
}

#define rocblas_assert(a) rocblas_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a rocBLAS error and terminates.
 * @param error rocBLAS status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void rocblas_assert_(
  rocblas_status error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == rocblas_status_success)
  {
    return;
  }
  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  std::cout << rocblas_status_to_string(error) << std::endl;
  exit(1);
}

#define rocblas_assert_info(a, b, c) rocblas_assert_info_(a, #a, b, c, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a rocBLAS error with an auxiliary info flag and terminates.
 * @param error rocBLAS status code to check.
 * @param call Source expression that produced the error.
 * @param info Device pointer to the backend info flag.
 * @param info_meaning Description of the info flag semantics.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void rocblas_assert_info_(
  rocblas_status error,
  const std::string& call,
  int* info,
  const std::string& info_meaning,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  hip_syncronize();

  int host_info = 0;
  hip_memcpy(&host_info, info, 1);

  // TODO<debug> handle info
  if (error == rocblas_status_success) // and (host_info == 0))
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  std::cout << "Info flag: " << host_info << std::endl;
  std::cout << "Info meaning: " << info_meaning << std::endl;

  std::cout << rocblas_status_to_string(error) << std::endl;
  exit(1);
}

/**
 * @brief Multiplies two device matrices with hipBLAS.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @tparam C_view_t Output view type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t>
void hipblas_gemm(
  A_view_t const& matrix_A,
  B_view_t const& matrix_B,
  C_view_t const& matrix_C)
{
  hipblasOperation_t hipblas_noop = HIPBLAS_OP_N;
  using data_t = C_view_t::data_t;
  using real_data_t = real_type_t<data_t>;
  auto const zero = PotentiallyComplex<data_t>::value(static_cast<real_data_t>(0));
  auto const one = PotentiallyComplex<data_t>::value(static_cast<real_data_t>(1));
  auto const m = static_cast<int>(matrix_A.sizes(0));
  auto const n = static_cast<int>(matrix_B.sizes(1));
  auto const k = static_cast<int>(matrix_B.sizes(0));
  auto const lda = static_cast<int>(matrix_A.sizes(0));
  auto const ldb = static_cast<int>(matrix_B.sizes(0));
  auto const ldc = static_cast<int>(matrix_C.sizes(0));

  // https://docs.amd.com/project/hipBLAS/hipblas/main/api-reference.html
  // C = alpha*op(A)*op(B) + beta*C
  // A is m x k
  // B is k x n
  // C is m x n
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipblas_assert(hipblasSgemm(
      hipblas_handle,
      hipblas_noop,
      hipblas_noop,
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
    hipblas_assert(hipblasDgemm(
      hipblas_handle,
      hipblas_noop,
      hipblas_noop,
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
    hipblas_assert(hipblasCgemm(
      hipblas_handle,
      hipblas_noop,
      hipblas_noop,
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
    hipblas_assert(hipblasZgemm(
      hipblas_handle,
      hipblas_noop,
      hipblas_noop,
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
    boba_error("Build error! Unsupported data type for hipblas_gemm");
  }
}

/**
 * @brief Computes a device dot product with hipBLAS.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @param vector_A Left operand vector.
 * @param vector_B Right operand vector.
 * @return Dot product value.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto hipblas_dot(
  A_view_t const& vector_A,
  B_view_t const& vector_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  using data_t = std::remove_const_t<typename A_view_t::data_t>;

  auto size = static_cast<int>(vector_A.size());
  constexpr int stride = 1;
  data_t result = data_t{};

  if constexpr (std::is_same_v<data_t, float>)
  {
    hipblas_assert(hipblasSdot(
      hipblas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipblas_assert(hipblasDdot(
      hipblas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipblas_assert(hipblasCdotu(
      hipblas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipblas_assert(hipblasZdotu(
      hipblas_handle,
      size,
      vector_A.const_data(),
      stride,
      vector_B.const_data(),
      stride,
      &result));
  }
  else
  {
    boba_error("Build error! Unsupported data type for hipblas_dot");
  }

  return result;
}

#else

/**
 * @brief Fallback integer type used by HIP code paths in non-HIP builds.
 */
using rocblas_int = int;

/**
 * @brief Reports that hipBLAS GEMM support is unavailable.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @tparam C_view_t Output view type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t>
void hipblas_gemm(
  A_view_t const& matrix_A,
  B_view_t const& matrix_B,
  C_view_t const& matrix_C)
{
  detail::ignore(matrix_A);
  detail::ignore(matrix_B);
  detail::ignore(matrix_C);
  boba_error("Build error! You are trying to run hipblas_gemm BOBA_HIP_LIBS");
}

/**
 * @brief Reports that hipBLAS dot-product support is unavailable.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @param vector_A Left operand vector.
 * @param vector_B Right operand vector.
 * @return This function does not return successfully.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto hipblas_dot(
  A_view_t const& vector_A,
  B_view_t const& vector_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  detail::ignore(vector_A);
  detail::ignore(vector_B);
  boba_error("Build error! You are trying to run hipblas_dot without BOBA_HIP_LIBS");
}

#endif

} // namespace detail

#ifndef BOBA_HIP_LIBS
using rocblas_int = detail::rocblas_int;
#endif

} // namespace boba
