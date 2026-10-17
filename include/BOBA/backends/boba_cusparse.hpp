// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_op.hpp"

#ifdef BOBA_CUDA_LIBS
#include <cusparse.h>
#endif

// NVIDIA Documentation
// https://docs.nvidia.com/cuda/cusparse/#cusparse-generic-api-reference

namespace boba
{

namespace detail
{

#ifdef BOBA_CUDA_LIBS

extern cusparseHandle_t cusparse_handle;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

#define cusparse_assert(a) cusparse_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a cuSPARSE error and terminates.
 * @param error cuSPARSE status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cusparse_assert_(
  cusparseStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == CUSPARSE_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << " returned an error code: " << std::endl;
  std::cout << cusparseGetErrorName(error) << ": " << cusparseGetErrorString(error) << std::endl;
  exit(1);
}

/**
 * @brief Maps a BoBa sparse operation to the cuSPARSE operation enum.
 */
[[nodiscard]]
inline cusparseOperation_t cusparse_operation(sparse_op op)
{
  return op == sparse_op::transpose ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
}

/**
 * @brief Owning cuSPARSE descriptor for a CSR view.
 *
 * The descriptor aliases the view's device storage; the CSR matrix must
 * outlive it. cuSPARSE takes mutable pointers, but read-only operations
 * such as SpMV and SpMM do not modify them.
 */
class cusparse_csr_descriptor
{
public:
  template <typename A_view_t>
  explicit cusparse_csr_descriptor(A_view_t const& A)
  {
    using data_t = typename A_view_t::value_t;
    cusparse_assert(cusparseCreateCsr(
      &m_descriptor,
      static_cast<int64_t>(A.rows()),
      static_cast<int64_t>(A.cols()),
      static_cast<int64_t>(A.nnz()),
      const_cast<std::int64_t*>(A.row_offsets_data()),
      const_cast<std::int64_t*>(A.column_indices_data()),
      const_cast<data_t*>(A.values_data()),
      CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_64I,
      CUSPARSE_INDEX_BASE_ZERO,
      cuda_data_t_v<data_t>));
  }

  cusparse_csr_descriptor(cusparse_csr_descriptor const&) = delete;
  cusparse_csr_descriptor& operator=(cusparse_csr_descriptor const&) = delete;

  ~cusparse_csr_descriptor()
  {
    cusparse_assert(cusparseDestroySpMat(m_descriptor));
  }

  [[nodiscard]]
  cusparseSpMatDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  cusparseSpMatDescr_t m_descriptor = nullptr;
};

/**
 * @brief Owning cuSPARSE dense-vector descriptor aliasing device storage.
 */
class cusparse_dense_vector_descriptor
{
public:
  template <typename data_t>
  cusparse_dense_vector_descriptor(data_t const* values, index_t size)
  {
    cusparse_assert(cusparseCreateDnVec(
      &m_descriptor,
      static_cast<int64_t>(size),
      const_cast<data_t*>(values),
      cuda_data_t_v<data_t>));
  }

  cusparse_dense_vector_descriptor(cusparse_dense_vector_descriptor const&) = delete;
  cusparse_dense_vector_descriptor& operator=(cusparse_dense_vector_descriptor const&) = delete;

  ~cusparse_dense_vector_descriptor()
  {
    cusparse_assert(cusparseDestroyDnVec(m_descriptor));
  }

  [[nodiscard]]
  cusparseDnVecDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  cusparseDnVecDescr_t m_descriptor = nullptr;
};

/**
 * @brief Owning cuSPARSE column-major dense-matrix descriptor aliasing device storage.
 */
class cusparse_dense_matrix_descriptor
{
public:
  template <typename data_t>
  cusparse_dense_matrix_descriptor(data_t const* values, index_t rows, index_t cols)
  {
    cusparse_assert(cusparseCreateDnMat(
      &m_descriptor,
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      static_cast<int64_t>(rows),
      const_cast<data_t*>(values),
      cuda_data_t_v<data_t>,
      CUSPARSE_ORDER_COL));
  }

  cusparse_dense_matrix_descriptor(cusparse_dense_matrix_descriptor const&) = delete;
  cusparse_dense_matrix_descriptor& operator=(cusparse_dense_matrix_descriptor const&) = delete;

  ~cusparse_dense_matrix_descriptor()
  {
    cusparse_assert(cusparseDestroyDnMat(m_descriptor));
  }

  [[nodiscard]]
  cusparseDnMatDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  cusparseDnMatDescr_t m_descriptor = nullptr;
};

/**
 * @brief Computes `y = alpha * op(A) * x + beta * y` with cuSPARSE SpMV.
 * @param A Device CSR view.
 * @param x Dense input vector view.
 * @param y Dense output vector view.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void cusparse_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  cusparse_csr_descriptor A_descriptor(A);
  cusparse_dense_vector_descriptor x_descriptor(x.const_data(), x.size());
  cusparse_dense_vector_descriptor y_descriptor(y.const_data(), y.size());
  auto const operation = cusparse_operation(op);

  size_t workspace_size = 0;
  cusparse_assert(cusparseSpMV_bufferSize(
    cusparse_handle,
    operation,
    &alpha,
    A_descriptor.get(),
    x_descriptor.get(),
    &beta,
    y_descriptor.get(),
    cuda_data_t_v<data_t>,
    CUSPARSE_SPMV_ALG_DEFAULT,
    &workspace_size));

  Vector<execution_space::CUDA, unsigned char> workspace({workspace_size});
  cusparse_assert(cusparseSpMV(
    cusparse_handle,
    operation,
    &alpha,
    A_descriptor.get(),
    x_descriptor.get(),
    &beta,
    y_descriptor.get(),
    cuda_data_t_v<data_t>,
    CUSPARSE_SPMV_ALG_DEFAULT,
    workspace.data()));
}

/**
 * @brief Computes `C = alpha * op(A) * B + beta * C` with cuSPARSE SpMM.
 * @param A Device CSR view.
 * @param B Dense column-major input matrix view.
 * @param C Dense column-major output matrix view.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void cusparse_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  cusparse_csr_descriptor A_descriptor(A);
  cusparse_dense_matrix_descriptor B_descriptor(B.const_data(), B.sizes(0), B.sizes(1));
  cusparse_dense_matrix_descriptor C_descriptor(C.const_data(), C.sizes(0), C.sizes(1));
  auto const operation = cusparse_operation(op);

  size_t workspace_size = 0;
  cusparse_assert(cusparseSpMM_bufferSize(
    cusparse_handle,
    operation,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    A_descriptor.get(),
    B_descriptor.get(),
    &beta,
    C_descriptor.get(),
    cuda_data_t_v<data_t>,
    CUSPARSE_SPMM_ALG_DEFAULT,
    &workspace_size));

  Vector<execution_space::CUDA, unsigned char> workspace({workspace_size});
  cusparse_assert(cusparseSpMM(
    cusparse_handle,
    operation,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    A_descriptor.get(),
    B_descriptor.get(),
    &beta,
    C_descriptor.get(),
    cuda_data_t_v<data_t>,
    CUSPARSE_SPMM_ALG_DEFAULT,
    workspace.data()));
}

#else

/**
 * @brief Reports that cuSPARSE SpMV support is unavailable.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void cusparse_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, x, y, alpha, beta, op);
  boba_error("Build error! You are trying to run cusparse_spmv without BOBA_CUDA_LIBS");
}

/**
 * @brief Reports that cuSPARSE SpMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void cusparse_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, B, C, alpha, beta, op);
  boba_error("Build error! You are trying to run cusparse_spmm without BOBA_CUDA_LIBS");
}

#endif

} // namespace detail
} // namespace boba
