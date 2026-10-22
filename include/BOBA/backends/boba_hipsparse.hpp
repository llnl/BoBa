// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_op.hpp"

#ifdef BOBA_HIP_LIBS
#include <hipsparse/hipsparse.h>
#endif

// AMD Documentation
// https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/reference/generic.html

namespace boba
{

namespace detail
{

#ifdef BOBA_HIP_LIBS

extern hipsparseHandle_t hipsparse_handle;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

#define hipsparse_assert(a) hipsparse_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipSPARSE error and terminates.
 * @param error hipSPARSE status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hipsparse_assert_(
  hipsparseStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == HIPSPARSE_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << " returned error code " << static_cast<int>(error) << std::endl;
  exit(1);
}

/**
 * @brief Maps a BoBa sparse operation to the hipSPARSE operation enum.
 */
[[nodiscard]]
inline hipsparseOperation_t hipsparse_operation(sparse_op op)
{
  return op == sparse_op::transpose ? HIPSPARSE_OPERATION_TRANSPOSE : HIPSPARSE_OPERATION_NON_TRANSPOSE;
}

/**
 * @brief Owning hipSPARSE descriptor for a CSR view.
 *
 * The descriptor aliases the view's device storage; the CSR matrix must
 * outlive it. hipSPARSE takes mutable pointers, but read-only operations
 * such as SpMV and SpMM do not modify them.
 */
class hipsparse_csr_descriptor
{
public:
  template <typename A_view_t>
  explicit hipsparse_csr_descriptor(A_view_t const& A)
  {
    using data_t = typename A_view_t::value_t;
    hipsparse_assert(hipsparseCreateCsr(
      &m_descriptor,
      static_cast<int64_t>(A.rows()),
      static_cast<int64_t>(A.cols()),
      static_cast<int64_t>(A.nnz()),
      const_cast<std::int64_t*>(A.row_offsets_data()),
      const_cast<std::int64_t*>(A.column_indices_data()),
      const_cast<data_t*>(A.values_data()),
      HIPSPARSE_INDEX_64I,
      HIPSPARSE_INDEX_64I,
      HIPSPARSE_INDEX_BASE_ZERO,
      hip_data_t_v<data_t>));
  }

  hipsparse_csr_descriptor(hipsparse_csr_descriptor const&) = delete;
  hipsparse_csr_descriptor& operator=(hipsparse_csr_descriptor const&) = delete;

  ~hipsparse_csr_descriptor()
  {
    hipsparse_assert(hipsparseDestroySpMat(m_descriptor));
  }

  [[nodiscard]]
  hipsparseSpMatDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  hipsparseSpMatDescr_t m_descriptor = nullptr;
};

/**
 * @brief Owning hipSPARSE dense-vector descriptor aliasing device storage.
 */
class hipsparse_dense_vector_descriptor
{
public:
  template <typename data_t>
  hipsparse_dense_vector_descriptor(data_t const* values, index_t size)
  {
    hipsparse_assert(hipsparseCreateDnVec(
      &m_descriptor,
      static_cast<int64_t>(size),
      const_cast<data_t*>(values),
      hip_data_t_v<data_t>));
  }

  hipsparse_dense_vector_descriptor(hipsparse_dense_vector_descriptor const&) = delete;
  hipsparse_dense_vector_descriptor& operator=(hipsparse_dense_vector_descriptor const&) = delete;

  ~hipsparse_dense_vector_descriptor()
  {
    hipsparse_assert(hipsparseDestroyDnVec(m_descriptor));
  }

  [[nodiscard]]
  hipsparseDnVecDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  hipsparseDnVecDescr_t m_descriptor = nullptr;
};

/**
 * @brief Owning hipSPARSE column-major dense-matrix descriptor aliasing device storage.
 */
class hipsparse_dense_matrix_descriptor
{
public:
  template <typename data_t>
  hipsparse_dense_matrix_descriptor(data_t const* values, index_t rows, index_t cols)
  {
    hipsparse_assert(hipsparseCreateDnMat(
      &m_descriptor,
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      static_cast<int64_t>(rows),
      const_cast<data_t*>(values),
      hip_data_t_v<data_t>,
      HIPSPARSE_ORDER_COL));
  }

  hipsparse_dense_matrix_descriptor(hipsparse_dense_matrix_descriptor const&) = delete;
  hipsparse_dense_matrix_descriptor& operator=(hipsparse_dense_matrix_descriptor const&) = delete;

  ~hipsparse_dense_matrix_descriptor()
  {
    hipsparse_assert(hipsparseDestroyDnMat(m_descriptor));
  }

  [[nodiscard]]
  hipsparseDnMatDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  hipsparseDnMatDescr_t m_descriptor = nullptr;
};

/**
 * @brief Computes `y = alpha * op(A) * x + beta * y` with hipSPARSE SpMV.
 * @param A Device CSR view.
 * @param x Dense input vector view.
 * @param y Dense output vector view.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void hipsparse_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  hipsparse_csr_descriptor A_descriptor(A);
  hipsparse_dense_vector_descriptor x_descriptor(x.const_data(), x.size());
  hipsparse_dense_vector_descriptor y_descriptor(y.const_data(), y.size());
  auto const operation = hipsparse_operation(op);

  size_t workspace_size = 0;
  hipsparse_assert(hipsparseSpMV_bufferSize(
    hipsparse_handle,
    operation,
    &alpha,
    A_descriptor.get(),
    x_descriptor.get(),
    &beta,
    y_descriptor.get(),
    hip_data_t_v<data_t>,
    HIPSPARSE_SPMV_ALG_DEFAULT,
    &workspace_size));

  Vector<execution_space::HIP, unsigned char> workspace({workspace_size});
  hipsparse_assert(hipsparseSpMV(
    hipsparse_handle,
    operation,
    &alpha,
    A_descriptor.get(),
    x_descriptor.get(),
    &beta,
    y_descriptor.get(),
    hip_data_t_v<data_t>,
    HIPSPARSE_SPMV_ALG_DEFAULT,
    workspace.data()));
}

/**
 * @brief Computes `C = alpha * op(A) * B + beta * C` with hipSPARSE SpMM.
 * @param A Device CSR view.
 * @param B Dense column-major input matrix view.
 * @param C Dense column-major output matrix view.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void hipsparse_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  hipsparse_csr_descriptor A_descriptor(A);
  hipsparse_dense_matrix_descriptor B_descriptor(B.const_data(), B.sizes(0), B.sizes(1));
  hipsparse_dense_matrix_descriptor C_descriptor(C.const_data(), C.sizes(0), C.sizes(1));
  auto const operation = hipsparse_operation(op);

  size_t workspace_size = 0;
  hipsparse_assert(hipsparseSpMM_bufferSize(
    hipsparse_handle,
    operation,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    A_descriptor.get(),
    B_descriptor.get(),
    &beta,
    C_descriptor.get(),
    hip_data_t_v<data_t>,
    HIPSPARSE_SPMM_ALG_DEFAULT,
    &workspace_size));

  Vector<execution_space::HIP, unsigned char> workspace({workspace_size});
  hipsparse_assert(hipsparseSpMM(
    hipsparse_handle,
    operation,
    HIPSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    A_descriptor.get(),
    B_descriptor.get(),
    &beta,
    C_descriptor.get(),
    hip_data_t_v<data_t>,
    HIPSPARSE_SPMM_ALG_DEFAULT,
    workspace.data()));
}

/**
 * @brief Computes the sparse product `C = A * B` with hipsparse SpGEMM.
 *
 * Only non-transposed operands are supported by the library. Output rows are
 * sorted.
 *
 * @param A Device CSR view.
 * @param B Device CSR view.
 * @param C CSR matrix of shape `(A.rows(), B.cols())` with allocated row
 *        offsets; its column and value storage is replaced.
 */
template <typename A_view_t, typename B_view_t, typename C_matrix_t>
void hipsparse_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  using data_t = typename C_matrix_t::data_t;
  const data_t alpha{1};
  const data_t beta{0};
  const auto operation = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr auto algorithm = HIPSPARSE_SPGEMM_DEFAULT;
  constexpr auto compute_type = hip_data_t_v<data_t>;

  hipsparse_csr_descriptor A_descriptor(A);
  hipsparse_csr_descriptor B_descriptor(B);
  auto C_view = C.view();
  hipsparse_csr_descriptor C_descriptor(C_view);

  hipsparseSpGEMMDescr_t spgemm_descriptor = nullptr;
  hipsparse_assert(hipsparseSpGEMM_createDescr(&spgemm_descriptor));

  size_t work_size = 0;
  hipsparse_assert(hipsparseSpGEMM_workEstimation(
    hipsparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &work_size, nullptr));
  Vector<execution_space::HIP, unsigned char> work_buffer({work_size});
  hipsparse_assert(hipsparseSpGEMM_workEstimation(
    hipsparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &work_size, work_buffer.data()));

  size_t compute_size = 0;
  hipsparse_assert(hipsparseSpGEMM_compute(
    hipsparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &compute_size, nullptr));
  Vector<execution_space::HIP, unsigned char> compute_buffer({compute_size});
  hipsparse_assert(hipsparseSpGEMM_compute(
    hipsparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &compute_size, compute_buffer.data()));

  int64_t rows = 0;
  int64_t cols = 0;
  int64_t nnz = 0;
  hipsparse_assert(hipsparseSpMatGetSize(C_descriptor.get(), &rows, &cols, &nnz));
  C.m_values.resize(static_cast<index_t>(nnz));
  C.m_column_indices.resize(static_cast<index_t>(nnz));
  auto values_view = C.m_values.view();
  auto columns_view = C.m_column_indices.view();
  auto offsets_view = C.m_row_offsets.view();
  hipsparse_assert(hipsparseCsrSetPointers(C_descriptor.get(), offsets_view.data(), columns_view.data(), values_view.data()));

  hipsparse_assert(hipsparseSpGEMM_copy(
    hipsparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor));
  hipsparse_assert(hipsparseSpGEMM_destroyDescr(spgemm_descriptor));
}

#else

/**
 * @brief Reports that hipSPARSE SpMV support is unavailable.
 */
template <typename A_view_t, typename x_view_t, typename y_view_t, typename data_t>
void hipsparse_spmv(
  A_view_t const& A,
  x_view_t const& x,
  y_view_t const& y,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, x, y, alpha, beta, op);
  boba_error("Build error! You are trying to run hipsparse_spmv without BOBA_HIP_LIBS");
}

/**
 * @brief Reports that hipSPARSE SpMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_view_t, typename data_t>
void hipsparse_spmm(
  A_view_t const& A,
  B_view_t const& B,
  C_view_t const& C,
  data_t alpha,
  data_t beta,
  sparse_op op)
{
  detail::ignore(A, B, C, alpha, beta, op);
  boba_error("Build error! You are trying to run hipsparse_spmm without BOBA_HIP_LIBS");
}


/**
 * @brief Reports that hipsparse SpGEMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_matrix_t>
void hipsparse_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  detail::ignore(A, B, C);
  boba_error("Build error! You are trying to run hipsparse_spgemm without BOBA_HIP_LIBS");
}

#endif

} // namespace detail
} // namespace boba
