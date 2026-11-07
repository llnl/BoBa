// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_op.hpp"

#include <cstdint>
#include <limits>

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

  template <typename data_t>
  cusparse_csr_descriptor(
    index_t rows,
    index_t cols,
    index_t nnz,
    std::int32_t* row_offsets,
    std::int32_t* column_indices,
    data_t const* values)
  {
    cusparse_assert(cusparseCreateCsr(
      &m_descriptor,
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      static_cast<int64_t>(nnz),
      row_offsets,
      column_indices,
      const_cast<data_t*>(values),
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
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
 * @brief Checks that a CSR matrix fits cuSPARSE SpGEMM's 32-bit index range.
 */
inline void cusparse_spgemm_check_index_range(index_t rows, index_t cols, index_t nnz)
{
  constexpr index_t max_index = static_cast<index_t>(std::numeric_limits<std::int32_t>::max());
  boba_always_assert(
    rows <= max_index && cols <= max_index && nnz <= max_index,
    "cuSPARSE SpGEMM requires dimensions and nonzero counts to fit signed 32-bit indices");
}

/**
 * @brief Converts CSR indices from BoBa's 64-bit storage to cuSPARSE SpGEMM's 32-bit format.
 */
template <typename csr_view_t>
void cusparse_spgemm_convert_indices(
  csr_view_t const& matrix,
  Vector<execution_space::CUDA, std::int32_t>& columns,
  Vector<execution_space::CUDA, std::int32_t>& offsets)
{
  auto columns_view = columns.view();
  auto offsets_view = offsets.view();
  auto const* source_columns = matrix.column_indices_data();
  auto const* source_offsets = matrix.row_offsets_data();

  ::boba::loop<execution_space::CUDA, 1>(0_z, matrix.nnz(), [=] __boba_host_device__(index_t i)
  {
    columns_view(i) = static_cast<std::int32_t>(source_columns[i]);
  });
  ::boba::loop<execution_space::CUDA, 1>(0_z, matrix.rows() + 1, [=] __boba_host_device__(index_t i)
  {
    offsets_view(i) = static_cast<std::int32_t>(source_offsets[i]);
  });
}

/**
 * @brief Copies 32-bit cuSPARSE SpGEMM indices into BoBa's 64-bit CSR storage.
 */
template <typename csr_matrix_t>
void cusparse_spgemm_copy_indices(
  std::int32_t const* columns,
  std::int32_t const* offsets,
  csr_matrix_t& matrix)
{
  auto columns_view = matrix.m_column_indices.view();
  auto offsets_view = matrix.m_row_offsets.view();

  ::boba::loop<execution_space::CUDA, 1>(0_z, matrix.nnz(), [=] __boba_host_device__(index_t i)
  {
    columns_view(i) = static_cast<std::int64_t>(columns[i]);
  });
  ::boba::loop<execution_space::CUDA, 1>(0_z, matrix.rows() + 1, [=] __boba_host_device__(index_t i)
  {
    offsets_view(i) = static_cast<std::int64_t>(offsets[i]);
  });
}

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

/**
 * @brief Computes the sparse product `C = A * B` with cusparse SpGEMM.
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
void cusparse_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  using data_t = typename C_matrix_t::data_t;
  const data_t alpha{1};
  const data_t beta{0};
  const auto operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
  constexpr auto algorithm = CUSPARSE_SPGEMM_DEFAULT;
  constexpr auto compute_type = cuda_data_t_v<data_t>;

  cusparse_spgemm_check_index_range(A.rows(), A.cols(), A.nnz());
  cusparse_spgemm_check_index_range(B.rows(), B.cols(), B.nnz());
  cusparse_spgemm_check_index_range(C.rows(), C.cols(), 0);

  Vector<execution_space::CUDA, std::int32_t> A_columns({A.nnz()});
  Vector<execution_space::CUDA, std::int32_t> A_offsets({A.rows() + 1});
  Vector<execution_space::CUDA, std::int32_t> B_columns({B.nnz()});
  Vector<execution_space::CUDA, std::int32_t> B_offsets({B.rows() + 1});
  Vector<execution_space::CUDA, std::int32_t> C_columns({0});
  Vector<execution_space::CUDA, std::int32_t> C_offsets({C.rows() + 1});
  cusparse_spgemm_convert_indices(A, A_columns, A_offsets);
  cusparse_spgemm_convert_indices(B, B_columns, B_offsets);
  C_offsets.fill_with_zeros();
  cuda_assert(cudaDeviceSynchronize());

  cusparse_csr_descriptor A_descriptor(
    A.rows(), A.cols(), A.nnz(), A_offsets.data(), A_columns.data(), A.values_data());
  cusparse_csr_descriptor B_descriptor(
    B.rows(), B.cols(), B.nnz(), B_offsets.data(), B_columns.data(), B.values_data());
  cusparse_csr_descriptor C_descriptor(
    C.rows(), C.cols(), 0, C_offsets.data(), C_columns.data(), static_cast<data_t const*>(nullptr));

  cusparseSpGEMMDescr_t spgemm_descriptor = nullptr;
  cusparse_assert(cusparseSpGEMM_createDescr(&spgemm_descriptor));

  size_t work_size = 0;
  cusparse_assert(cusparseSpGEMM_workEstimation(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &work_size, nullptr));
  Vector<execution_space::CUDA, unsigned char> work_buffer({work_size});
  cusparse_assert(cusparseSpGEMM_workEstimation(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &work_size, work_buffer.data()));

  size_t compute_size = 0;
  cusparse_assert(cusparseSpGEMM_compute(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &compute_size, nullptr));
  Vector<execution_space::CUDA, unsigned char> compute_buffer({compute_size});
  cusparse_assert(cusparseSpGEMM_compute(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor, &compute_size, compute_buffer.data()));

  int64_t rows = 0;
  int64_t cols = 0;
  int64_t nnz = 0;
  cusparse_assert(cusparseSpMatGetSize(C_descriptor.get(), &rows, &cols, &nnz));
  boba_always_assert(
    nnz >= 0 && nnz <= std::numeric_limits<std::int32_t>::max(),
    "cuSPARSE SpGEMM result exceeds the signed 32-bit index range");
  C_columns.resize({static_cast<index_t>(nnz)});
  C.m_values.resize(static_cast<index_t>(nnz));
  C.m_column_indices.resize(static_cast<index_t>(nnz));
  auto values_view = C.m_values.view();
  cusparse_assert(cusparseCsrSetPointers(C_descriptor.get(), C_offsets.data(), C_columns.data(), values_view.data()));

  cusparse_assert(cusparseSpGEMM_copy(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor));
  cuda_assert(cudaDeviceSynchronize());

  cusparse_spgemm_copy_indices(C_columns.data(), C_offsets.data(), C);
  cuda_assert(cudaDeviceSynchronize());
  cusparse_assert(cusparseSpGEMM_destroyDescr(spgemm_descriptor));
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


/**
 * @brief Reports that cusparse SpGEMM support is unavailable.
 */
template <typename A_view_t, typename B_view_t, typename C_matrix_t>
void cusparse_spgemm(
  A_view_t const& A,
  B_view_t const& B,
  C_matrix_t& C)
{
  detail::ignore(A, B, C);
  boba_error("Build error! You are trying to run cusparse_spgemm without BOBA_CUDA_LIBS");
}

#endif

} // namespace detail
} // namespace boba
