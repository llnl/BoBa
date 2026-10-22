// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/sparse/sparse_op.hpp"

#ifdef BOBA_CUDA_LIBS
#include <cusparse.h>
#endif

#include <cstdint>
#include <limits>

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

struct csr_index_copy_to_32
{
  std::int64_t const* source;
  std::int32_t* destination;

  __boba_host_device__ void operator()(index_t i) const
  {
    destination[i] = static_cast<std::int32_t>(source[i]);
  }
};

struct csr_index_copy_to_64
{
  std::int32_t const* source;
  std::int64_t* destination;

  __boba_host_device__ void operator()(index_t i) const
  {
    destination[i] = static_cast<std::int64_t>(source[i]);
  }
};

/**
 * @brief Owning cuSPARSE CSR descriptor with converted 32-bit indices.
 *
 * BoBa keeps CSR indices as signed 64-bit values, but some cuSPARSE SpGEMM
 * implementations only support 32-bit index descriptors. The converted
 * buffers remain alive for the lifetime of the descriptor and are copied back
 * to the owning CSR matrix after SpGEMM completes.
 */
class cusparse_csr_descriptor_32
{
public:
  template <typename A_view_t>
  explicit cusparse_csr_descriptor_32(A_view_t const& A)
      : m_column_indices({A.nnz()}),
        m_row_offsets({A.rows() + 1})
  {
    constexpr auto max_index = static_cast<index_t>(std::numeric_limits<std::int32_t>::max());
    boba_always_assert(A.rows() <= max_index, "cuSPARSE CSR rows exceed 32-bit index support");
    boba_always_assert(A.cols() <= max_index, "cuSPARSE CSR columns exceed 32-bit index support");
    boba_always_assert(A.nnz() <= max_index, "cuSPARSE CSR nonzeros exceed 32-bit index support");

    auto column_indices = m_column_indices.view();
    auto row_offsets = m_row_offsets.view();
    const auto source_column_indices = A.column_indices_data();
    const auto source_row_offsets = A.row_offsets_data();
    ::boba::loop<::boba::execution_space::CUDA, 1>(
      A.nnz(), csr_index_copy_to_32{source_column_indices, column_indices.data()});
    ::boba::loop<::boba::execution_space::CUDA, 1>(
      A.rows() + 1, csr_index_copy_to_32{source_row_offsets, row_offsets.data()});

    using data_t = typename A_view_t::value_t;
    cusparse_assert(cusparseCreateCsr(
      &m_descriptor,
      static_cast<int64_t>(A.rows()),
      static_cast<int64_t>(A.cols()),
      static_cast<int64_t>(A.nnz()),
      m_row_offsets.data(),
      m_column_indices.data(),
      const_cast<data_t*>(A.values_data()),
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      cuda_data_t_v<data_t>));
  }

  cusparse_csr_descriptor_32(cusparse_csr_descriptor_32 const&) = delete;
  cusparse_csr_descriptor_32& operator=(cusparse_csr_descriptor_32 const&) = delete;

  ~cusparse_csr_descriptor_32()
  {
    cusparse_assert(cusparseDestroySpMat(m_descriptor));
  }

  template <typename C_matrix_t>
  void set_output_pointers(C_matrix_t& C)
  {
    m_column_indices.resize({C.nnz()});
    cusparse_assert(cusparseCsrSetPointers(
      m_descriptor,
      m_row_offsets.data(),
      m_column_indices.data(),
      C.m_values.data()));
  }

  template <typename C_matrix_t>
  void copy_indices_to(C_matrix_t& C) const
  {
    auto row_offsets = C.m_row_offsets.view();
    auto column_indices = C.m_column_indices.view();
    auto source_row_offsets = m_row_offsets.const_view();
    auto source_column_indices = m_column_indices.const_view();
    ::boba::loop<::boba::execution_space::CUDA, 1>(
      C.rows() + 1, csr_index_copy_to_64{source_row_offsets.data(), row_offsets.data()});
    ::boba::loop<::boba::execution_space::CUDA, 1>(
      C.nnz(), csr_index_copy_to_64{source_column_indices.data(), column_indices.data()});
  }

  [[nodiscard]]
  cusparseSpMatDescr_t get() const noexcept
  {
    return m_descriptor;
  }

private:
  Vector<::boba::execution_space::CUDA, std::int32_t> m_column_indices;
  Vector<::boba::execution_space::CUDA, std::int32_t> m_row_offsets;
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

  cusparse_csr_descriptor_32 A_descriptor(A);
  cusparse_csr_descriptor_32 B_descriptor(B);
  auto C_view = C.view();
  cusparse_csr_descriptor_32 C_descriptor(C_view);

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
  C.m_values.resize(static_cast<index_t>(nnz));
  C.m_column_indices.resize(static_cast<index_t>(nnz));
  C_descriptor.set_output_pointers(C);

  cusparse_assert(cusparseSpGEMM_copy(
    cusparse_handle, operation, operation, &alpha, A_descriptor.get(), B_descriptor.get(), &beta, C_descriptor.get(),
    compute_type, algorithm, spgemm_descriptor));
  C_descriptor.copy_indices_to(C);
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
