// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_HIP
#include <hipsolver/hipsolver.h>
#include <rocsolver/rocsolver.h>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_HIP_LIBS

extern hipsolverDnHandle_t hipsolver_handle;

template <typename data_t>
struct hipsolver_api_data
{
  using type = data_t;
};

template <>
struct hipsolver_api_data<complex<float>>
{
  using type = hipFloatComplex;
};

template <>
struct hipsolver_api_data<complex<double>>
{
  using type = hipDoubleComplex;
};

template <typename data_t>
using hipsolver_api_data_t = typename hipsolver_api_data<data_t>::type;

template <typename data_t>
struct rocsolver_api_data
{
  using type = data_t;
};

template <>
struct rocsolver_api_data<complex<float>>
{
  using type = rocblas_float_complex;
};

template <>
struct rocsolver_api_data<complex<double>>
{
  using type = rocblas_double_complex;
};

template <typename data_t>
using rocsolver_api_data_t = typename rocsolver_api_data<data_t>::type;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

/**
 * @brief Expands a hipSOLVER status code into assertion failures with readable labels.
 * @param error hipSOLVER status code to decode.
 */
inline void hipsolver_error_codes_(hipsolverStatus_t error)
{
  boba_always_assert(error != HIPSOLVER_STATUS_NOT_INITIALIZED, "HIPSOLVER_STATUS_NOT_INITIALIZED ");
  boba_always_assert(error != HIPSOLVER_STATUS_ALLOC_FAILED, "HIPSOLVER_STATUS_ALLOC_FAILED");
  boba_always_assert(error != HIPSOLVER_STATUS_INVALID_VALUE, "HIPSOLVER_STATUS_INVALID_VALUE ");
  boba_always_assert(error != HIPSOLVER_STATUS_ARCH_MISMATCH, "HIPSOLVER_STATUS_ARCH_MISMATCH ");
  boba_always_assert(error != HIPSOLVER_STATUS_EXECUTION_FAILED, "HIPSOLVER_STATUS_EXECUTION_FAILED ");
  boba_always_assert(error != HIPSOLVER_STATUS_INTERNAL_ERROR, "HIPSOLVER_STATUS_INTERNAL_ERROR");
  boba_always_assert_equal(error, HIPSOLVER_STATUS_SUCCESS, "hipsolver error");
}

#define hipsolver_assert(a) hipsolver_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipSOLVER error and terminates.
 * @param error hipSOLVER status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hipsolver_assert_(
  hipsolverStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == HIPSOLVER_STATUS_SUCCESS)
  {
    return;
  }
  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  hipsolver_error_codes_(error);
  exit(1);
}

#define hipsolver_assert_info(a, b, c) hipsolver_assert_info_(a, #a, b, c, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipSOLVER error or nonzero info flag and terminates.
 * @param error hipSOLVER status code to check.
 * @param call Source expression that produced the error.
 * @param info Device pointer to the backend info flag.
 * @param info_meaning Description of the info flag semantics.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hipsolver_assert_info_(
  hipsolverStatus_t error,
  const std::string& call,
  int* info,
  const std::string& info_meaning,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  int host_info = 0;
  hip_memcpy(&host_info, info, 1);

  if ((error == HIPSOLVER_STATUS_SUCCESS) and (host_info == 0))
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  std::cout << "Info flag: " << host_info << std::endl;
  std::cout << "Info meaning: " << info_meaning << std::endl;

  if (error != HIPSOLVER_STATUS_SUCCESS)
  {
    hipsolver_error_codes_(error);
  }
  boba_error("hipSOLVER reported a nonzero info flag");
}

/**
 * \brief Solves a least-squares system with hipSOLVER's QR path.
 *
 * hipSOLVER's `hipsolverDnDgeqrf` factorization overwrites \p matrix_A with
 * the packed Householder representation. The right-hand side \p matrix_B
 * remains in-place because `hipsolverDnDormqr` and `hipblasDtrsm` overwrite it
 * with the solution.
 *
 * ROCm follows LAPACK-style `info` reporting here: a negative value from
 * `Dgeqrf` or `Dormqr` means the corresponding 1-based argument index was
 * invalid.
 *
 * \see https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/api/lapack.html
 */

template <typename data_t>
void ls_solve_qr_hipsolver(Matrix<execution_space::HIP, data_t>& matrix_A,
                           Matrix<execution_space::HIP, data_t>& matrix_B)
{
  auto m = matrix_A.rows();
  auto n = matrix_A.cols();
  auto nrhs = matrix_B.cols();
  auto lda = matrix_A.rows();
  auto ldb = matrix_B.rows();

  boba_always_assert_ge(matrix_A.rows(), matrix_A.cols(), "Incorrect solver.");

  Vector<execution_space::HIP, data_t> tau({n});
  Vector<execution_space::HIP, int> info({1});

  // 1) geqrf
  device_sync();
  int lwork_geqrf = 0;
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(
      hipsolverDnSgeqrf_bufferSize(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        &lwork_geqrf));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(
      hipsolverDnDgeqrf_bufferSize(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        &lwork_geqrf));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(
      hipsolverDnCgeqrf_bufferSize(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        &lwork_geqrf));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(
      hipsolverDnZgeqrf_bufferSize(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        &lwork_geqrf));
  }
  else
  {
    boba_error("Build error! Unsupported data type for ls_solve_qr_hipsolver");
  }

  Vector<execution_space::HIP, data_t> work_geqrf({static_cast<::boba::index_t>(lwork_geqrf)});

  device_sync();
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverDnSgeqrf(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        work_geqrf.data(),
        lwork_geqrf,
        info.data()),
      info.data(),
      "Sgeqrf returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDnDgeqrf(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        work_geqrf.data(),
        lwork_geqrf,
        info.data()),
      info.data(),
      "Dgeqrf returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverDnCgeqrf(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        work_geqrf.data(),
        lwork_geqrf,
        info.data()),
      info.data(),
      "Cgeqrf returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverDnZgeqrf(
        hipsolver_handle,
        m,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        work_geqrf.data(),
        lwork_geqrf,
        info.data()),
      info.data(),
      "Zgeqrf returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else
  {
    boba_error("Build error! Unsupported data type for ls_solve_qr_hipsolver");
  }

  // 2) B <- Q^T B
  int lwork_ormqr = 0;
  device_sync();
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(
      hipsolverDnSormqr_bufferSize(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_T,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        &lwork_ormqr));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(
      hipsolverDnDormqr_bufferSize(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_T,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        &lwork_ormqr));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(
      hipsolverDnCormqr_bufferSize(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_C,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        &lwork_ormqr));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(
      hipsolverDnZormqr_bufferSize(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_C,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        &lwork_ormqr));
  }
  else
  {
    boba_error("Build error! Unsupported data type for ls_solve_qr_hipsolver");
  }

  Vector<execution_space::HIP, data_t> work_ormqr({static_cast<::boba::index_t>(lwork_ormqr)});

  device_sync();
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverDnSormqr(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_T,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        work_ormqr.data(),
        lwork_ormqr,
        info.data()),
      info.data(),
      "Sormqr returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDnDormqr(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_T,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        work_ormqr.data(),
        lwork_ormqr,
        info.data()),
      info.data(),
      "Dormqr returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverDnCormqr(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_C,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        work_ormqr.data(),
        lwork_ormqr,
        info.data()),
      info.data(),
      "Cormqr returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverDnZormqr(
        hipsolver_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_OP_C,
        m,
        nrhs,
        n,
        matrix_A.data(),
        lda,
        tau.data(),
        matrix_B.data(),
        ldb,
        work_ormqr.data(),
        lwork_ormqr,
        info.data()),
      info.data(),
      "Zormqr returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");
  }
  else
  {
    boba_error("Build error! Unsupported data type for ls_solve_qr_hipsolver");
  }

  // 3) Solve R X = B(1:n,:)
  const auto one = PotentiallyComplex<data_t>::value(1.0);

  device_sync();
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipblas_assert(
      hipblasStrsm(
        hipblas_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_FILL_MODE_UPPER,
        HIPBLAS_OP_N,
        HIPBLAS_DIAG_NON_UNIT,
        n,
        nrhs,
        &one,
        matrix_A.data(),
        lda,
        matrix_B.data(),
        ldb));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipblas_assert(
      hipblasDtrsm(
        hipblas_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_FILL_MODE_UPPER,
        HIPBLAS_OP_N,
        HIPBLAS_DIAG_NON_UNIT,
        n,
        nrhs,
        &one,
        matrix_A.data(),
        lda,
        matrix_B.data(),
        ldb));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipblas_assert(
      hipblasCtrsm(
        hipblas_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_FILL_MODE_UPPER,
        HIPBLAS_OP_N,
        HIPBLAS_DIAG_NON_UNIT,
        n,
        nrhs,
        &one,
        matrix_A.data(),
        lda,
        matrix_B.data(),
        ldb));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipblas_assert(
      hipblasZtrsm(
        hipblas_handle,
        HIPBLAS_SIDE_LEFT,
        HIPBLAS_FILL_MODE_UPPER,
        HIPBLAS_OP_N,
        HIPBLAS_DIAG_NON_UNIT,
        n,
        nrhs,
        &one,
        matrix_A.data(),
        lda,
        matrix_B.data(),
        ldb));
  }
  else
  {
    boba_error("Build error! Unsupported data type for ls_solve_qr_hipsolver");
  }

  device_sync();
}

template <typename data_t>
void ls_solve_qr_hipsolver(Matrix<execution_space::HIP, data_t> const& matrix_A,
                           Matrix<execution_space::HIP, data_t>& matrix_B)
{
  Matrix<execution_space::HIP, data_t> factor_A(matrix_A);
  ls_solve_qr_hipsolver(factor_A, matrix_B);
}

/**
 * \brief Computes the in-place QR panel factorization used by BoBa's HIP QR.
 *
 * On return, \p matrix_A contains the packed Householder representation
 * produced by hipSOLVER and \p tau stores the reflector coefficients.
 *
 * \see https://hipsolver.readthedocs.io/en/rocm-5.2.3/api_lapackfunc.html#list-of-orthogonal-factorizations
 */
template <typename data_t>
void factor_qr_hipsolver(
  Matrix<execution_space::HIP, data_t>& matrix_A,
  Vector<execution_space::HIP, data_t>& tau)
{
  auto rows = matrix_A.rows();
  auto cols = matrix_A.cols();
  BOBA_CALI_BEGIN("hipsolver_get_qr_buffer_sizes");

  int qr_device_work_size = 0;
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(
      hipsolverDnSgeqrf_bufferSize(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        &qr_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(
      hipsolverDnDgeqrf_bufferSize(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        &qr_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(
      hipsolverDnCgeqrf_bufferSize(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        &qr_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(
      hipsolverDnZgeqrf_bufferSize(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        &qr_device_work_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_qr_hipsolver");
  }

  BOBA_CALI_SWITCH("hipsolver_get_qr_buffer_sizes", "allocate_buffers");
  Vector<execution_space::HIP, data_t> qr_device_work({static_cast<size_t>(qr_device_work_size)});
  Vector<execution_space::HIP, int> qr_info({1_z});

  BOBA_CALI_SWITCH("allocate_buffers", "hipsolver_qr");
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverDnSgeqrf(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        tau.data(),
        qr_device_work.data(),
        qr_device_work.size(),
        qr_info.data()),
      qr_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDnDgeqrf(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        tau.data(),
        qr_device_work.data(),
        qr_device_work.size(),
        qr_info.data()),
      qr_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverDnCgeqrf(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        tau.data(),
        qr_device_work.data(),
        qr_device_work.size(),
        qr_info.data()),
      qr_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverDnZgeqrf(
        hipsolver_handle,
        static_cast<int>(rows),
        static_cast<int>(cols),
        matrix_A.data(),
        static_cast<int>(rows),
        tau.data(),
        qr_device_work.data(),
        qr_device_work.size(),
        qr_info.data()),
      qr_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_qr_hipsolver");
  }
  BOBA_CALI_END("hipsolver_qr");
}

/**
 * \brief Expands packed Householder data into the reduced Q factor on HIP.
 *
 * \param matrix_Q Matrix containing the packed Householder vectors on entry and
 * the reduced orthonormal factor on exit.
 * \param tau Reflector coefficients returned by factor_qr_hipsolver().
 * \param ranks Number of reflectors to apply.
 *
 * \see https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/api/lapack.html
 */
template <typename data_t>
void form_q_hipsolver(
  Matrix<execution_space::HIP, data_t>& matrix_Q,
  Vector<execution_space::HIP, data_t>& tau,
  size_t ranks)
{
  int q_device_work_size = 0;
  BOBA_CALI_BEGIN("hipsolverDorgqr_bufferSize");
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(hipsolverDnSorgqr_bufferSize(
      hipsolver_handle,
      static_cast<int>(matrix_Q.rows()),
      static_cast<int>(matrix_Q.cols()),
      static_cast<int>(ranks),
      matrix_Q.data(),
      static_cast<int>(matrix_Q.rows()),
      tau.data(),
      &q_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(hipsolverDnDorgqr_bufferSize(
      hipsolver_handle,
      static_cast<int>(matrix_Q.rows()),
      static_cast<int>(matrix_Q.cols()),
      static_cast<int>(ranks),
      matrix_Q.data(),
      static_cast<int>(matrix_Q.rows()),
      tau.data(),
      &q_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(hipsolverDnCungqr_bufferSize(
      hipsolver_handle,
      static_cast<int>(matrix_Q.rows()),
      static_cast<int>(matrix_Q.cols()),
      static_cast<int>(ranks),
      matrix_Q.data(),
      static_cast<int>(matrix_Q.rows()),
      tau.data(),
      &q_device_work_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(hipsolverDnZungqr_bufferSize(
      hipsolver_handle,
      static_cast<int>(matrix_Q.rows()),
      static_cast<int>(matrix_Q.cols()),
      static_cast<int>(ranks),
      matrix_Q.data(),
      static_cast<int>(matrix_Q.rows()),
      tau.data(),
      &q_device_work_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for form_q_hipsolver");
  }

  Vector<execution_space::HIP, data_t> q_device_work({static_cast<size_t>(q_device_work_size)});
  Vector<execution_space::HIP, int> q_info({1_z});

  BOBA_CALI_SWITCH("hipsolverDorgqr_bufferSize", "hipsolverDorgqr");
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverDnSorgqr(
        hipsolver_handle,
        static_cast<int>(matrix_Q.rows()),
        static_cast<int>(matrix_Q.cols()),
        static_cast<int>(ranks),
        matrix_Q.data(),
        static_cast<int>(matrix_Q.rows()),
        tau.data(),
        q_device_work.data(),
        q_device_work.size(),
        q_info.data()),
      q_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDnDorgqr(
        hipsolver_handle,
        static_cast<int>(matrix_Q.rows()),
        static_cast<int>(matrix_Q.cols()),
        static_cast<int>(ranks),
        matrix_Q.data(),
        static_cast<int>(matrix_Q.rows()),
        tau.data(),
        q_device_work.data(),
        q_device_work.size(),
        q_info.data()),
      q_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverDnCungqr(
        hipsolver_handle,
        static_cast<int>(matrix_Q.rows()),
        static_cast<int>(matrix_Q.cols()),
        static_cast<int>(ranks),
        matrix_Q.data(),
        static_cast<int>(matrix_Q.rows()),
        tau.data(),
        q_device_work.data(),
        q_device_work.size(),
        q_info.data()),
      q_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverDnZungqr(
        hipsolver_handle,
        static_cast<int>(matrix_Q.rows()),
        static_cast<int>(matrix_Q.cols()),
        static_cast<int>(ranks),
        matrix_Q.data(),
        static_cast<int>(matrix_Q.rows()),
        tau.data(),
        q_device_work.data(),
        q_device_work.size(),
        q_info.data()),
      q_info.data(),
      "Documentation is nonexistent. Assume the ith argument is wrong (not including handle)");
  }
  else
  {
    boba_error("Build error! Unsupported data type for form_q_hipsolver");
  }
  BOBA_CALI_END("hipsolverDorgqr");
}

/**
 * \brief Forms a partial-pivot LU factorization with hipSOLVER.
 *
 * On return, `P * A = L * U`.
 *
 * \see https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/api/lapack.html
 */
template <typename data_t>
void factor_lu_hipsolver(
  Matrix<execution_space::HIP, data_t> const& input_in,
  Matrix<execution_space::HIP, data_t>& matrix_L,
  Matrix<execution_space::HIP, data_t>& matrix_U,
  PermutationMatrix<execution_space::HIP, ::boba::index_t>& matrix_P)
{
  BOBA_CALI_BEGIN("initialize_lu");
  boba_always_assert_equal(input_in.rows(), input_in.cols(), "LU requires nonsingular square matrix");

  auto input = input_in;

  int lu_device_temp_size = 0;
  BOBA_CALI_SWITCH("initialize_lu", "compute_LU");
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(
      hipsolverSgetrf_bufferSize(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(
      hipsolverDgetrf_bufferSize(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(
      hipsolverCgetrf_bufferSize(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(
      hipsolverZgetrf_bufferSize(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_lu_hipsolver");
  }

  Vector<execution_space::HIP, int> info({1});
  info.fill_with_zeros();

  Vector<execution_space::HIP, data_t> lu_device_temp({static_cast<size_t>(lu_device_temp_size)});
  Vector<execution_space::HIP, int> perms({input.rows()});

  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverSgetrf(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        lu_device_temp.size(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDgetrf(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        lu_device_temp.size(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverCgetrf(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        lu_device_temp.size(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverZgetrf(
        hipsolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        lu_device_temp.size(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_lu_hipsolver");
  }

  BOBA_CALI_SWITCH("compute_LU", "form_L");
  matrix_L.resize({input.rows(), input.cols()});
  matrix_U.resize({input.rows(), input.cols()});
  auto L_view = matrix_L.view();
  auto U_view = matrix_U.view();
  auto input_view = input.const_view();

  BOBA_CALI_SWITCH("form_L", "form_U");
  matrix_P.resize({input.rows()});
  auto P_view = matrix_P.view();
  auto perms_view = perms.view();

  BOBA_CALI_SWITCH("form_U", "form_P");
  ::boba::loop<execution_space::HIP, 1>(matrix_P.size(),
                                        [=] __boba_host_device__(::boba::index_t i)
  {
    P_view(i) = static_cast<::boba::index_t>(perms_view(i) - 1);
  });

  ::boba::loop<execution_space::HIP, 2>(input.sizes(),
                                        [=] __boba_host_device__(Array<::boba::index_t, 2> rc)
  {
    auto row = rc[0];
    auto col = rc[1];
    auto L_value = PotentiallyComplex<data_t>::value(0);
    auto U_value = PotentiallyComplex<data_t>::value(0);
    auto solver_value = input_view(rc);

    if (col >= row)
    {
      U_value = solver_value;
    }
    if (col < row)
    {
      L_value = solver_value;
    }
    else if (col == row)
    {
      L_value = PotentiallyComplex<data_t>::value(1);
    }

    L_view(rc) = L_value;
    U_view(rc) = U_value;
  });
  BOBA_CALI_END("form_P");
}

/**
 * \brief Forms a lower-triangular Cholesky factorization with hipSOLVER.
 *
 * On return, `matrix_L` stores the lower-triangular factor `L` such that
 * `A = L * L^T`.
 *
 * \see https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/api/lapack.html
 */
template <typename data_t>
void factor_cholesky_hipsolver(
  Matrix<execution_space::HIP, data_t> const& input_in,
  Matrix<execution_space::HIP, data_t>& matrix_L)
{
  BOBA_CALI_BEGIN("initialize_cholesky");
  boba_always_assert_equal(input_in.rows(), input_in.cols(), "Cholesky requires SPD matrix");

  auto input = input_in;

  int chol_device_temp_size = 0;
  BOBA_CALI_SWITCH("initialize_cholesky", "compute_cholesky");
  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert(
      hipsolverSpotrf_bufferSize(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert(
      hipsolverDpotrf_bufferSize(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert(
      hipsolverCpotrf_bufferSize(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert(
      hipsolverZpotrf_bufferSize(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_cholesky_hipsolver");
  }

  Vector<execution_space::HIP, int> info({1});
  info.fill_with_zeros();

  Vector<execution_space::HIP, data_t> chol_device_temp({static_cast<size_t>(chol_device_temp_size)});

  if constexpr (std::is_same_v<data_t, float>)
  {
    hipsolver_assert_info(
      hipsolverSpotrf(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    hipsolver_assert_info(
      hipsolverDpotrf(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    hipsolver_assert_info(
      hipsolverCpotrf(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    hipsolver_assert_info(
      hipsolverZpotrf(
        hipsolver_handle,
        HIPBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_cholesky_hipsolver");
  }

  BOBA_CALI_SWITCH("compute_cholesky", "form_L");
  matrix_L.resize({input.rows(), input.cols()});
  auto input_view = input.const_view();
  auto L_view = matrix_L.view();

  ::boba::loop<execution_space::HIP, 2>(input.sizes(),
                                        [=] __boba_host_device__(Array<::boba::index_t, 2> rc)
  {
    auto row = rc[0];
    auto col = rc[1];
    L_view(rc) = (col <= row) ? input_view(rc) : PotentiallyComplex<data_t>::value(0);
  });
  BOBA_CALI_END("form_L");
}

/**
 * \brief Computes an economical SVD with rocSOLVER and returns `V^T`.
 *
 * The caller owns any singular-value truncation policy and may transpose
 * `matrix_VT` into `V` after inspection.
 *
 * \see https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html#_CPPv416rocsolver_dgesvd14rocblas_handleK13rocblas_svectK13rocblas_svectK11rocblas_intK11rocblas_intPdK11rocblas_intPdPdK11rocblas_intPdK11rocblas_intPdK16rocblas_workmodeP11rocblas_int
 */
template <typename data_t>
void factor_svd_rocsolver(
  Matrix<execution_space::HIP, data_t>& input,
  Matrix<execution_space::HIP, data_t>& matrix_U,
  Vector<execution_space::HIP, real_type_t<data_t>>& singular_values,
  Matrix<execution_space::HIP, data_t>& matrix_VT)
{
  BOBA_CALI_BEGIN("initialize_U_S_V_VT");
  const size_t rows = input.rows();
  const size_t cols = input.cols();
  const size_t svals = boba::min(rows, cols);

  matrix_U.resize({rows, svals});
  singular_values.resize({svals});
  matrix_VT.resize({svals, cols});

  BOBA_CALI_SWITCH("initialize_U_S_V_VT", "rocsolver_svd");
  ::boba::Vector<execution_space::CPU, rocblas_int> svd_info({1});
  ::boba::Vector<execution_space::HIP, real_type_t<data_t>> E_vec({svals});

  if constexpr (std::is_same_v<data_t, float>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_sgesvd(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rows,
        cols,
        input.data(),
        input.rows(),
        singular_values.data(),
        matrix_U.data(),
        matrix_U.rows(),
        matrix_VT.data(),
        matrix_VT.rows(),
        E_vec.data(),
        rocblas_workmode::rocblas_inplace,
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_dgesvd(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rows,
        cols,
        input.data(),
        input.rows(),
        singular_values.data(),
        matrix_U.data(),
        matrix_U.rows(),
        matrix_VT.data(),
        matrix_VT.rows(),
        E_vec.data(),
        rocblas_workmode::rocblas_inplace,
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_cgesvd(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rows,
        cols,
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(input.data()),
        input.rows(),
        singular_values.data(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_U.data()),
        matrix_U.rows(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_VT.data()),
        matrix_VT.rows(),
        E_vec.data(),
        rocblas_workmode::rocblas_inplace,
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_zgesvd(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rows,
        cols,
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(input.data()),
        input.rows(),
        singular_values.data(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_U.data()),
        matrix_U.rows(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_VT.data()),
        matrix_VT.rows(),
        E_vec.data(),
        rocblas_workmode::rocblas_inplace,
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, BDSQR did not converge. i elements of E did not converge to zero");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_svd_rocsolver");
  }
  BOBA_CALI_END("rocsolver_svd");
}

/**
 * \brief Computes a value-range truncated SVD with rocSOLVER and returns `V^T`.
 *
 * Singular values in the inclusive interval
 * `[smallest_singular_value, largest_singular_value]` are returned.
 *
 * \see https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html#rocsolver-type-gesvdx
 */
template <typename data_t>
void factor_svdx_rocsolver(
  Matrix<execution_space::HIP, data_t>& input,
  real_type_t<data_t> smallest_singular_value,
  real_type_t<data_t> largest_singular_value,
  int index_first_singular_value,
  int index_last_singular_value,
  Vector<execution_space::HIP, real_type_t<data_t>>& singular_values,
  Matrix<execution_space::HIP, data_t>& matrix_U,
  Matrix<execution_space::HIP, data_t>& matrix_VT,
  size_t& significant_singular_values)
{
  const size_t rows = input.rows();
  const size_t cols = input.cols();
  const size_t svals = boba::min(rows, cols);

  matrix_U.resize({rows, svals});
  singular_values.resize({svals});
  matrix_VT.resize({svals, cols});

  BOBA_CALI_BEGIN("rocsolver_svd");
  ::boba::Vector<execution_space::HIP, rocblas_int> nonzero_singular_values({1});
  ::boba::Vector<execution_space::HIP, rocblas_int> svd_info({1});
  ::boba::Vector<execution_space::HIP, rocblas_int> ifail({svals});

  if constexpr (std::is_same_v<data_t, float>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_sgesvdx(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rocblas_srange_value,
        rows,
        cols,
        input.data(),
        input.rows(),
        smallest_singular_value,
        largest_singular_value,
        static_cast<rocblas_int>(index_first_singular_value),
        static_cast<rocblas_int>(index_last_singular_value),
        nonzero_singular_values.data(),
        singular_values.data(),
        matrix_U.data(),
        matrix_U.rows(),
        matrix_VT.data(),
        matrix_VT.rows(),
        ifail.data(),
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, i eigenvectors did not converge in BDSVDX; their indices are stored in ifail.");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_dgesvdx(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rocblas_srange_value,
        rows,
        cols,
        input.data(),
        input.rows(),
        smallest_singular_value,
        largest_singular_value,
        static_cast<rocblas_int>(index_first_singular_value),
        static_cast<rocblas_int>(index_last_singular_value),
        nonzero_singular_values.data(),
        singular_values.data(),
        matrix_U.data(),
        matrix_U.rows(),
        matrix_VT.data(),
        matrix_VT.rows(),
        ifail.data(),
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, i eigenvectors did not converge in BDSVDX; their indices are stored in ifail.");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_cgesvdx(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rocblas_srange_value,
        rows,
        cols,
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(input.data()),
        input.rows(),
        smallest_singular_value,
        largest_singular_value,
        static_cast<rocblas_int>(index_first_singular_value),
        static_cast<rocblas_int>(index_last_singular_value),
        nonzero_singular_values.data(),
        singular_values.data(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_U.data()),
        matrix_U.rows(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_VT.data()),
        matrix_VT.rows(),
        ifail.data(),
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, i eigenvectors did not converge in BDSVDX; their indices are stored in ifail.");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    ::boba::detail::rocblas_assert_info(
      rocsolver_zgesvdx(
        boba::detail::_rocblas_handle,
        rocblas_svect::rocblas_svect_singular,
        rocblas_svect::rocblas_svect_singular,
        rocblas_srange_value,
        rows,
        cols,
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(input.data()),
        input.rows(),
        smallest_singular_value,
        largest_singular_value,
        static_cast<rocblas_int>(index_first_singular_value),
        static_cast<rocblas_int>(index_last_singular_value),
        nonzero_singular_values.data(),
        singular_values.data(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_U.data()),
        matrix_U.rows(),
        reinterpret_cast<rocsolver_api_data_t<data_t>*>(matrix_VT.data()),
        matrix_VT.rows(),
        ifail.data(),
        svd_info.data()),
      svd_info.data(),
      "If info = i > 0, i eigenvectors did not converge in BDSVDX; their indices are stored in ifail.");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_svdx_rocsolver");
  }

  significant_singular_values = static_cast<size_t>(nonzero_singular_values.sum_reduce());
  BOBA_CALI_END("rocsolver_svd");
}

/**
 * \brief Solves a square system with hipSOLVER's LU path.
 *
 * `hipsolverDnDgetrf` overwrites \p matrix_A with the LU factors.
 * `hipsolverDnDgetrs` then consumes those factors and overwrites \p matrix_B
 * with the solution.
 *
 * Per ROCm's LAPACK-style documentation, a negative `info` value marks an
 * invalid 1-based argument index. For `Dgetrf`, a positive `info` value means
 * `U(info,info)` is zero and the matrix is singular.
 *
 * \see https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/reference/api/lapack.html
 */

inline void ls_solve_lu_hipsolver(Matrix<execution_space::HIP, double>& matrix_A,
                                  Matrix<execution_space::HIP, double>& matrix_B)
{
  auto n = matrix_A.cols();
  auto nrhs = matrix_B.cols();
  auto lda = matrix_A.rows();
  auto ldb = matrix_B.rows();

  boba_always_assert_ge(matrix_A.rows(), matrix_A.cols(), "Incorrect solver.");

  Vector<execution_space::HIP, int> ipiv({n});
  Vector<execution_space::HIP, int> info({1});

  // 1) geqrf
  int lwork_getrf = 0;
  device_sync();
  hipsolver_assert(
    hipsolverDnDgetrf_bufferSize(
      hipsolver_handle,
      n,
      n,
      matrix_A.data(),
      lda,
      &lwork_getrf));

  Vector<execution_space::HIP, double> work_geqrf({static_cast<::boba::index_t>(lwork_getrf)});

  device_sync();
  hipsolver_assert_info(
    hipsolverDnDgetrf(
      hipsolver_handle,
      n,
      n,
      matrix_A.data(),
      lda,
      work_geqrf.data(),
      ipiv.data(),
      info.data()),
    info.data(),
    "Dgetrf returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid; a positive value means U(info,info) is zero and the matrix is singular.");

  // 2) Solve A X = L U X = B
  device_sync();
  hipsolver_assert_info(
    hipsolverDnDgetrs(
      hipsolver_handle,
      HIPBLAS_OP_N,
      n,
      nrhs,
      matrix_A.data(),
      lda,
      ipiv.data(),
      matrix_B.data(),
      ldb,
      info.data()),
    info.data(),
    "Dgetrs returned a nonzero info flag. A negative value means the corresponding 1-based argument index was invalid.");

  device_sync();
}

inline void ls_solve_lu_hipsolver(Matrix<execution_space::HIP, double> const& matrix_A,
                                  Matrix<execution_space::HIP, double>& matrix_B)
{
  Matrix<execution_space::HIP, double> factor_A(matrix_A);
  ls_solve_lu_hipsolver(factor_A, matrix_B);
}

#else

template <execution_space space>
void ls_solve_qr_hipsolver(Matrix<space, double>& matrix_A,
                           Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_qr_hipsolver requires a HIP build.");
}

template <execution_space space>
void ls_solve_qr_hipsolver(Matrix<space, double> const& matrix_A,
                           Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_qr_hipsolver requires a HIP build.");
}

template <execution_space space>
void ls_solve_lu_hipsolver(Matrix<space, double>& matrix_A,
                           Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_lu_hipsolver requires a HIP build.");
}

template <execution_space space>
void ls_solve_lu_hipsolver(Matrix<space, double> const& matrix_A,
                           Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_lu_hipsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void factor_qr_hipsolver(
  Matrix<space, data_t>& matrix_A,
  Vector<space, data_t>& tau)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(tau);
  boba_error("factor_qr_hipsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void form_q_hipsolver(
  Matrix<space, data_t>& matrix_Q,
  Vector<space, data_t>& tau,
  size_t ranks)
{
  ::boba::detail::ignore(matrix_Q);
  ::boba::detail::ignore(tau);
  ::boba::detail::ignore(ranks);
  boba_error("form_q_hipsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void factor_lu_hipsolver(
  Matrix<space, data_t> const& input_in,
  Matrix<space, data_t>& matrix_L,
  Matrix<space, data_t>& matrix_U,
  PermutationMatrix<space, ::boba::index_t>& matrix_P)
{
  ::boba::detail::ignore(input_in);
  ::boba::detail::ignore(matrix_L);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(matrix_P);
  boba_error("factor_lu_hipsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void factor_cholesky_hipsolver(
  Matrix<space, data_t> const& input_in,
  Matrix<space, data_t>& matrix_L)
{
  ::boba::detail::ignore(input_in);
  ::boba::detail::ignore(matrix_L);
  boba_error("factor_cholesky_hipsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void factor_svd_rocsolver(
  Matrix<space, data_t>& input,
  Matrix<space, data_t>& matrix_U,
  Vector<space, real_type_t<data_t>>& singular_values,
  Matrix<space, data_t>& matrix_VT)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(singular_values);
  ::boba::detail::ignore(matrix_VT);
  boba_error("factor_svd_rocsolver requires a HIP build.");
}

template <execution_space space, typename data_t>
void factor_svdx_rocsolver(
  Matrix<space, data_t>& input,
  real_type_t<data_t> smallest_singular_value,
  real_type_t<data_t> largest_singular_value,
  int index_first_singular_value,
  int index_last_singular_value,
  Vector<space, real_type_t<data_t>>& singular_values,
  Matrix<space, data_t>& matrix_U,
  Matrix<space, data_t>& matrix_VT,
  size_t& significant_singular_values)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(smallest_singular_value);
  ::boba::detail::ignore(largest_singular_value);
  ::boba::detail::ignore(index_first_singular_value);
  ::boba::detail::ignore(index_last_singular_value);
  ::boba::detail::ignore(singular_values);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(matrix_VT);
  ::boba::detail::ignore(significant_singular_values);
  boba_error("factor_svdx_rocsolver requires a HIP build.");
}

#endif

} // namespace detail
} // namespace boba
