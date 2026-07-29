// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_CUDA_LIBS
#include "BOBA/backends/boba_cuda.hpp"

#include <cusolverDn.h>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_CUDA_LIBS

extern cusolverDnHandle_t cusolver_handle;

// -----------------------------------------------------
// Lib Debugging
// -----------------------------------------------------

/**
 * @brief Expands a cuSOLVER status code into assertion failures with readable labels.
 * @param error cuSOLVER status code to decode.
 */
inline void cusolver_error_codes_(cusolverStatus_t error)
{
  boba_always_assert(error != CUSOLVER_STATUS_NOT_INITIALIZED, "CUSOLVER_STATUS_NOT_INITIALIZED ");
  boba_always_assert(error != CUSOLVER_STATUS_ALLOC_FAILED, "CUSOLVER_STATUS_ALLOC_FAILED");
  boba_always_assert(error != CUSOLVER_STATUS_INVALID_VALUE, "CUSOLVER_STATUS_INVALID_VALUE ");
  boba_always_assert(error != CUSOLVER_STATUS_ARCH_MISMATCH, "CUSOLVER_STATUS_ARCH_MISMATCH ");
  boba_always_assert(error != CUSOLVER_STATUS_EXECUTION_FAILED, "CUSOLVER_STATUS_EXECUTION_FAILED ");
  boba_always_assert(error != CUSOLVER_STATUS_INTERNAL_ERROR, "CUSOLVER_STATUS_INTERNAL_ERROR");
  boba_always_assert(error != CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED ");
  boba_always_assert(error != CUSOLVER_STATUS_SUCCESS, "cusolver error");
}

#define cusolver_assert(a) cusolver_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a cuSOLVER error and terminates.
 * @param error cuSOLVER status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cusolver_assert_(
  cusolverStatus_t error,
  const std::string& call,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  if (error == CUSOLVER_STATUS_SUCCESS)
  {
    return;
  }
  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  cusolver_error_codes_(error);
  exit(1);
}

// Pass in solver 'info' flag
#define cusolver_assert_info(a, b, c) cusolver_assert_info_(a, #a, b, c, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a cuSOLVER error or nonzero info flag and terminates.
 * @param error cuSOLVER status code to check.
 * @param call Source expression that produced the error.
 * @param info Device pointer to the backend info flag.
 * @param info_meaning Description of the info flag semantics.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cusolver_assert_info_(
  cusolverStatus_t error,
  const std::string& call,
  int* info,
  const std::string& info_meaning,
  size_t line,
  const std::string& function,
  const std::string& file)
{
  int host_info = 0;
  cuda_memcpy(&host_info, info, 1);

  if ((error == CUSOLVER_STATUS_SUCCESS) and (host_info == 0))
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  std::cout << "Info flag: " << host_info << std::endl;
  std::cout << "Info meaning: " << info_meaning << std::endl;

  if (error != CUSOLVER_STATUS_SUCCESS)
  {
    cusolver_error_codes_(error);
  }
  boba_error("cuSOLVER reported a nonzero info flag");
}

/**
 * \brief Solves a least-squares system with cuSOLVER's QR path.
 *
 * cuSOLVER's `cusolverDnDgeqrf` factorization overwrites \p matrix_A with the
 * packed Householder representation. The right-hand side \p matrix_B remains
 * in-place because `cusolverDnDormqr` and `cublasDtrsm` overwrite it with the
 * solution.
 *
 * According to NVIDIA's LAPACK-style interface documentation, a negative
 * `info` value from `Dgeqrf` or `Dormqr` means the corresponding 1-based
 * argument index was invalid.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-lt-t-gt-geqrf
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-lt-t-gt-ormqr
 */

inline void ls_solve_qr_cusolver(Matrix<execution_space::CUDA, double>& matrix_A,
                                 Matrix<execution_space::CUDA, double>& matrix_B)
{
  auto m = matrix_A.rows();
  auto n = matrix_A.cols();
  auto nrhs = matrix_B.cols();
  auto lda = matrix_A.rows();
  auto ldb = matrix_B.rows();

  boba_always_assert_ge(matrix_A.rows(), matrix_A.cols(), "Incorrect solver.");

  Vector<execution_space::CUDA, double> tau({n});
  Vector<execution_space::CUDA, int> info({1});

  // 1) geqrf
  int lwork_geqrf = 0;
  cusolver_assert(
    cusolverDnDgeqrf_bufferSize(
      cusolver_handle,
      m,
      n,
      matrix_A.data(),
      lda,
      &lwork_geqrf));

  Vector<execution_space::CUDA, double> work_geqrf({static_cast<::boba::index_t>(lwork_geqrf)});

  device_sync();
  cusolver_assert_info(
    cusolverDnDgeqrf(
      cusolver_handle,
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

  // 2) B <- Q^T B
  int lwork_ormqr = 0;
  device_sync();
  cusolver_assert(
    cusolverDnDormqr_bufferSize(
      cusolver_handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_OP_T,
      m,
      nrhs,
      n,
      matrix_A.data(),
      lda,
      tau.data(),
      matrix_B.data(),
      ldb,
      &lwork_ormqr));

  Vector<execution_space::CUDA, double> work_ormqr({static_cast<::boba::index_t>(lwork_ormqr)});

  device_sync();
  cusolver_assert_info(
    cusolverDnDormqr(
      cusolver_handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_OP_T,
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

  // 3) Solve R X = B(1:n,:)
  const double one = 1.0;

  device_sync();
  cublas_assert(
    cublasDtrsm(
      cublas_handle,
      CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_UPPER,
      CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT,
      n,
      nrhs,
      &one,
      matrix_A.data(),
      lda,
      matrix_B.data(),
      ldb));

  device_sync();
}

inline void ls_solve_qr_cusolver(Matrix<execution_space::CUDA, double> const& matrix_A,
                                 Matrix<execution_space::CUDA, double>& matrix_B)
{
  Matrix<execution_space::CUDA, double> factor_A(matrix_A);
  ls_solve_qr_cusolver(factor_A, matrix_B);
}

/**
 * \brief Computes the in-place QR panel factorization used by BoBa's CUDA QR.
 *
 * On return, \p matrix_A contains the packed Householder representation
 * produced by cuSOLVER and \p tau stores the reflector coefficients.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDnXgeqrf
 */
template <typename data_t>
void factor_qr_cusolver(
  Matrix<execution_space::CUDA, data_t>& matrix_A,
  Vector<execution_space::CUDA, data_t>& tau)
{
  auto rows = matrix_A.rows();
  auto cols = matrix_A.cols();
  BOBA_CALI_BEGIN("cusolver_get_qr_buffer_sizes");
  auto cuda_type = cuda_data_t_v<data_t>;

  cusolverDnParams_t qr_params{};
  size_t qr_device_work_size = 0_z;
  size_t qr_host_work_size = 0_z;

  cusolver_assert(cusolverDnXgeqrf_bufferSize(
    cusolver_handle,
    qr_params,
    static_cast<int64_t>(rows),
    static_cast<int64_t>(cols),
    cuda_type,
    matrix_A.data(),
    static_cast<int64_t>(rows),
    cuda_type,
    tau.data(),
    cuda_type,
    &qr_device_work_size,
    &qr_host_work_size));

  BOBA_CALI_SWITCH("cusolver_get_qr_buffer_sizes", "allocate_buffers");
  Vector<execution_space::CUDA, unsigned char> qr_device_work({qr_device_work_size});
  Vector<execution_space::CPU, unsigned char> qr_host_work({qr_host_work_size});
  Vector<execution_space::CUDA, int> qr_info({1_z});

  BOBA_CALI_SWITCH("allocate_buffers", "cusolver_qr");
  cusolver_assert_info(
    cusolverDnXgeqrf(
      cusolver_handle,
      qr_params,
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      cuda_type,
      matrix_A.data(),
      static_cast<int64_t>(rows),
      cuda_type,
      tau.data(),
      cuda_type,
      qr_device_work.data(),
      qr_device_work.size(),
      qr_host_work.data(),
      qr_host_work.size(),
      qr_info.data()),
    qr_info.data(),
    "If info = -i, the i-th parameter is wrong (not counting handle).");
  BOBA_CALI_END("cusolver_qr");
}

/**
 * \brief Expands packed Householder data into the reduced Q factor on CUDA.
 *
 * \param matrix_Q Matrix containing the packed Householder vectors on entry and
 * the reduced orthonormal factor on exit.
 * \param tau Reflector coefficients returned by factor_qr_cusolver().
 * \param ranks Number of reflectors to apply.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-orgqr
 */
inline void form_q_cusolver(
  Matrix<execution_space::CUDA, double>& matrix_Q,
  Vector<execution_space::CUDA, double>& tau,
  size_t ranks)
{
  int q_device_work_size = 0;
  BOBA_CALI_BEGIN("cusolverDnDorgqr_bufferSize");

  cusolver_assert(cusolverDnDorgqr_bufferSize(
    cusolver_handle,
    static_cast<int64_t>(matrix_Q.rows()),
    static_cast<int64_t>(matrix_Q.cols()),
    static_cast<int64_t>(ranks),
    matrix_Q.data(),
    static_cast<int64_t>(matrix_Q.rows()),
    tau.data(),
    &q_device_work_size));

  Vector<execution_space::CUDA, double> q_device_work({static_cast<size_t>(q_device_work_size)});
  Vector<execution_space::CUDA, int> q_info({1_z});

  BOBA_CALI_SWITCH("cusolverDnDorgqr_bufferSize", "cusolverDnDorgqr");
  cusolver_assert_info(
    cusolverDnDorgqr(
      cusolver_handle,
      static_cast<int64_t>(matrix_Q.rows()),
      static_cast<int64_t>(matrix_Q.cols()),
      static_cast<int64_t>(ranks),
      matrix_Q.data(),
      static_cast<int64_t>(matrix_Q.rows()),
      tau.data(),
      q_device_work.data(),
      q_device_work.size(),
      q_info.data()),
    q_info.data(),
    "If output parameter devInfo < 0 (less than zero), the i-th parameter is wrong (not counting handle).");
  BOBA_CALI_END("cusolverDnDorgqr");
}

inline void form_q_cusolver(
  Matrix<execution_space::CUDA, float>& matrix_Q,
  Vector<execution_space::CUDA, float>& tau,
  size_t ranks)
{
  int q_device_work_size = 0;
  BOBA_CALI_BEGIN("cusolverDnSorgqr_bufferSize");
  cusolver_assert(cusolverDnSorgqr_bufferSize(
    cusolver_handle,
    static_cast<int64_t>(matrix_Q.rows()),
    static_cast<int64_t>(matrix_Q.cols()),
    static_cast<int64_t>(ranks),
    matrix_Q.data(),
    static_cast<int64_t>(matrix_Q.rows()),
    tau.data(),
    &q_device_work_size));

  Vector<execution_space::CUDA, float> q_device_work({static_cast<size_t>(q_device_work_size)});
  Vector<execution_space::CUDA, int> q_info({1_z});

  BOBA_CALI_SWITCH("cusolverDnSorgqr_bufferSize", "cusolverDnSorgqr");
  cusolver_assert_info(
    cusolverDnSorgqr(
      cusolver_handle,
      static_cast<int64_t>(matrix_Q.rows()),
      static_cast<int64_t>(matrix_Q.cols()),
      static_cast<int64_t>(ranks),
      matrix_Q.data(),
      static_cast<int64_t>(matrix_Q.rows()),
      tau.data(),
      q_device_work.data(),
      q_device_work.size(),
      q_info.data()),
    q_info.data(),
    "If output parameter devInfo < 0 (less than zero), the i-th parameter is wrong (not counting handle).");
  BOBA_CALI_END("cusolverDnSorgqr");
}

inline void form_q_cusolver(
  Matrix<execution_space::CUDA, complex<double>>& matrix_Q,
  Vector<execution_space::CUDA, complex<double>>& tau,
  size_t ranks)
{
  int q_device_work_size = 0;
  BOBA_CALI_BEGIN("cusolverDnZorgqr_bufferSize");
  cusolver_assert(cusolverDnZungqr_bufferSize(
    cusolver_handle,
    static_cast<int64_t>(matrix_Q.rows()),
    static_cast<int64_t>(matrix_Q.cols()),
    static_cast<int64_t>(ranks),
    matrix_Q.data(),
    static_cast<int64_t>(matrix_Q.rows()),
    tau.data(),
    &q_device_work_size));

  Vector<execution_space::CUDA, complex<double>> q_device_work({static_cast<size_t>(q_device_work_size)});
  Vector<execution_space::CUDA, int> q_info({1_z});

  BOBA_CALI_SWITCH("cusolverDnZorgqr_bufferSize", "cusolverDnZorgqr");
  cusolver_assert_info(
    cusolverDnZungqr(
      cusolver_handle,
      static_cast<int64_t>(matrix_Q.rows()),
      static_cast<int64_t>(matrix_Q.cols()),
      static_cast<int64_t>(ranks),
      matrix_Q.data(),
      static_cast<int64_t>(matrix_Q.rows()),
      tau.data(),
      q_device_work.data(),
      q_device_work.size(),
      q_info.data()),
    q_info.data(),
    "If output parameter devInfo < 0 (less than zero), the i-th parameter is wrong (not counting handle).");
  BOBA_CALI_END("cusolverDnZorgqr");
}

inline void form_q_cusolver(
  Matrix<execution_space::CUDA, complex<float>>& matrix_Q,
  Vector<execution_space::CUDA, complex<float>>& tau,
  size_t ranks)
{
  int q_device_work_size = 0;
  BOBA_CALI_BEGIN("cusolverDnCungqr_bufferSize");
  cusolver_assert(cusolverDnCungqr_bufferSize(
    cusolver_handle,
    static_cast<int64_t>(matrix_Q.rows()),
    static_cast<int64_t>(matrix_Q.cols()),
    static_cast<int64_t>(ranks),
    matrix_Q.data(),
    static_cast<int64_t>(matrix_Q.rows()),
    tau.data(),
    &q_device_work_size));

  Vector<execution_space::CUDA, complex<float>> q_device_work({static_cast<size_t>(q_device_work_size)});
  Vector<execution_space::CUDA, int> q_info({1_z});

  BOBA_CALI_SWITCH("cusolverDnCungqr_bufferSize", "cusolverDnCungqr");
  cusolver_assert_info(
    cusolverDnCungqr(
      cusolver_handle,
      static_cast<int64_t>(matrix_Q.rows()),
      static_cast<int64_t>(matrix_Q.cols()),
      static_cast<int64_t>(ranks),
      matrix_Q.data(),
      static_cast<int64_t>(matrix_Q.rows()),
      tau.data(),
      q_device_work.data(),
      q_device_work.size(),
      q_info.data()),
    q_info.data(),
    "If output parameter devInfo < 0 (less than zero), the i-th parameter is wrong (not counting handle).");
  BOBA_CALI_END("cusolverDnCungqr");
}

/**
 * \brief Forms a partial-pivot LU factorization with cuSOLVER.
 *
 * On return, `P * A = L * U`.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDn-t-getrf
 */
template <typename data_t>
void factor_lu_cusolver(
  Matrix<execution_space::CUDA, data_t> const& input_in,
  Matrix<execution_space::CUDA, data_t>& matrix_L,
  Matrix<execution_space::CUDA, data_t>& matrix_U,
  PermutationMatrix<execution_space::CUDA, ::boba::index_t>& matrix_P)
{
  BOBA_CALI_BEGIN("initialize_lu");
  auto input = input_in;
  int lu_device_temp_size = 0;
  if constexpr (std::is_same_v<data_t, float>)
  {
    cusolver_assert(
      cusolverDnSgetrf_bufferSize(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    cusolver_assert(
      cusolverDnDgetrf_bufferSize(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    cusolver_assert(
      cusolverDnCgetrf_bufferSize(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cusolver_assert(
      cusolverDnZgetrf_bufferSize(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        &lu_device_temp_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_lu_cusolver");
  }

  Vector<execution_space::CUDA, int> info({1});
  info.fill_with_zeros();

  Vector<execution_space::CUDA, data_t> lu_device_temp({static_cast<size_t>(lu_device_temp_size)});
  Vector<execution_space::CUDA, int> perms({input.rows()});

  if constexpr (std::is_same_v<data_t, float>)
  {
    cusolver_assert_info(
      cusolverDnSgetrf(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    cusolver_assert_info(
      cusolverDnDgetrf(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    cusolver_assert_info(
      cusolverDnCgetrf(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cusolver_assert_info(
      cusolverDnZgetrf(
        cusolver_handle,
        input.rows(),
        input.cols(),
        input.data(),
        input.rows(),
        lu_device_temp.data(),
        perms.data(),
        info.data()),
      info.data(),
      "If info = -i, the i-th parameter is wrong (not counting handle).");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_lu_cusolver");
  }

  BOBA_CALI_SWITCH("compute_LU", "form_L");
  matrix_L.resize({input.rows(), input.cols()});
  matrix_U.resize({input.rows(), input.cols()});
  auto L_view = matrix_L.view();
  auto U_view = matrix_U.view();
  auto input_view = input.const_view();

  BOBA_CALI_SWITCH("form_L", "form_U");
  matrix_P.resize(::boba::filled_array<1>(static_cast<::boba::index_t>(input.rows())));
  auto P_view = matrix_P.view();
  auto perms_view = perms.view();

  BOBA_CALI_SWITCH("form_U", "form_P");
  ::boba::loop<execution_space::CUDA, 1>(matrix_P.size(),
                                         [=] __boba_host_device__(::boba::index_t i)
  {
    P_view(i) = static_cast<::boba::index_t>(perms_view(i) - 1);
  });

  ::boba::loop<execution_space::CUDA, 2>(input.sizes(),
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
 * \brief Forms a lower-triangular Cholesky factorization with cuSOLVER.
 *
 * On return, `matrix_L` stores the lower-triangular factor `L` such that
 * `A = L * L^T`.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverDn-t-potrf
 */
template <typename data_t>
void factor_cholesky_cusolver(
  Matrix<execution_space::CUDA, data_t> const& input_in,
  Matrix<execution_space::CUDA, data_t>& matrix_L)
{
  BOBA_CALI_BEGIN("initialize_cholesky");
  boba_always_assert_equal(input_in.rows(), input_in.cols(), "Cholesky requires SPD matrix");

  auto input = input_in;
  int chol_device_temp_size = 0;

  BOBA_CALI_SWITCH("initialize_cholesky", "compute_cholesky");
  if constexpr (std::is_same_v<data_t, float>)
  {
    cusolver_assert(
      cusolverDnSpotrf_bufferSize(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, double>)
  {
    cusolver_assert(
      cusolverDnDpotrf_bufferSize(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<float>>)
  {
    cusolver_assert(
      cusolverDnCpotrf_bufferSize(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cusolver_assert(
      cusolverDnZpotrf_bufferSize(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        &chol_device_temp_size));
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_cholesky_cusolver");
  }

  Vector<execution_space::CUDA, int> info({1});
  info.fill_with_zeros();

  Vector<execution_space::CUDA, data_t> chol_device_temp({static_cast<size_t>(chol_device_temp_size)});

  if constexpr (std::is_same_v<data_t, float>)
  {
    cusolver_assert_info(
      cusolverDnSpotrf(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
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
    cusolver_assert_info(
      cusolverDnDpotrf(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
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
    cusolver_assert_info(
      cusolverDnCpotrf(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th argument is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else if constexpr (std::is_same_v<data_t, complex<double>>)
  {
    cusolver_assert_info(
      cusolverDnZpotrf(
        cusolver_handle,
        CUBLAS_FILL_MODE_LOWER,
        input.rows(),
        input.data(),
        input.rows(),
        chol_device_temp.data(),
        chol_device_temp_size,
        info.data()),
      info.data(),
      "If info = -i, the i-th argument is wrong (not counting handle). If info > 0, the leading minor is not positive definite.");
  }
  else
  {
    boba_error("Build error! Unsupported data type for factor_cholesky_cusolver");
  }

  BOBA_CALI_SWITCH("compute_cholesky", "form_L");
  matrix_L.resize({input.rows(), input.cols()});
  auto input_view = input.const_view();
  auto L_view = matrix_L.view();

  ::boba::loop<execution_space::CUDA, 2>(input.sizes(),
                                         [=] __boba_host_device__(Array<::boba::index_t, 2> rc)
  {
    auto row = rc[0];
    auto col = rc[1];
    L_view(rc) = (col <= row) ? input_view(rc) : PotentiallyComplex<data_t>::value(0);
  });
  BOBA_CALI_END("form_L");
}

/**
 * \brief Computes an economical SVD with cuSOLVER and returns `V^T`.
 *
 * The caller owns any singular-value truncation policy and may transpose
 * `matrix_VT` into `V` after inspection.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDnXgesvd
 */
template <typename data_t>
void factor_svd_cusolver(
  Matrix<execution_space::CUDA, data_t>& input,
  Matrix<execution_space::CUDA, data_t>& matrix_U,
  Vector<execution_space::CUDA, real_type_t<data_t>>& singular_values,
  Matrix<execution_space::CUDA, data_t>& matrix_VT)
{
  BOBA_CALI_BEGIN("initialize_U_S_V_VT");
  const size_t rows = input.rows();
  const size_t cols = input.cols();
  const size_t svals = boba::min(rows, cols);
  auto cuda_type = cuda_data_t_v<data_t>;
  auto real_cuda_type = cuda_data_t_v<real_type_t<data_t>>;

  matrix_U.resize({rows, svals});
  singular_values.resize({svals});
  matrix_VT.resize({svals, cols});

  signed char U_opt = 'S';
  signed char VT_opt = 'S';

  cusolverDnParams_t params{};
  size_t device_work_size = 0_z;
  size_t host_work_size = 0_z;

  BOBA_CALI_SWITCH("initialize_U_S_V_VT", "cusolver_get_buffer_size");
  cusolver_assert(cusolverDnXgesvd_bufferSize(
    cusolver_handle,
    params,
    U_opt,
    VT_opt,
    static_cast<int64_t>(rows),
    static_cast<int64_t>(cols),
    cuda_type,
    input.data(),
    static_cast<int64_t>(input.rows()),
    real_cuda_type,
    singular_values.data(),
    cuda_type,
    matrix_U.data(),
    static_cast<int64_t>(matrix_U.rows()),
    cuda_type,
    matrix_VT.data(),
    static_cast<int64_t>(matrix_VT.rows()),
    cuda_type,
    &device_work_size,
    &host_work_size));

  BOBA_CALI_SWITCH("cusolver_get_buffer_size", "allocate_buffers");
  Vector<execution_space::CUDA, unsigned char> device_work({static_cast<size_t>(device_work_size)});
  Vector<execution_space::CPU, unsigned char> host_work({static_cast<size_t>(host_work_size)});
  Vector<execution_space::CUDA, int> svd_info({1});

  BOBA_CALI_SWITCH("allocate_buffers", "cusolver_svd");
  cusolver_assert_info(
    cusolverDnXgesvd(
      cusolver_handle,
      params,
      U_opt,
      VT_opt,
      static_cast<int64_t>(rows),
      static_cast<int64_t>(cols),
      cuda_type,
      input.data(),
      static_cast<int64_t>(input.rows()),
      real_cuda_type,
      singular_values.data(),
      cuda_type,
      matrix_U.data(),
      static_cast<int64_t>(matrix_U.rows()),
      cuda_type,
      matrix_VT.data(),
      static_cast<int64_t>(matrix_VT.rows()),
      cuda_type,
      device_work.data(),
      device_work.size(),
      host_work.data(),
      host_work.size(),
      svd_info.data()),
    svd_info.data(),
    "info < 0, the i-th parameter is wrong (not counting handle). if bdsqr did not converge, devInfo specifies how many superdiagonals of an intermediate bidiagonal form did not converge to zero.");
  BOBA_CALI_END("cusolver_svd");
}

/**
 * \brief Solves a square system with cuSOLVER's LU path.
 *
 * `cusolverDnDgetrf` overwrites \p matrix_A with the LU factors.
 * `cusolverDnDgetrs` then consumes those factors and overwrites \p matrix_B
 * with the solution.
 *
 * Per NVIDIA's documentation, a negative `info` value marks an invalid 1-based
 * argument index. For `Dgetrf`, a positive `info` value means `U(info,info)` is
 * exactly zero and the matrix is singular.
 *
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-lt-t-gt-getrf
 * \see https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-lt-t-gt-getrs
 */

inline void ls_solve_lu_cusolver(Matrix<execution_space::CUDA, double>& matrix_A,
                                 Matrix<execution_space::CUDA, double>& matrix_B)
{
  auto n = matrix_A.cols();
  auto nrhs = matrix_B.cols();
  auto lda = matrix_A.rows();
  auto ldb = matrix_B.rows();

  boba_always_assert_ge(matrix_A.rows(), matrix_A.cols(), "Incorrect solver.");

  Vector<execution_space::CUDA, int> ipiv({n});
  Vector<execution_space::CUDA, int> info({1});

  // 1) geqrf
  int lwork_getrf = 0;
  device_sync();
  cusolver_assert(
    cusolverDnDgetrf_bufferSize(
      cusolver_handle,
      n,
      n,
      matrix_A.data(),
      lda,
      &lwork_getrf));

  Vector<execution_space::CUDA, double> work_geqrf({static_cast<::boba::index_t>(lwork_getrf)});

  device_sync();
  cusolver_assert_info(
    cusolverDnDgetrf(
      cusolver_handle,
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
  cusolver_assert_info(
    cusolverDnDgetrs(
      cusolver_handle,
      CUBLAS_OP_N,
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

inline void ls_solve_lu_cusolver(Matrix<execution_space::CUDA, double> const& matrix_A,
                                 Matrix<execution_space::CUDA, double>& matrix_B)
{
  Matrix<execution_space::CUDA, double> factor_A(matrix_A);
  ls_solve_lu_cusolver(factor_A, matrix_B);
}

#else

template <execution_space space>
void ls_solve_qr_cusolver(Matrix<space, double>& matrix_A,
                          Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_qr_cusolver requires a CUDA build.");
}

template <execution_space space>
void ls_solve_qr_cusolver(Matrix<space, double> const& matrix_A,
                          Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_qr_cusolver requires a CUDA build.");
}

template <execution_space space>
void ls_solve_lu_cusolver(Matrix<space, double>& matrix_A,
                          Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_lu_cusolver requires a CUDA build.");
}

template <execution_space space>
void ls_solve_lu_cusolver(Matrix<space, double> const& matrix_A,
                          Matrix<space, double>& matrix_B)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  boba_error("ls_solve_lu_cusolver requires a CUDA build.");
}

template <execution_space space, typename data_t>
void factor_qr_cusolver(
  Matrix<space, data_t>& matrix_A,
  Vector<space, data_t>& tau)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(tau);
  boba_error("factor_qr_cusolver requires a CUDA build.");
}

template <execution_space space, typename data_t>
void form_q_cusolver(
  Matrix<space, data_t>& matrix_Q,
  Vector<space, data_t>& tau,
  size_t ranks)
{
  ::boba::detail::ignore(matrix_Q);
  ::boba::detail::ignore(tau);
  ::boba::detail::ignore(ranks);
  boba_error("form_q_cusolver requires a CUDA build.");
}

template <execution_space space, typename data_t>
void factor_lu_cusolver(
  Matrix<space, data_t> const& input_in,
  Matrix<space, data_t>& matrix_L,
  Matrix<space, data_t>& matrix_U,
  PermutationMatrix<space, ::boba::index_t>& matrix_P)
{
  ::boba::detail::ignore(input_in);
  ::boba::detail::ignore(matrix_L);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(matrix_P);
  boba_error("factor_lu_cusolver requires a CUDA build.");
}

template <execution_space space, typename data_t>
void factor_cholesky_cusolver(
  Matrix<space, data_t> const& input_in,
  Matrix<space, data_t>& matrix_L)
{
  ::boba::detail::ignore(input_in);
  ::boba::detail::ignore(matrix_L);
  boba_error("factor_cholesky_cusolver requires a CUDA build.");
}

template <execution_space space, typename data_t>
void factor_svd_cusolver(
  Matrix<space, data_t>& input,
  Matrix<space, data_t>& matrix_U,
  Vector<space, real_type_t<data_t>>& singular_values,
  Matrix<space, data_t>& matrix_VT)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(singular_values);
  ::boba::detail::ignore(matrix_VT);
  boba_error("factor_svd_cusolver requires a CUDA build.");
}

#endif

} // namespace detail
} // namespace boba
