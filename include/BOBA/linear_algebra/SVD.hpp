// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <stdexcept>

#ifdef BOBA_ENABLE_APPLE
#include <Accelerate/Accelerate.h>
#endif

namespace boba
{
/**
 * \brief Abstraction class for the singular value decomposition.
 */

template <execution_space space, typename data_t>
struct SVD
{
  using real_data_t = real_type_t<data_t>;

  enum struct svd_types : size_t
  {
    economical,
    truncated,
    randomized,
  };

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> U;
  boba::Vector<space, real_data_t> S;
  boba::Matrix<space, data_t> V;

  size_t significant_singular_values = 0;

  svd_types svd_type = svd_types::economical;

  real_data_t tolerance_relative = real_data_t(50) * std::numeric_limits<real_data_t>::epsilon();
  real_data_t tolerance_absolute = real_data_t(100) * std::numeric_limits<real_data_t>::denorm_min();
  size_t max_kept_singular_values = highest_value<size_t>(); // default to keeping all values

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  SVD()
  {
    U.rename("U");
    S.rename("S");
    V.rename("V");
    parse_type_from_env();
  }

  /**
   * \brief copy constructor
   */
  SVD(SVD const&) = default;

  /**
   * \brief move constructor
   */
  SVD(SVD&&) = default;

  /**
   * \brief copy assignment operator
   */
  SVD& operator=(SVD const&) = default;

  /**
   * \brief move assignment operator
   */
  SVD& operator=(SVD&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Factorization to copy from.
   */
  template <execution_space rhs_space>
  SVD(SVD<rhs_space, data_t> const& rhs)
      : U(rhs.U),
        S(rhs.S),
        V(rhs.V),
        significant_singular_values(rhs.significant_singular_values),
        svd_type(rhs.svd_type),
        tolerance_relative(rhs.tolerance_relative),
        tolerance_absolute(rhs.tolerance_absolute),
        max_kept_singular_values(rhs.max_kept_singular_values)
  {
    parse_type_from_env();
  }

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Factorization to copy from.
   * \return This factorization after assignment.
   */
  template <execution_space rhs_space>
  SVD& operator=(SVD<rhs_space, data_t> const& rhs)
  {
    U = rhs.U;
    S = rhs.S;
    V = rhs.V;
    significant_singular_values = rhs.significant_singular_values;
    svd_type = rhs.svd_type;
    tolerance_relative = rhs.tolerance_relative;
    tolerance_absolute = rhs.tolerance_absolute;
    max_kept_singular_values = rhs.max_kept_singular_values;
    parse_type_from_env();
    return *this;
  }

  /**
   * \brief destructor
   */
  ~SVD() = default;

  /**
   * \brief Reads `svd_types` from the environment.
   */
  void parse_type_from_env()
  {
    svd_type = svd_types::economical;

    if (env_match("svd_types", "truncated") and (space == execution_space::HIP))
    {
      svd_type = svd_types::truncated;
    }

    if (env_match("svd_types", "randomized"))
    {
      svd_type = svd_types::randomized;
    }
  }

  // -------------------------------------------------------------------------------------
  //  Architecture-agnostic implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reconstructs a matrix from the stored SVD factors.
   * \return Matrix reconstructed as `U * diag(S) * V^H`.
   */
  [[nodiscard]]
  Matrix<space, data_t> reform_matrix()
  {
    BOBA_CALI_MARK
    checkpoint();
    auto US = U;
    apply_as_diagonal_right_in_place(this->S, US);
    checkpoint();
    auto Vt = V.conjugate_transpose();
    checkpoint();
    auto USVt = US * Vt;
    return USVt;
  }

  /**
   * \brief Truncates the stored factors to a requested number of singular values.
   * \param singular_values Number of singular values and corresponding vectors to keep.
   */
  void truncate(index_t singular_values)
  {
    BOBA_CALI_MARK
    checkpoint();
    boba_always_assert_le(singular_values, U.cols(), "May only shrink");
    boba_always_assert_le(singular_values, V.cols(), "May only shrink");
    U.resize({U.rows(), singular_values});
    V.resize({V.rows(), singular_values});
    S.resize({singular_values});
  }

  /**
   * \brief Discards singular values below the configured significance thresholds.
   */
  void truncate_singular_values()
  {
    BOBA_CALI_MARK
    checkpoint();
    size_t num_values_upper_bound = S.size();
    if (max_kept_singular_values > 0)
    {
      num_values_upper_bound = ::boba::min(num_values_upper_bound, max_kept_singular_values);
    }

    auto singular_values_view = S.const_view();
    // TODO<org> change tolerance_relative to m_tolerance_relative
    real_data_t _tolerance_relative = tolerance_relative;
    real_data_t _tolerance_absolute = tolerance_absolute;

    //
    // Search for the largest index among significant singular values
    // Uses a reduction loop
    //
    auto last_significant_value_index = 0_z;

    ::boba::max_reduce<space>(last_significant_value_index, index_t(0), num_values_upper_bound, [=] __boba_host_device__(size_t index, max_reducer_operator<size_t>& local_last_significant_value_index)
    {
      auto max_sigma = singular_values_view({0});
      auto sigma = singular_values_view({index});
      bool relative_check = sigma > _tolerance_relative * max_sigma;
      bool absolute_check = sigma > _tolerance_absolute;
      if (relative_check and absolute_check)
      {
        local_last_significant_value_index.max(index);
      }
    });
    checkpoint();
    significant_singular_values = last_significant_value_index + 1;

    S.resize({significant_singular_values});
    checkpoint();
  }

  /**
   * \brief Computes an SVD of the input matrix.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK

    checkpoint();
    //
    // Handle transpose case
    //
    bool need_transpose = false;

    if constexpr (space == execution_space::CUDA)
    {
      need_transpose = input.rows() < input.cols();
    }

    boba::Matrix<space, data_t> svd_input;

    if (need_transpose)
    {
      // A^T = V*S*U^T
      svd_input = input.transpose();
    }
    else
    {
      svd_input = input;
    }
    svd_input.rename("svd_input");

    //
    // Compute
    //
    compute(svd_input);

    checkpoint();
    // A = U*S*V^T
    // or
    // A^T = V*S*U^T as U*S*V^T

    if (need_transpose)
    {
      checkpoint();
      // correct for transpose
      boba::Matrix<space, data_t> temp(V);
      V = U;
      U = temp;
    }

    S.resize({significant_singular_values});
    U.resize({U.rows(), significant_singular_values});
    V.resize({V.rows(), significant_singular_values});

    checkpoint();
  }

  void compute(Matrix<execution_space::CPU, data_t> svd_input)
  {
    //
    // Compute SVD
    //
    checkpoint();
    switch (svd_type)
    {
    //
    // Full/thin SVDs
    //
    case svd_types::economical:
    {
#ifdef BOBA_ENABLE_APPLE
      operator_svd_apple(svd_input);
#else
      operator_svd_eigen(svd_input);
#endif
      break;
    }
    //
    // Truncated SVDs
    //
    case svd_types::truncated:
    {
      boba_error("truncated SVD not yet implemented for eigen");
      break;
    }
    case svd_types::randomized:
    {
      operator_svd_randomized(svd_input);
      break;
    }
    default:
    {
      boba_error("undefined svd_type");
      break;
    }
    }
  }

  void compute(Matrix<execution_space::CUDA, data_t> svd_input)
  {
    //
    // Compute SVD
    //
    checkpoint();
    switch (svd_type)
    {
    //
    // Full/thin SVDs
    //
    case svd_types::economical:
    {
      operator_svd_cuda(svd_input);
      break;
    }
    //
    // Truncated SVDs
    //
    case svd_types::truncated:
    {
      boba_error("truncated SVD not yet implemented for CUDA");
      break;
    }
    case svd_types::randomized:
    {
      operator_svd_randomized(svd_input);
      break;
    }
    default:
    {
      boba_error("undefined svd_type");
      break;
    }
    }
  }

  void compute(Matrix<execution_space::HIP, data_t> svd_input)
  {
    //
    // Compute SVD
    //

    switch (svd_type)
    {
    //
    // Full/thin SVDs
    //
    case svd_types::economical:
    {
      operator_svdx_rocm(svd_input);
      //  svd rocm needs repairs
      // operator_svd_rocm(svd_input);
      break;
    }
    //
    // Truncated SVDs
    //
    case svd_types::truncated:
    {
      operator_svdx_rocm(svd_input);
      break;
    }
    case svd_types::randomized:
    {
      operator_svd_randomized(svd_input);
      break;
    }
    default:
    {
      boba_error("undefined svd_type");
      break;
    }
    }
  }

  /**
   * \brief Randomized SVD using a range finder followed by an exact SVD on the projected matrix.
   *
   * The randomized stage builds a small orthonormal basis Q that approximates the column space of
   * the input. The exact SVD is then computed on Q^H A, which is typically much smaller than A.
   */
  void operator_svd_randomized(boba::Matrix<space, data_t>& input)
  {
    checkpoint();

    const size_t rows = input.rows();
    const size_t cols = input.cols();
    const size_t rank_target = boba::min(rows, cols);
    const size_t oversampling = boba::min<size_t>(8, boba::max(rank_target, 1_z));
    const size_t sketch_cols = boba::min(cols, rank_target + oversampling);

    if ((rows == 0) or (cols == 0))
    {
      U.resize({rows, 0});
      S.resize({0});
      V.resize({cols, 0});
      significant_singular_values = 0;
      return;
    }

    boba::Matrix<space, data_t> omega({cols, sketch_cols});
    omega.rename("omega");
    omega.fill_with_random();

    auto y = input * omega;
    y.rename("randomized_range");

    boba::QR<space, data_t> qrf;
    qrf(y);
    auto q = qrf.Q;
    q.rename("Q");

    auto b = q.conjugate_transpose() * input;
    b.rename("B");

    SVD<space, data_t> reduced_svd;
    reduced_svd.svd_type = svd_types::economical;
    reduced_svd.tolerance_relative = tolerance_relative;
    reduced_svd.tolerance_absolute = tolerance_absolute;
    reduced_svd.max_kept_singular_values = max_kept_singular_values;
    reduced_svd(b);

    U = q * reduced_svd.U;
    S = reduced_svd.S;
    V = reduced_svd.V;
    significant_singular_values = reduced_svd.significant_singular_values;

    S.resize({significant_singular_values});
    U.resize({U.rows(), significant_singular_values});
    V.resize({V.rows(), significant_singular_values});
  }

  // -------------------------------------------------------------------------------------
  //  Eigen implementations
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Eigen SVD.
   */

  void operator_svd_eigen(boba::Matrix<execution_space::CPU, data_t>& input)
  {
    checkpoint();

    size_t number_nonzeros = 0;
    detail::factor_svd_eigen(input, U, S, V, number_nonzeros, tolerance_relative);

    if (number_nonzeros == 0)
    {
      significant_singular_values = 1;
      return;
    }
    checkpoint();

    BOBA_CALI_BEGIN("truncate_singular_values");
    checkpoint();
    this->truncate_singular_values();
    U.resize({U.rows(), significant_singular_values});
    V.resize({V.rows(), significant_singular_values});

    BOBA_CALI_END("truncate_singular_values");
  }

// -------------------------------------------------------------------------------------
//  Cuda implementations
// -------------------------------------------------------------------------------------
#ifdef BOBA_CUDA_LIBS

  /**
   * \brief
   * cusolver SVD.
   */

  void operator_svd_cuda(boba::Matrix<execution_space::CUDA, data_t>& input)
  {
    //
    // Compute SVD
    //
    checkpoint();

    const size_t cols = input.cols();
    boba::Matrix<execution_space::CUDA, data_t> VT;
    VT.rename("VT");

    detail::factor_svd_cusolver(input, U, S, VT);

    BOBA_CALI_BEGIN("truncate_singular_values");
    this->truncate_singular_values();

    BOBA_CALI_SWITCH("truncate_singular_values", "V_=_V^T");

    V = VT.conjugate_transpose_resize({cols, significant_singular_values});
    V.rename("V");

    if constexpr (boba::is_boba_debug_mode())
    {
      checkpoint();
      ::boba::nan_check(U);
      ::boba::nan_check(S);
      ::boba::nan_check(V);
      checkpoint();
    }

    BOBA_CALI_END("V_=_V^T");
  }
#endif

  // -------------------------------------------------------------------------------------
  //  Hip implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * rocsolver SVD.
   */

  void operator_svd_rocm(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    //
    // Compute truncated SVD
    //
    checkpoint();

    const size_t cols = input.cols();

    boba::Matrix<execution_space::HIP, data_t> VT;
    VT.rename("VT");
    detail::factor_svd_rocsolver(input, U, S, VT);

    BOBA_CALI_BEGIN("truncate_singular_values");
    checkpoint();
    this->truncate_singular_values();

    checkpoint();
    BOBA_CALI_SWITCH("truncate_singular_values", "V_=_V^T");

    U.resize({input.rows(), significant_singular_values});

    V = VT.conjugate_transpose_resize({cols, significant_singular_values});
    V.rename("V");

    if constexpr (boba::is_boba_debug_mode())
    {
      checkpoint();
      ::boba::nan_check(U);
      ::boba::nan_check(S);
      ::boba::nan_check(V);
      checkpoint();
    }

    checkpoint();
    BOBA_CALI_END("V_=_V^T");
  }

  /**
   * \brief
   * rocm truncated SVD.  Only solves for a range of singular values.
   */

  void operator_svdx_rocm(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    BOBA_CALI_MARK
    //
    // Compute truncated SVD
    //
    checkpoint();
    if constexpr (boba::is_ci_mode())
    {
      ::boba::nan_check(input);
    }

    const size_t rows = input.rows();
    const size_t cols = input.cols();
    const size_t svals = boba::min(rows, cols);

    BOBA_CALI_BEGIN("initialize_U_S_V_VT");

    U.resize({rows, svals});
    S.resize({svals});
    V.resize({cols, svals});

    if (rows == 1)
    {
      checkpoint();
      U.fill_with(1.0);
      V = input.transpose();
      auto V_norm = ::boba::norm_frobenius(V);
      if (V_norm < tolerance_absolute)
      {
        U.fill_with_zeros();
        S.fill_with_zeros();
        V.fill_with_zeros();
        significant_singular_values = 1;
        return;
      }
      V *= PotentiallyComplex<data_t>::value(real_data_t(1) / V_norm);
      S.fill_with(V_norm);
      significant_singular_values = 1;
      return;
    }
    if (cols == 1)
    {
      checkpoint();
      V.fill_with(1.0);
      U = input;
      auto U_norm = ::boba::norm_frobenius(U);
      if (U_norm < tolerance_absolute)
      {
        U.fill_with_zeros();
        S.fill_with_zeros();
        V.fill_with_zeros();
        significant_singular_values = 1;
        return;
      }
      U *= PotentiallyComplex<data_t>::value(real_data_t(1) / U_norm);
      S.fill_with(U_norm);
      significant_singular_values = 1;
      return;
    }

    boba::Matrix<execution_space::HIP, data_t> VT;
    VT.rename("VT");

    checkpoint();
    BOBA_CALI_SWITCH("initialize_U_S_V_VT", "estimate_sigma_max");

    // Ideally we seek the singular values in the range [simga_max(A)*svd_tolerance_relative, simga_max(A)]
    // However we don't have simga_max(A), so we need to find
    // [smallest_singular_value, largest_singular_value] such that
    // smallest_singular_value <= simga_max(A)*svd_tolerance_relative <= simga_max(A) <= largest_singular_value

    real_data_t estimate_max_norm = input.max_abs_reduce();
    real_data_t estimate_inf_norm = 0;
    real_data_t estimate_one_norm = 0;
    real_data_t estimate_frobenius_norm = ::boba::norm_frobenius(input);

    // Recall that (https://en.wikipedia.org/wiki/Matrix_norm)
    // sigma_max(A) <= sqrt(rows*cos) * ||A||_max
    // simga_max(A) <= sqrt(rows) * ||A||_inf
    // simga_max(A) <= sqrt(cols) * ||A||_1
    // simga_max(A) <= sqrt(svals) * ||A||_F
    real_data_t sigma_upper_bound = boba::sqrt(real_data_t(rows * cols)) * estimate_max_norm;
    if constexpr (std::is_same_v<data_t, real_data_t>)
    {
      estimate_inf_norm = input.matrix_inf_norm();
      estimate_one_norm = input.matrix_one_norm();
      sigma_upper_bound = boba::min(sigma_upper_bound, boba::sqrt(real_data_t(rows)) * estimate_inf_norm);
      sigma_upper_bound = boba::min(sigma_upper_bound, boba::sqrt(real_data_t(cols)) * estimate_one_norm);
    }
    sigma_upper_bound = boba::min(sigma_upper_bound, boba::sqrt(real_data_t(svals)) * estimate_frobenius_norm);
    sigma_upper_bound *= real_data_t(1.01);

    // For the lower bound
    // ||A||_max <= sigma_max(A)
    // ||A||_inf / sqrt(cols) <= simga_max(A)
    // ||A||_1 / sqrt(rows) <= simga_max(A)
    // ||A||_F / sqrt(svals) <= simga_max(A)
    real_data_t sigma_lower_bound = estimate_max_norm;
    ;
    if constexpr (std::is_same_v<data_t, real_data_t>)
    {
      sigma_lower_bound = boba::max(sigma_lower_bound, estimate_inf_norm / boba::sqrt(real_data_t(cols)));
      sigma_lower_bound = boba::max(sigma_lower_bound, estimate_one_norm / boba::sqrt(real_data_t(rows)));
    }
    sigma_lower_bound = boba::max(sigma_lower_bound, estimate_frobenius_norm / boba::sqrt(real_data_t(svals)));
    sigma_lower_bound *= real_data_t(0.99);

    real_data_t largest_singular_value = sigma_upper_bound;
    real_data_t smallest_singular_value = boba::max(sigma_lower_bound * tolerance_relative, tolerance_absolute);

    if (largest_singular_value < tolerance_absolute)
    {
      S.resize({1});
      U.resize({U.rows(), 1});
      V.resize({V.rows(), 1});
      S.fill_with_zeros();
      U.fill_with_zeros();
      V.fill_with_zeros();
      return;
    }

    rocblas_int index_first_singular_value = 0;
    rocblas_int index_last_singular_value = static_cast<rocblas_int>(min(max_kept_singular_values, svals));

    checkpoint();
    detail::factor_svdx_rocsolver(
      input,
      smallest_singular_value,
      largest_singular_value,
      index_first_singular_value,
      index_last_singular_value,
      S,
      U,
      VT,
      significant_singular_values);

    BOBA_CALI_SWITCH("estimate_sigma_max", "truncate_singular_values");

    // Truncation is performed within rocsolver_dgesvdx
    // this->truncate_singular_values();
    S.resize(significant_singular_values);

    BOBA_CALI_SWITCH("truncate_singular_values", "get_U_V");

    V = VT.conjugate_transpose_resize({cols, significant_singular_values});
    V.rename("V");
    U.resize({U.rows(), significant_singular_values});

    BOBA_CALI_END("get_U_V");
  }

// -------------------------------------------------------------------------------------
// Section: Apple Accelerate implementations
// -------------------------------------------------------------------------------------
#ifdef BOBA_ENABLE_APPLE

  /**
   * \brief
   * Apple Accelerate SVD.
   */

  void operator_svd_apple(boba::Matrix<execution_space::CPU, double>& input)
  {
    BOBA_CALI_MARK
    //
    // Compute SVD
    //
    checkpoint();

    const size_t rows = input.rows();
    const size_t cols = input.cols();
    const size_t svals = boba::min(rows, cols);

    BOBA_CALI_BEGIN("initialize_U_S_V_VT");

    U.resize({rows, rows});
    S.resize({svals});
    V.resize({cols, cols});

    boba::Matrix<execution_space::CPU, double> VT({V.cols(), V.rows()});
    VT.rename("VT");

    signed char U_opt = 'S';
    signed char VT_opt = 'S';

    // Workspace and parameters
    int lda = input.rows(); // Leading dimension of A
    int ldu = U.rows();     // Leading dimension of U
    int ldvt = VT.rows();   // Leading dimension of VT
    int lwork = -1;

    // Perform
    char jobu = 'A';  // Compute all m left singular vectors
    char jobvt = 'A'; // Compute all n right singular vectors

    int info = 0; // Output info (0 means success)

    int irows = input.rows();
    int icols = input.cols();

    checkpoint();
    BOBA_CALI_SWITCH("initialize_U_S_V_VT", "workspace_query");

    Vector<execution_space::CPU, double> host_work({static_cast<size_t>(1)});

    // Workspace query
    {
      dgesvd_(
        &jobu,
        &jobvt,
        &irows,
        &icols,
        input.data(),
        &lda,
        S.data(),
        U.data(),
        &ldu,
        VT.data(),
        &ldvt,
        host_work.data(),
        &lwork,
        &info);

      lwork = host_work({0});
      host_work.resize(static_cast<size_t>(lwork));
    }

    checkpoint();
    BOBA_CALI_SWITCH("workspace_query", "apple_svd");

    // Compute
    {
      dgesvd_(
        &jobu,
        &jobvt,
        &irows,
        &icols,
        input.data(),
        &lda,
        S.data(),
        U.data(),
        &ldu,
        VT.data(),
        &ldvt,
        host_work.data(),
        &lwork,
        &info);

      boba_always_assert_nonnegative(info, "dgesvd_ did not converge.");
      boba_always_assert_nonnegative(-info, "info < 0, the i-th parameter is invalid");
    }

    this->truncate_singular_values();

    BOBA_CALI_SWITCH("apple_svd", "V_=_V^T");

    V = VT.conjugate_transpose_resize({cols, significant_singular_values});
    V.rename("V");
    U.resize({U.rows(), significant_singular_values});

    if constexpr (boba::is_boba_debug_mode())
    {
      checkpoint();
      ::boba::nan_check(U);
      ::boba::nan_check(S);
      ::boba::nan_check(V);
      checkpoint();
    }

    BOBA_CALI_END("V_=_V^T");
  }
#else
  /**
   * \brief Reports that Apple Accelerate SVD is unavailable.
   * \param input Matrix to factor.
   */
  void operator_svd_apple(boba::Matrix<execution_space::CPU, double>& input)
  {
    detail::ignore(input);
    boba_error("Apple Accelerate is not enabled for SVD.");
  }
#endif
};

} // namespace boba
