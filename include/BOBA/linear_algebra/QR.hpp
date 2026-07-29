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
 * \brief Abstraction class for several QR factorizations.
 *
 * Supported variants include non-pivoted Householder QR and rank-revealing pivoted QR variants.
 */

template <execution_space space, typename _data_t>
struct QR
{
  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  enum struct qr_types : size_t
  {
    householder,
    column_pivot,
    full_pivot
  };

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> Q;
  boba::Matrix<space, data_t> R;

  qr_types qr_type = qr_types::householder;

  real_data_t qr_rank_tolerance = real_data_t(50) * std::numeric_limits<real_data_t>::epsilon();
  size_t ranks = 0;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  QR()
  {
    Q.rename("Q");
    R.rename("R");
    parse_type_from_env();
  }

  /**
   * \brief copy constructor
   */
  QR(QR const&) = default;

  /**
   * \brief move constructor
   */
  QR(QR&&) = default;

  /**
   * \brief copy assignment operator
   */
  QR& operator=(QR const&) = default;

  /**
   * \brief move assignment operator
   */
  QR& operator=(QR&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Factorization to copy from.
   */
  template <execution_space rhs_space>
  QR(QR<rhs_space, data_t> const& rhs)
      : Q(rhs.Q),
        R(rhs.R),
        qr_type(rhs.qr_type),
        qr_rank_tolerance(rhs.qr_rank_tolerance),
        ranks(rhs.ranks)
  {
    parse_type_from_env();
  }

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Factorization to copy from.
   * \return This factorization after assignment.
   */
  template <execution_space rhs_space>
  QR& operator=(QR<rhs_space, data_t> const& rhs)
  {
    parse_type_from_env();
    Q = rhs.Q;
    R = rhs.R;
    qr_type = rhs.qr_type;
    qr_rank_tolerance = rhs.qr_rank_tolerance;
    ranks = rhs.ranks;
    return *this;
  }

  /**
   * \brief destructor
   */
  ~QR() = default;

  // -------------------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reads `QR_TYPE` from the environment.
   *
   * If `QR_TYPE=qrrr`, use the column-pivot QR routine.
   */
  void parse_type_from_env()
  {
    qr_type = qr_types::householder;

    if (env_match("QR_TYPE", "qrrr"))
    {
      qr_type = qr_types::column_pivot;
    }

    if (env_match("QR_TYPE", "qrfull"))
    {
      qr_type = qr_types::full_pivot;
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Architecture-agnostic implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reconstructs a matrix from the stored QR factors.
   * \return Matrix reconstructed from the stored factors.
   */
  [[nodiscard]]
  Matrix<space, data_t> reform_matrix()
  {
    BOBA_CALI_MARK
    checkpoint();
    return apply_householder_left(this->R);
  }

  /**
   * \brief Left-multiplies an output matrix by the stored `R` factor in place.
   * \param output Matrix overwritten with `R * output`.
   */
  template <execution_space output_space>
  void apply_R_left_in_place(boba::Matrix<output_space, data_t>& output)
  {
    BOBA_CALI_MARK
    checkpoint();
    boba::Matrix<space, data_t> input(output);
    input.resize({R.cols(), input.cols()});
    output = R * input;
  }

  /**
   * \brief Applies the stored Householder factor from the left.
   * \param input Matrix to multiply by the stored `Q` factor.
   * \return Product of the stored Householder factor and `input`.
   */
  [[nodiscard]]
  ::boba::Matrix<space, data_t> apply_householder_left(boba::Matrix<space, data_t> const& input)
  {
    BOBA_CALI_MARK
    size_t number_householder_vectors = Q.cols();
    size_t input_rows = input.rows();
    size_t common_size = boba::min(input_rows, number_householder_vectors);
    Q.resize({Q.rows(), common_size});
    auto input_copy = input;
    input_copy.resize({common_size, input.cols()});
    return Q * input_copy;
  }

  /**
   * \brief Computes a CPU QR factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CPU, data_t>& input)
  {
    switch (qr_type)
    {
    case qr_types::householder:
    {
#ifdef BOBA_ENABLE_APPLE
      operator_qr_apple(input);
#else
      operator_qr_eigen(input);
#endif
      break;
    }
    case qr_types::column_pivot:
    {
      operator_qrrr_eigen(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a CUDA QR factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CUDA, data_t>& input)
  {
    switch (qr_type)
    {
    case qr_types::householder:
    {
      operator_qr_cuda(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a HIP QR factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    switch (qr_type)
    {
    case qr_types::householder:
    {
      operator_qr_hip(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Eigen implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes a non-pivoted Householder QR factorization with Eigen.
   * \param input Matrix to factor.
   */
  void operator_qr_eigen(boba::Matrix<execution_space::CPU, data_t>& input)
  {
    checkpoint();
    detail::factor_qr_eigen(input, this->Q, this->R, ranks);
    checkpoint();
  }

  /**
   * \brief Computes a column-pivoted rank-revealing QR factorization with Eigen.
   * \param input Matrix to factor.
   */
  void operator_qrrr_eigen(boba::Matrix<execution_space::CPU, data_t>& input)
  {
    checkpoint();
    detail::factor_qrrr_eigen(input, this->Q, this->R, ranks, qr_rank_tolerance);
    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Cuda implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes a non-pivoted Householder QR factorization with cuSOLVER.
   * \param input Matrix to factor.
   */
  void operator_qr_cuda(boba::Matrix<execution_space::CUDA, data_t>& input)
  {
    checkpoint();

    const size_t rows = input.rows();
    const size_t cols = input.cols();
    ranks = std::min(rows, cols); /* upper bound, not true rank */

    boba::Vector<execution_space::CUDA, data_t> tau({ranks});

    checkpoint();
    detail::factor_qr_cusolver(input, tau);

    BOBA_CALI_BEGIN("extract_QR");
    checkpoint();
    R.resize({ranks, cols});
    R.fill_with_zeros();
    checkpoint();
    Q.resize({rows, ranks});
    Q.fill_with_zeros();
    checkpoint();
    auto input_view = input.view();
    auto Q_view = this->Q.view();
    auto R_view = this->R.view();
    checkpoint();
    // copy R: in upper triangular part
    //   fixup householder vectors with leading 0s and 1
    ::boba::loop<execution_space::CUDA, 2>({rows, cols},
                                           [=] __boba_host_device__(Array<index_t, 2> rc)
    {
      auto row = rc[0];
      auto col = rc[1];

      if (col > row)
      {
        R_view({row, col}) = input_view({row, col});
      }
      else if (col == row)
      {
        Q_view({row, col}) = ::boba::PotentiallyComplex<data_t>::value(1.0);
        R_view({row, col}) = input_view({row, col});
      }
      else
      {
        Q_view({row, col}) = input_view({row, col});
      }
    });
    BOBA_CALI_END("extract_QR");
    checkpoint();

    checkpoint();
    detail::form_q_cusolver(this->Q, tau, ranks);
    checkpoint();
  }

  // ---------------------------------------
  //  Hip implementations
  // ---------------------------------------

  /**
   * \brief
   * hip's householder QR, no pivoting. Not rank-revealing.
   */

  void operator_qr_hip(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    checkpoint();

    const size_t rows = input.rows();
    const size_t cols = input.cols();
    ranks = std::min(rows, cols); /* upper bound, not true rank */

    boba::Vector<execution_space::HIP, data_t> tau({ranks});

    checkpoint();
    detail::factor_qr_hipsolver(input, tau);

    BOBA_CALI_BEGIN("extract_QR");
    checkpoint();
    R.resize({ranks, cols});
    R.fill_with_zeros();
    checkpoint();
    Q.resize({rows, ranks});
    Q.fill_with_zeros();
    checkpoint();
    auto input_view = input.view();
    auto Q_view = this->Q.view();
    auto R_view = this->R.view();
    checkpoint();
    // copy R: in upper triangular part
    //   fixup householder vectors with leading 0s and 1
    ::boba::loop<execution_space::HIP, 2>({rows, cols},
                                          [=] __boba_host_device__(Array<size_t, 2> rc)
    {
      size_t row = rc[0];
      size_t col = rc[1];
      if (col > row)
      {
        R_view({row, col}) = input_view({row, col});
      }
      else if (col == row)
      {
        Q_view({row, col}) = ::boba::PotentiallyComplex<data_t>::value(1.0);
        R_view({row, col}) = input_view({row, col});
      }
      else
      {
        Q_view({row, col}) = input_view({row, col});
      }
    });
    BOBA_CALI_END("extract_QR");
    checkpoint();

    checkpoint();
    detail::form_q_hipsolver(this->Q, tau, ranks);
    checkpoint();
  }

// -------------------------------------------------------------------------------------
// Section: Apple Accelerate implementations
// -------------------------------------------------------------------------------------
#ifdef BOBA_ENABLE_APPLE

  /**
   * \brief
   * Apple Accelerate QR
   */

  void operator_qr_apple(boba::Matrix<execution_space::CPU, double>& input)
  {
    BOBA_CALI_MARK
    checkpoint();

    int rows = input.rows();
    int cols = input.cols();
    ranks = std::min(static_cast<size_t>(rows), static_cast<size_t>(cols)); /* upper bound, not true rank */
    int iranks = ranks;

    boba::Vector<execution_space::CPU, double> tau({ranks});

    // Workspace query for optimal size
    int lwork = -1; // Query for optimal workspace size

    boba::Vector<execution_space::CPU, double> work({1});

    int info;

    // Query optimal workspace size
    {
      dgeqrf_(&rows, &cols, input.data(), &rows, tau.data(), work.data(), &lwork, &info);

      boba_always_assert_equal(info, 0, "Error during workspace query for QR decomposition.");

      // Allocate workspace with the optimal size
      lwork = work({0});
      work.resize(static_cast<size_t>(lwork));
    }

    // Perform QR decomposition
    {
      dgeqrf_(&rows, &cols, input.data(), &rows, tau.data(), work.data(), &lwork, &info);

      boba_always_assert_equal(info, 0, "Error during QR decomposition computation.");
    }

    checkpoint();
    R.resize({static_cast<index_t>(ranks), static_cast<index_t>(cols)});
    R.fill_with_zeros();
    checkpoint();
    Q.resize({static_cast<index_t>(rows), static_cast<index_t>(ranks)});
    Q.fill_with_zeros();
    checkpoint();
    auto input_view = input.view();
    auto Q_view = this->Q.view();
    auto R_view = this->R.view();
    checkpoint();

    // copy R in upper triangular part
    //   fixup householder vectors with leading 0s and 1
    ::boba::loop<execution_space::CPU, 2>({static_cast<index_t>(rows), static_cast<index_t>(cols)},
                                          [=] __boba_host_device__(Array<index_t, 2> rc)
    {
      auto row = rc[0];
      auto col = rc[1];

      if (col > row)
      {
        R_view({row, col}) = input_view({row, col});
      }
      else if (col == row)
      {
        Q_view({row, col}) = ::boba::PotentiallyComplex<data_t>::value(1.0);
        R_view({row, col}) = input_view({row, col});
      }
      else
      {
        Q_view({row, col}) = input_view({row, col});
      }
    });

    // Query optimal workspace size
    checkpoint();
    {
      lwork = -1; // Query for optimal workspace size for Q computation
      dorgqr_(&rows, &iranks, &iranks, Q.data(), &rows, tau.data(), work.data(), &lwork, &info);

      boba_always_assert_equal(info, 0, "Error during workspace query for Q computation.");

      // Allocate workspace with the optimal size
      lwork = work({0});
      work.resize(static_cast<size_t>(lwork));
    }

    // Generate Q
    checkpoint();
    {
      dorgqr_(&rows, &iranks, &iranks, Q.data(), &rows, tau.data(), work.data(), &lwork, &info);

      boba_always_assert_equal(info, 0, "Error while forming Q.");
    }

    checkpoint();
  }
#else
  /**
   * \brief Reports that Apple Accelerate QR is unavailable.
   * \param input Matrix to factor.
   */
  void operator_qr_apple(boba::Matrix<execution_space::CPU, double>& input)
  {
    detail::ignore(input);
    boba_error("Apple Accelerate is not enabled for QR.");
  }
#endif
};

} // namespace boba
