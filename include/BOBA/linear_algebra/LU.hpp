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
 * \brief Abstraction class for LU factorizations.
 */

template <execution_space space, typename data_t>
struct LU
{
  enum struct lu_types : size_t
  {
    partial_pivot,
    full_pivot
  };

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> L;
  boba::Matrix<space, data_t> U;
  boba::PermutationMatrix<space, index_t> P;
  boba::PermutationMatrix<space, index_t> Q;

  lu_types lu_type = lu_types::partial_pivot;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  LU()
  {
    L.rename("L");
    U.rename("U");
    P.rename("P");
    Q.rename("Q");
    parse_type_from_env();
  }

  /**
   * \brief copy constructor
   */
  LU(LU const&) = default;

  /**
   * \brief move constructor
   */
  LU(LU&&) = default;

  /**
   * \brief copy assignment operator
   */
  LU& operator=(LU const&) = default;

  /**
   * \brief move assignment operator
   */
  LU& operator=(LU&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Factorization to copy from.
   */
  template <execution_space rhs_space>
  LU(LU<rhs_space, data_t> const& rhs)
      : L(rhs.L),
        U(rhs.U),
        P(rhs.P),
        Q(rhs.Q),
        lu_type(rhs.lu_type)
  {
  }

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Factorization to copy from.
   * \return This factorization after assignment.
   */
  template <execution_space rhs_space>
  LU& operator=(LU<rhs_space, data_t> const& rhs)
  {
    L = rhs.L;
    U = rhs.U;
    P = rhs.P;
    Q = rhs.Q;
    lu_type = rhs.lu_type;
    return *this;
  }

  /**
   * \brief destructor
   */
  ~LU() = default;

  // -------------------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reads `LU_TYPE` from the environment.
   */
  void parse_type_from_env()
  {
    lu_type = lu_types::partial_pivot;

    if (env_match("LU_TYPE", "full_pivot"))
    {
      lu_type = lu_types::full_pivot;
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Architecture-agnostic implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reconstructs the factored matrix from the stored factors.
   * \return Matrix reconstructed as `P^T * L * U * Q^T` for full pivoting or `P^T * L * U` for partial pivoting.
   */
  [[nodiscard]]
  Matrix<space, data_t> reform_matrix()
  {
    BOBA_CALI_MARK
    checkpoint();
    auto LU_factor = this->L * this->U;
    auto Pt = this->P.transpose();
    auto PtLU = Pt * LU_factor;
    if (lu_type == lu_types::partial_pivot)
    {
      return PtLU;
    }
    auto Qt = this->Q.transpose();
    auto PtLUQt = PtLU * Qt;
    return PtLUQt;
  }

  /**
   * \brief Computes a CPU LU factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CPU, data_t> const& input)
  {
    switch (lu_type)
    {
    case lu_types::partial_pivot:
    {
#ifdef BOBA_ENABLE_APPLE
      operator_lu_apple(input);
#else
      operator_lu_eigen(input);
#endif
      break;
    }
    case lu_types::full_pivot:
    {
      operator_lufull_eigen(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a CUDA LU factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CUDA, data_t> const& input)
  {
    switch (lu_type)
    {
    case lu_types::partial_pivot:
    {
      operator_lu_cuda(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a HIP LU factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    switch (lu_type)
    {
    case lu_types::partial_pivot:
    {
      operator_lu_hip(input);
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
   * \brief Computes a partial-pivot LU factorization with Eigen.
   * \param input Matrix to factor.
   */
  void operator_lu_eigen(boba::Matrix<execution_space::CPU, data_t> const& input)
  {
#ifndef DEVICE_CODE
    checkpoint();
    detail::factor_lu_eigen(input, L, U, P);
    checkpoint();
#endif
  }

  /**
   * \brief Computes a full-pivot LU factorization with Eigen.
   * \param input Matrix to factor.
   */
  void operator_lufull_eigen(boba::Matrix<execution_space::CPU, data_t> const& input)
  {
#ifndef DEVICE_CODE
    checkpoint();
    detail::factor_lufull_eigen(input, L, U, P, Q);
#endif
  }

  // -------------------------------------------------------------------------------------
  // Section: Cuda implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes a partial-pivot LU factorization with cuSOLVER.
   * \param input Matrix to factor.
   */
  template <typename backend_data_type>
  void operator_lu_cuda(boba::Matrix<execution_space::CUDA, backend_data_type> const& _input)
  {
    BOBA_CALI_MARK
    detail::factor_lu_cusolver(_input, L, U, P);
  }

  // ---------------------------------------
  //  Hip implementations
  // ---------------------------------------

  /**
   * \brief Computes a partial-pivot LU factorization with hipSOLVER.
   * \param input Matrix to factor.
   */
  template <typename backend_data_type>
  void operator_lu_hip(boba::Matrix<execution_space::HIP, backend_data_type> const& _input)
  {
    BOBA_CALI_MARK
    detail::factor_lu_hipsolver(_input, L, U, P);
  }

// -------------------------------------------------------------------------------------
// Section: Apple Accelerate implementations
// -------------------------------------------------------------------------------------
#ifdef BOBA_ENABLE_APPLE

  /**
   * \brief Computes a partial-pivot LU factorization with Apple Accelerate.
   * \param input Matrix to factor.
   */
  void operator_lu_apple(const boba::Matrix<execution_space::CPU, double>& _input)
  {
    BOBA_CALI_MARK
    checkpoint();

    boba_always_assert_equal(_input.rows(), _input.cols(), "LU requires nonsingular square matrix");

    auto input = _input;
    int rows = input.rows();
    int cols = input.cols();
    int info;

    checkpoint();
    Vector<execution_space::CPU, int> perms({input.rows()});

    // Perform LU decomposition
    checkpoint();
    dgetrf_(&rows, &rows, input.data(), &rows, perms.data(), &info);

    boba_always_assert_equal(info, 0, "LU decomposition failed.");

    checkpoint();
    P.resize({input.rows()});
    L.resize(input.sizes());
    U.resize(input.sizes());

    auto perms_view = perms.const_view();
    auto input_view = input.view();
    auto P_view = P.view();
    auto L_view = L.view();
    auto U_view = U.view();

    // Copy permutation indices
    checkpoint();
    ::boba::loop<execution_space::CPU, 1, index_t>(input.rows(),
                                                   [=] __boba_host_device__(index_t i)
    {
      P_view(i) = static_cast<index_t>(perms_view(i) - 1);
    });

    // Copy L, U
    ::boba::loop<execution_space::CPU, 2, index_t>(input.sizes(),
                                                   [=] __boba_host_device__(Array<index_t, 2> rc)
    {
      auto row = rc[0];
      auto col = rc[1];
      auto L_value = 0.0;
      auto U_value = 0.0;
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
        L_value = 1.0;
      }

      L_view(rc) = L_value;
      U_view(rc) = U_value;
    });

    checkpoint();
  }
#else
  /**
   * \brief Reports that Apple Accelerate LU is unavailable.
   * \param _input Matrix to factor.
   */
  void operator_lu_apple(const boba::Matrix<execution_space::CPU, double>& _input)
  {
    detail::ignore(_input);
    boba_error("Apple Accelerate is not enabled for LU.");
  }
#endif
};

} // namespace boba
