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
 * \brief Abstraction class for Cholesky factorizations.
 */

template <execution_space space, typename data_t>
struct Cholesky
{
  enum struct cholesky_types : size_t
  {
    no_pivot,
    full_pivot
  };

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> L;
  // boba::PermutationMatrix<space, index_t> P;

  cholesky_types cholesky_type = cholesky_types::no_pivot;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  Cholesky()
  {
    L.rename("L");
    // P.rename("P");
    parse_type_from_env();
  }

  /**
   * \brief copy constructor
   */
  Cholesky(Cholesky const&) = default;

  /**
   * \brief move constructor
   */
  Cholesky(Cholesky&&) = default;

  /**
   * \brief copy assignment operator
   */
  Cholesky& operator=(Cholesky const&) = default;

  /**
   * \brief move assignment operator
   */
  Cholesky& operator=(Cholesky&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Factorization to copy from.
   */
  template <execution_space rhs_space>
  Cholesky(Cholesky<rhs_space, data_t> const& rhs)
      : L(rhs.L)
        //, P(rhs.P)
        ,
        cholesky_type(rhs.cholesky_type)
  {
  }

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Factorization to copy from.
   * \return This factorization after assignment.
   */
  template <execution_space rhs_space>
  Cholesky& operator=(Cholesky<rhs_space, data_t> const& rhs)
  {
    L = rhs.L;
    // P = rhs.P;
    cholesky_type = rhs.cholesky_type;
    return *this;
  }

  /**
   * \brief destructor
   */
  ~Cholesky() = default;

  // -------------------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reads `CHOLESKY_TYPE` from the environment.
   */
  void parse_type_from_env()
  {
    cholesky_type = cholesky_types::no_pivot;

    if (env_match("CHOLESKY_TYPE", "full_pivot"))
    {
      cholesky_type = cholesky_types::full_pivot;
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Architecture-agnostic implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reconstructs the factored matrix from the stored factor.
   * \return Matrix reconstructed as `L * L^H` for complex data or `L * L^T` for real data.
   */
  [[nodiscard]]
  Matrix<space, data_t> reform_matrix()
  {
    BOBA_CALI_MARK
    checkpoint();
    auto LLT = this->L * this->L.conjugate_transpose();
    return LLT;
  }

  /**
   * \brief Computes a CPU Cholesky factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CPU, data_t> const& input)
  {
    switch (cholesky_type)
    {
    case cholesky_types::no_pivot:
    {
      // TODO - make branch logic for Cholesky
      // operator_cholesky_apple(input);
      operator_cholesky_eigen(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a CUDA Cholesky factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::CUDA, data_t> const& input)
  {
    switch (cholesky_type)
    {
    case cholesky_types::no_pivot:
    {
      operator_cholesky_cuda(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  /**
   * \brief Computes a HIP Cholesky factorization.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<execution_space::HIP, data_t>& input)
  {
    switch (cholesky_type)
    {
    case cholesky_types::no_pivot:
    {
      operator_cholesky_hip(input);
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
   * \brief Computes the factorization with Eigen.
   * \param input Matrix to factor.
   */
  void operator_cholesky_eigen(boba::Matrix<execution_space::CPU, data_t> const& input)
  {
#ifndef DEVICE_CODE
    checkpoint();
    detail::factor_cholesky_eigen(input, L);
#endif
  }

  // -------------------------------------------------------------------------------------
  // Section: Cuda implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes the factorization with cuSOLVER.
   * \param input Matrix to factor.
   */
  template <typename backend_data_type>
  void operator_cholesky_cuda(boba::Matrix<execution_space::CUDA, backend_data_type> const& _input)
  {
    BOBA_CALI_MARK
    detail::factor_cholesky_cusolver(_input, L);
  }

  // ---------------------------------------
  //  Hip implementations
  // ---------------------------------------

  /**
   * \brief Computes the factorization with hipSOLVER.
   * \param input Matrix to factor.
   */
  template <typename backend_data_type>
  void operator_cholesky_hip(boba::Matrix<execution_space::HIP, backend_data_type> const& _input)
  {
    BOBA_CALI_MARK
    detail::factor_cholesky_hipsolver(_input, L);
  }

  // -------------------------------------------------------------------------------------
  // Section: Apple Accelerate implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes the factorization with Apple Accelerate.
   * \param input Matrix to factor.
   */
  void operator_cholesky_apple(const boba::Matrix<execution_space::CPU, double>& _input)
  {
    BOBA_CALI_MARK
    detail::ignore(_input);
    checkpoint();

    boba_error("operator_cholesky_apple not yet implemented");
  }
};

} // namespace boba
