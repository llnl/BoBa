// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <stdexcept>

namespace boba
{

/**
 * \brief Abstraction class for NNMF factorizations.
 *
 * See:
 * https://arxiv.org/pdf/1401.5226
 * https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
 */

template <execution_space space, typename _data_t>
struct NNMF
{
  using data_t = _data_t;

  enum struct nnmf_types : int
  {
    svd,
  };

  enum struct update_schemes : int
  {
    multiplicative_update,
    nonnegative_least_squares
  };

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  boba::Matrix<space, data_t> W;
  boba::Matrix<space, data_t> H;

  nnmf_types nnmf_type = nnmf_types::svd;
  update_schemes update_scheme = update_schemes::multiplicative_update;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief default constructor
   */
  NNMF()
  {
    W.rename("W");
    H.rename("H");
    parse_type_from_env();
  }

  /**
   * \brief copy constructor
   */
  NNMF(NNMF const&) = default;
  /**
   * \brief move constructor
   */
  NNMF(NNMF&&) = default;

  /**
   * \brief copy assignment operator
   */
  NNMF& operator=(NNMF const&) = default;
  /**
   * \brief move assignment operator
   */
  NNMF& operator=(NNMF&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Factorization to copy from.
   */
  template <execution_space rhs_space>
  NNMF(NNMF<rhs_space, data_t> const& rhs)
      : W(rhs.W),
        H(rhs.H),
        nnmf_type(rhs.nnmf_type)
  {
    parse_type_from_env();
  }

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Factorization to copy from.
   * \return This factorization after assignment.
   */
  template <execution_space rhs_space>
  NNMF& operator=(NNMF<rhs_space, data_t> const& rhs)
  {
    parse_type_from_env();
    W = rhs.W;
    H = rhs.H;
    nnmf_type = rhs.nnmf_type;
    return *this;
  }

  /**
   * \brief destructor
   */
  ~NNMF() = default;

  // -------------------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------------------

  /**
   * \brief Reads NNMF settings from the environment.
   */
  void parse_type_from_env()
  {
    nnmf_type = nnmf_types::svd;
  }

  // -------------------------------------------------------------------------------------
  // Section: Architecture-agnostic implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes an NNMF factorization of the input matrix.
   * \param input Matrix to factor.
   */
  void operator()(boba::Matrix<space, data_t>& input)
  {
    switch (nnmf_type)
    {
    case nnmf_types::svd:
    {
      svd_initialization(input);
      operator_nnmf(input);
      break;
    }
    default:
    {
      boba_error("not yet implemented");
    }
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: implementations
  // -------------------------------------------------------------------------------------

  /**
   * \brief Initializes the factors from an SVD-based nonnegative split.
   * \param input Matrix to factor.
   *
   * See Section 3.1.8 of https://arxiv.org/pdf/1401.5226
   */
  void svd_initialization(boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK

    size_t rows = input.rows();
    size_t cols = input.cols();
    size_t smaller_dimension = ::boba::min(rows, cols);

    W.resize({rows, smaller_dimension});
    H.resize({smaller_dimension, cols});

    SVD<space, data_t> svd_init;
    svd_init(input);

    auto ranks = svd_init.significant_singular_values;

    auto nnU = svd_init.U.nonnegative_part();
    auto npU = svd_init.U.nonpositive_part();
    auto Vt = svd_init.V.transpose();
    auto nnVt = Vt.nonnegative_part();
    auto npVt = Vt.nonpositive_part();

    W = nnU;
    H = nnVt;

    for (size_t k = 0; k < ranks; k++)
    {
      auto k_nnU = nnU.get_submatrix({0, rows}, {k, k + 1});
      auto k_nnVt = nnVt.get_submatrix({k, k + 1}, {0, cols});

      // term_nonnegative = k_nnU*k_nnVt;
      // ||term_nonnegative||_F = ||k_nnU||_F * ||k_nnVt||_F;
      auto norm_term_nonnegative = ::boba::norm_frobenius(k_nnU) * ::boba::norm_frobenius(k_nnVt);

      auto k_npU = npU.get_submatrix({0, rows}, {k, k + 1});
      auto k_npVt = npVt.get_submatrix({k, k + 1}, {0, cols});

      auto norm_term_nonpositve = ::boba::norm_frobenius(k_npU) * ::boba::norm_frobenius(k_npVt);

      if (norm_term_nonpositve > norm_term_nonnegative)
      {
        W.replace_submatrix({0, rows}, {k, k + 1}, k_npU);
        H.replace_submatrix({k, k + 1}, {0, cols}, k_npVt);
      }
    }
  }

  /**
   * \brief Computes the next iterate for `H`.
   * \param W_old Current `W` factor.
   * \param H_old Current `H` factor.
   * \param V Input matrix being factorized.
   * \return Updated `H` factor.
   */
  [[nodiscard]]
  Matrix<space, data_t> generate_H_new(Matrix<space, data_t>& W_old, Matrix<space, data_t>& H_old, Matrix<space, data_t>& V)
  {
    BOBA_CALI_MARK
    checkpoint();
    auto H_new = H_old;
    auto H_new_view = H_new.view();

    if (update_scheme == update_schemes::multiplicative_update)
    {
      auto update_numerator = W_old.transpose() * V;
      auto WtW = W_old.transpose() * W_old;
      auto update_denominator = WtW * H_old;

      checkpoint();
      auto update_numerator_view = update_numerator.const_view();
      auto update_denominator_view = update_denominator.const_view();
      auto H_old_view = H_old.view();

      ::boba::loop<space, 1>(H_old.size(),
                             [=] __boba_host_device__(size_t i)
      {
        auto numerator = H_old_view(i) * update_numerator_view(i);
        H_new_view(i) = is_tiny(numerator) ? 0.0 : numerator / update_denominator_view(i);
      });
    }
    else if (update_scheme == update_schemes::nonnegative_least_squares)
    {
      for (size_t col = 0; col < H_old.cols(); col++)
      {
        auto H_col_new = nnls(W_old, flatten(V.extract_columns(col)));
        auto H_update_view = H_col_new.view();
        ::boba::loop<space, 1>({H_old.rows()},
                               [=] __boba_host_device__(index_t row)
        {
          H_new_view({row, col}) = H_update_view({row});
        });
      }
    }
    else
    {
      boba_error("Not yet implemented");
    }

    checkpoint();
    H_new.rename("H_new");
    return H_new;
  }

  /**
   * \brief Computes the next iterate for `W`.
   * \param W_old Current `W` factor.
   * \param H_new Updated `H` factor.
   * \param V Input matrix being factorized.
   * \return Updated `W` factor.
   */
  [[nodiscard]]
  Matrix<space, data_t> generate_W_new(Matrix<space, data_t>& W_old, Matrix<space, data_t>& H_new, Matrix<space, data_t>& V)
  {
    BOBA_CALI_MARK
    checkpoint();
    auto W_new = W_old;
    auto W_new_view = W_new.view();

    if (update_scheme == update_schemes::multiplicative_update)
    {
      auto update_numerator = V * H_new.transpose();
      auto HHt = H_new * H_new.transpose();
      auto update_denominator = W_old * HHt;

      checkpoint();
      auto update_numerator_view = update_numerator.const_view();
      auto update_denominator_view = update_denominator.const_view();
      auto W_old_view = W_old.view();

      ::boba::loop<space, 1>(W_old.size(),
                             [=] __boba_host_device__(size_t i)
      {
        auto numerator = W_old_view(i) * update_numerator_view(i);
        W_new_view(i) = is_tiny(numerator) ? 0.0 : numerator / update_denominator_view(i);
      });
    }
    else if (update_scheme == update_schemes::nonnegative_least_squares)
    {
      auto H_newT = H_new.transpose();
      auto VT = V.transpose();
      for (size_t row = 0; row < W_old.rows(); row++)
      {
        auto VT_slice = flatten(VT.extract_columns(row));
        auto W_col_new = nnls(H_newT, VT_slice);
        auto W_update_view = W_col_new.view();
        ::boba::loop<space, 1>({W_old.cols()},
                               [=] __boba_host_device__(index_t col)
        {
          W_new_view({row, col}) = W_update_view({col});
        });
      }
    }
    else
    {
      boba_error("Not yet implemented");
    }

    checkpoint();
    W_new.rename("W_new");
    return W_new;
  }

  /**
   * \brief Performs the iterative NNMF update loop.
   * \param input Matrix being factorized.
   *
   * Assumes `W` and `H` have already been initialized.
   */
  void operator_nnmf(boba::Matrix<space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();

    boba::Matrix<space, data_t> W_old = W;
    boba::Matrix<space, data_t> H_old = H;
    W_old.rename("W_old");
    H_old.rename("H_old");

    bool iterating = true;
    size_t iterations = 0;

    while (iterating)
    {
      iterations++;

      auto H_new = generate_H_new(W_old, H_old, input);
      auto W_new = generate_W_new(W_old, H_new, input);

      /*
      TODO<implementation details> Consider a normalization (see paper)
      auto W_new_frob = ::boba::norm_frobenius(W_new);
      auto H_new_frob = ::boba::norm_frobenius(H_new);

      if(H_new_frob > W_new_frob)
      {
        H_new *= (W_new_frob/H_new_frob);
      }
      else
      {
        W_new *= (H_new_frob/W_new_frob);
      }
      */

      auto W_change = norm_difference_frobenius(W_new, W_old);
      auto H_change = norm_difference_frobenius(H_new, H_old);

      bool is_W_changing = W_change > 1.0e-07;
      bool is_H_changing = H_change > 1.0e-07;

      iterating = (is_W_changing or is_H_changing) and (iterations < 100);

      if (iterating)
      {
        W_old = W_new;
        H_old = H_new;
      }
      else
      {
        W = W_new;
        H = H_new;
      }
    }
  }
};

} // namespace boba
