// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief Selects interpolation indices with the DEIM algorithm.
 * \param U Basis matrix used to choose interpolation indices.
 * \return Vector of selected row indices.
 *
 * Reference: Sorensen and Embree, "A DEIM Induced CUR Factorization", SIAM Journal on Scientific
 * Computing, 2016.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
inline boba::Vector<space, index_t> DEIM(const boba::Matrix<space, data_t>& U)
{
  BOBA_CALI_BEGIN("DEIM");
  checkpoint();

  index_t n = static_cast<index_t>(U.rows());
  index_t m = static_cast<index_t>(U.cols());

  boba::Vector<space, index_t> p({static_cast<size_t>(m)});
  p.rename("DEIM_indices");
  auto p_view = p.view();

  checkpoint();
  BOBA_CALI_SWITCH("DEIM", "select_first_index");

  // Step 1: Select first index as maximum of first basis vector
  boba::Vector<space, index_t> first_col_indices({1});
  first_col_indices.fill_with(0);

  auto first_col_vec = flatten(U.extract_columns(first_col_indices));
  auto [max_val, max_idx] = first_col_vec.max_abs_loc_reduce();
  p_view(0) = max_idx;

  checkpoint();
  BOBA_CALI_SWITCH("select_first_index", "DEIM_iteration");

  // Step 2: Iteratively select remaining indices
  for (index_t j = 1; j < m; ++j)
  {
    checkpoint();

    // Extract indices for columns 0:j-1 and column j
    boba::Vector<space, index_t> col_indices_0_to_j_minus_1({static_cast<size_t>(j)});
    boba::Vector<space, index_t> col_index_j({1});

    auto col_view = col_indices_0_to_j_minus_1.view();
    ::boba::detail::loop<space>(0, j, [=] __boba_host_device__(index_t i)
    {
      col_view(i) = i;
    });
    col_index_j.fill_with(j);

    // Extract row indices p[0:j-1] (j elements)
    boba::Vector<space, index_t> row_indices_p({static_cast<size_t>(j)});
    auto row_view = row_indices_p.view();
    ::boba::detail::loop<space>(0, j, [=] __boba_host_device__(index_t i)
    {
      row_view(i) = p_view(i);
    });

    checkpoint();

    // Solve: U(p(1:j-1), 1:j-1) * c = U(p(1:j-1), j)
    boba::Matrix<space, data_t> U_p =
      U.extract_rows(row_indices_p).extract_columns(col_indices_0_to_j_minus_1);
    boba::Matrix<space, data_t> U_pj_matrix =
      U.extract_rows(row_indices_p).extract_columns(col_index_j);

    U_p.rename("DEIM_U_p");
    U_pj_matrix.rename("DEIM_U_pj");

    checkpoint();

    boba::Matrix<space, data_t> c_matrix = backsolve(U_p, U_pj_matrix);
    auto c = flatten(c_matrix);
    c.rename("DEIM_c");

    checkpoint();

    // Compute residual: r = U(:, j) - U(:, 1:j-1) * c
    boba::Matrix<space, data_t> U_0_to_j_minus_1 =
      U.extract_columns(col_indices_0_to_j_minus_1);
    boba::Matrix<space, data_t> U_j = U.extract_columns(col_index_j);

    auto U_c = U_0_to_j_minus_1 * c;
    auto r_matrix = U_j - reshape_to_matrix(U_c, {n, static_cast<index_t>(1)});
    auto r = flatten(r_matrix);
    r.rename("DEIM_residual");

    checkpoint();

    // Select index with maximum absolute residual
    auto [r_max_val, r_max_idx] = r.max_abs_loc_reduce();
    p_view(j) = r_max_idx;

    checkpoint();
  }

  checkpoint();
  BOBA_CALI_END("DEIM_iteration");
  return p;
}

/**
 * \brief CUR decomposition using the DEIM algorithm.
 *
 * Computes `A ~= C * U * R` where `C` and `R` are selected columns and rows of `A`.
 */
template <execution_space space, typename data_t>
struct CUR
{
  using real_data_t = real_type_t<data_t>;

  // Parameters
  real_data_t tolerance_relative = real_data_t(1e-6);
  real_data_t tolerance_absolute = real_data_t(1e-12);
  size_t kick_rank = 2;
  size_t max_rank = 100000;
  size_t final_rank = 0;

  // CUR factors: A \approx C * U * R
  boba::Matrix<space, data_t> C; // m x k selected columns
  boba::Matrix<space, data_t> U; // k x k connecting matrix
  boba::Matrix<space, data_t> R; // k x n selected rows

  /**
   * \brief default constructor
   */
  CUR()
  {
    C.rename("C");
    U.rename("U");
    R.rename("R");
  }

  /**
   * \brief copy constructor
   */
  CUR(CUR const&) = default;
  /**
   * \brief move constructor
   */
  CUR(CUR&&) = default;
  /**
   * \brief copy assignment operator
   */
  CUR& operator=(CUR const&) = default;
  /**
   * \brief move assignment operator
   */
  CUR& operator=(CUR&&) = default;
  /**
   * \brief destructor
   */
  ~CUR() = default;

  /**
   * \brief Reconstructs the CUR approximation from the stored factors.
   * \return Matrix reconstructed as `C * U * R`.
   */
  [[nodiscard]]
  Matrix<space, data_t> reform_matrix()
  {
    BOBA_CALI_BEGIN("CUR_reform");
    checkpoint();
    auto CU = C * U;
    checkpoint();
    auto CUR_result = CU * R;
    checkpoint();
    BOBA_CALI_END("CUR_reform");
    return CUR_result;
  }

  /**
   * \brief Computes the middle CUR factor from the intersection matrix.
   * \param W Intersection matrix formed from selected rows and columns.
   * \return Pseudoinverse-based middle factor.
   */
  [[nodiscard]]
  boba::Matrix<space, data_t> compute_U_matrix(
    boba::Matrix<space, data_t>& W)
  {
    BOBA_CALI_BEGIN("compute_U_matrix");
    checkpoint();

    W.rename("CUR_W");

    checkpoint();
    BOBA_CALI_SWITCH("compute_U_matrix", "compute_svd");

    boba::SVD<space, data_t> svd_solver;
    svd_solver.tolerance_relative = tolerance_relative;
    svd_solver.tolerance_absolute = tolerance_absolute;
    svd_solver(W);

    checkpoint();
    BOBA_CALI_SWITCH("compute_svd", "construct_pseudoinverse");

    size_t k = svd_solver.significant_singular_values;
    if (k == 0)
    {
      k = 1;
    }

    boba::Matrix<space, data_t> S_inv({k, k});
    S_inv.rename("CUR_S_inv");
    S_inv.fill_with_zeros();

    auto S_inv_view = S_inv.view();
    auto S_view = svd_solver.S.const_view();

    ::boba::detail::loop<space>(0, static_cast<index_t>(k), [=] __boba_host_device__(index_t i)
    {
      S_inv_view({i, i}) = data_t(1.0) / S_view(i);
    });

    checkpoint();

    auto V_S_inv = svd_solver.V * S_inv;
    checkpoint();
    auto U_conj_transpose = svd_solver.U.conjugate_transpose();
    checkpoint();
    auto W_pinv = V_S_inv * U_conj_transpose;
    W_pinv.rename("CUR_W_pinv");

    checkpoint();
    BOBA_CALI_END("construct_pseudoinverse");
    BOBA_CALI_END("compute_U_matrix");
    return W_pinv;
  }

  /**
   * \brief Computes a CUR decomposition of a matrix.
   * \param A Matrix to factor.
   */
  void operator()(boba::Matrix<space, data_t>& A)
  {
    BOBA_CALI_BEGIN("CUR_decomposition");
    checkpoint();

    size_t m = A.rows();
    size_t n = A.cols();
    size_t current_max_rank = boba::min(boba::min(m, n), max_rank);

    A.rename("CUR_input");

    checkpoint();
    BOBA_CALI_SWITCH("CUR_decomposition", "compute_svd_once");

    // Compute SVD once
    boba::Matrix<space, data_t> A_copy = A;
    boba::SVD<space, data_t> svd_solver;
    svd_solver(A_copy);

    // Get left and right singular vectors
    boba::Matrix<space, data_t> U_full = svd_solver.U; // m × min(m,n)
    boba::Matrix<space, data_t> V_full = svd_solver.V; // n × min(m,n)

    checkpoint();
    BOBA_CALI_SWITCH("compute_svd_once", "adaptive_rank_loop");

    // Precompute norm for error checking
    real_data_t A_norm = ::boba::norm_frobenius(A);

    size_t r = 1;

    while (r <= current_max_rank)
    {
      checkpoint();

      // Extract first r singular vectors
      size_t k = boba::min(r, boba::min(U_full.cols(), V_full.cols()));

      checkpoint();
      // Get rank-r basis for rows
      boba::Matrix<space, data_t> basis_rows({m, k});
      for (size_t i = 0; i < m; ++i)
      {
        for (size_t j = 0; j < k; ++j)
        {
          basis_rows({i, j}) = U_full({i, j});
        }
      }
      basis_rows.rename("basis_rows");

      // Get rank-r basis for columns
      boba::Matrix<space, data_t> basis_cols({n, k});
      for (size_t i = 0; i < n; ++i)
      {
        for (size_t j = 0; j < k; ++j)
        {
          basis_cols({i, j}) = V_full({i, j});
        }
      }
      basis_cols.rename("basis_cols");

      checkpoint();
      // DEIM selects k row indices and k column indices
      auto row_indices = DEIM(basis_rows);
      auto col_indices = DEIM(basis_cols);

      checkpoint();

      // Extract C (m × k), R (k × n), W (k × k) from A
      C = A.extract_columns(col_indices);
      C.rename("C");

      R = A.extract_rows(row_indices);
      R.rename("R");

      checkpoint();

      boba::Matrix<space, data_t> W =
        A.extract_rows(row_indices).extract_columns(col_indices);

      U = compute_U_matrix(W);
      U.rename("U");

      checkpoint();

      // Compute A \approx C * U * R
      auto A_approx = reform_matrix();
      checkpoint();
      A_approx.rename("A_approx");

      checkpoint();

      real_data_t rel_error = ::boba::norm_difference_frobenius(A, A_approx) / A_norm;

      checkpoint();

      // Check convergence
      bool rel_converged = rel_error < tolerance_relative;
      bool abs_converged = rel_error * A_norm < tolerance_absolute;

      if (rel_converged or abs_converged)
      {
        final_rank = k;
        checkpoint();
        BOBA_CALI_END("adaptive_rank_loop");
        return;
      }

      r += kick_rank;
      checkpoint();
    }

    // Maximum rank reached
    final_rank = boba::min(r - kick_rank, current_max_rank);

    checkpoint();
    BOBA_CALI_END("adaptive_rank_loop");
  }
};

} // namespace boba
