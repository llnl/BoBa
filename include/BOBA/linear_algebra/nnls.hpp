// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace boba
{

/**
 * @brief Solve a nonnegative least squares (NNLS) problem.
 *
 * This function solves the constrained least squares problem
 * @f[
 *   \min_x \; \| A x - b \|_2
 *   \quad \text{subject to} \quad x \ge 0
 * @f]
 * where all inequalities are interpreted componentwise.
 *
 * The interface is analogous to MATLAB's @c lsqnonneg(A, b); it computes a
 * vector @c x that minimizes the 2-norm of the residual while enforcing
 * elementwise nonnegativity.
 *
 * The implementation is based on the classic Lawson and Hanson active set
 * algorithm for NNLS. Internally, it iteratively:
 *   - Maintains two index sets:
 *       - The passive set @c P containing variables allowed to be positive
 *       - The active set @c R containing variables fixed at zero
 *   - Solves an unconstrained least squares problem on the current passive set
 *   - Projects back onto the nonnegative orthant if any components become
 *     nonpositive
 *   - Updates the sets using gradient information until the KKT conditions
 *     are approximately satisfied or a maximum iteration count is reached
 *
 * @tparam space
 * @tparam data_t
 * @tparam index_t
 *
 * @param A_in
 *   Coefficient matrix @c A of size @f$m \times n@f$. The number of rows
 *   @f$m@f$ must match the length of @p b_in. The problem is typically
 *   overdetermined (@f$m \ge n@f$), but any least squares configuration
 *   is allowed.
 *
 * @param b_in
 *   Right hand side vector @c b of length @f$m@f$, representing the
 *   target values in @f$\| A x - b \|_2@f$.
 *
 * @param maxIter
 *   Maximum number of outer/inner iterations of the active set algorithm.
 *   This acts as a safeguard against pathological cases. The default is
 *   @c 3000. If the maximum is reached before convergence, the current
 *   iterate is returned; callers can detect this via convergence criteria
 *   implemented outside this function if needed.
 *
 * @param tol
 *   Tolerance used for:
 *     - Detecting sufficiently small/negative candidate coefficients in the
 *       passive solution (to decide whether to move variables back to the
 *       active set),
 *     - Determining whether the dual variables (gradient entries) are
 *       "sufficiently positive" to justify moving indices into the passive set,
 *     - Comparing values against zero when enforcing nonnegativity.
 *   The default is @c 1e-12 for double precision. For single precision input
 *   you may wish to use a larger tolerance (for example @c 1e-6).
 *
 * @return
 *   A @c boba::Vector<space, data_t, size_t> of length @f$n@f$ containing
 *   the NNLS solution @c x. All entries satisfy @f$x_i \ge 0@f$ up to the
 *   provided tolerance.
 *
 * @note
 *   - Numerical properties depend on the conditioning of @c A. For highly
 *     ill conditioned problems, you may see sensitivity to @p tol and
 *     @p maxIter.
 *
 * @see
 *   Lawson, C. L., & Hanson, R. J. (1974). Solving Least Squares Problems.
 *   Prentice Hall.
 *
 * @warning
 *   This routine does not currently return an explicit convergence flag.
 *   If you need to distinguish between "converged" and "maxIter reached,"
 *   you should either:
 *     - Wrap this function and track convergence criteria externally, or
 *     - Extend the interface to return auxiliary information
 *       (for example residual norm, number of iterations, and exit code).
 */

template <boba::execution_space space, typename data_t>
[[nodiscard]]
boba::Vector<space, data_t> nnls(
  const boba::Matrix<space, data_t>& A_in,
  const boba::Vector<space, data_t>& b_in,
  size_t maxIter = 3000,
  data_t tol = 1e-12)
  requires(space == boba::host_space)
{
  const size_t rows = A_in.rows();
  const size_t cols = A_in.cols();

  boba_always_assert_equal(b_in.size(), rows, "Dimension mismatch between A and b");

  boba::Vector<space, data_t> solution({cols});
  solution.fill_with(data_t(0.0));

  if (::boba::norm_frobenius(b_in) < tol)
  {
    return solution;
  }

  // Sets: P = passive set (indices allowed to be > 0)
  //       R = active set (indices forced to 0)
  std::vector<bool> inPassive(cols, false);

  auto A_inT = A_in.transpose();

  boba::Vector<space, data_t> gradient = A_inT * (b_in - A_in * solution);

  size_t iter = 0;

  bool is_iterating = true;
  while (is_iterating)
  {
    // 1) Find index in R where w is most positive
    data_t wMax = 0.0;
    int t = -1;
    for (size_t j = 0; j < cols; ++j)
    {
      if (!inPassive.at(j) && gradient({j}) > wMax + tol)
      {
        wMax = gradient({j});
        t = j;
      }
    }

    if (t == -1)
    {
      // Optimality condition satisfied: no positive gradient in R
      break;
    }

    // Move t from R to P
    inPassive[static_cast<size_t>(t)] = true;

    // Inner loop variables
    auto x_passive = solution;
    detail::ignore(x_passive);
    bool is_inner_iterating = true;
    size_t inner_iter = 0;
    while (is_inner_iterating)
    {
      // 2) Solve least squares on passive set:
      //    A_P * z = b, where z >= 0 on P and 0 on R
      size_t pCount = 0;
      for (bool v : inPassive)
      {
        if (v)
        {
          ++pCount;
        }
      }

      if (pCount == 0)
      {
        break; // defensive, should not happen
      }

      boba::Matrix<space, data_t> A_P({rows, pCount});
      boba::Vector<space, index_t> idxP({pCount});

      for (size_t j = 0, k = 0; j < cols; ++j)
      {
        if (inPassive.at(j))
        {
          for (size_t l = 0; l < A_P.rows(); l++)
          {
            A_P({l, k}) = A_in({l, j});
          }
          idxP({k}) = j;
          ++k;
        }
      }

      // z_P = (A_P^T A_P)^{-1} A_P^T b   (normal equations)
      auto APt = A_P.transpose();
      auto AtA = APt * A_P;
      auto Atb = APt * b_in;

      auto z_P = boba::backsolve(AtA, Atb);

      // Build full z with zeros outside P
      boba::Vector<space, data_t> z({cols});
      z.fill_with(0.0);
      for (size_t k = 0; k < pCount; ++k)
      {
        z({idxP({k})}) = z_P({k});
      }

      // 3) Check if any z_j <= 0 in P. If so, step back toward x
      bool anyNegative = false;
      data_t alpha = 1.0;

      for (size_t j = 0; j < cols; ++j)
      {
        if (inPassive.at(j) && z({j}) <= tol)
        {
          anyNegative = true;
          // Compute maximum step alpha in [0,1] so that x + alpha*(z - x) >= 0
          if (solution({j}) - z({j}) > tol)
          {
            data_t alpha_j = solution({j}) / (solution({j}) - z({j}));
            if (alpha_j < alpha)
            {
              alpha = alpha_j;
            }
          }
        }
      }

      // Update x along direction z - x
      solution += alpha * (z - solution);

      // Remove variables that became nonpositive from P
      for (size_t j = 0; j < cols; ++j)
      {
        if (inPassive.at(j) && solution({j}) <= tol)
        {
          solution({j}) = 0.0;
          inPassive.at(j) = false;
        }
      }

      inner_iter++;
      bool is_nonnegative = !anyNegative;

      if (is_nonnegative || (inner_iter > maxIter))
      {
        is_inner_iterating = false;
      }
    }

    // 4) Recompute gradient
    gradient = A_inT * (b_in - A_in * solution);

    iter++;

    if (iter > maxIter)
    {
      std::cerr << "nnls: maximum iteration limit reached" << std::endl;
      is_iterating = false;
    }
  }

  // Enforce nonnegativity explicitly
  for (size_t j = 0; j < cols; ++j)
  {
    if (solution({j}) < 0.0)
    {
      solution({j}) = 0.0;
    }
  }

  return solution;
}

/**
 * \brief Reports that device-side NNLS is not yet implemented.
 * \param A Input coefficient matrix.
 * \param b Input right-hand side vector.
 * \param maxIter Maximum iteration count requested by the caller.
 * \param tol Tolerance requested by the caller.
 * \return An empty vector after reporting the unsupported path.
 */
template <boba::execution_space space, typename data_t>
[[nodiscard]]
boba::Vector<space, data_t> nnls(
  const boba::Matrix<space, data_t>& A,
  const boba::Vector<space, data_t>& b,
  size_t maxIter = 3000,
  data_t tol = 1e-12)
  requires(not(space == boba::host_space))
{
  detail::ignore(A, b, maxIter, tol);
  boba_error("Not yet implemented");
  return boba::Vector<space, data_t>{};
}

} // namespace boba
