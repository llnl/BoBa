// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *
 * Implements a basic version of the exp(tt) function from
 *
 * Martin Eigel, Nando Farchmin, Sebastian Heidenreich, and Philipp Trunschke
 * (2023). Efficient approximation of high-dimensional exponentials by tensor
 * networks. International Journal for Uncertainty Quantification 13(1).
 * doi: 10.1615/Int.J.UncertaintyQuantification.2022039164
 * arXiv:2105.09064
 *
 * The shifted exponential $w(x) = \exp(h(x) - h(0)) - 1$ is cast as the solution
 * of a system of partial differential equations:
 * \[
 * \partial_{x_k} w(x) - w(x) \partial_{x_k} h(x) = \partial_{x_k} h(x) \forall k,
 * w(0) = 0.
 * \]
 * The operators are then discretized, and the exponential is obtained by solving
 * the linear system.
 *
 */

#pragma once

/**
 * @brief Eigel exponential function.
 *
 * @tparam dimension
 * @tparam operator_t
 * @tparam vector_t
 * @tparam data_t
 *
 * @param[in] exponent
 * @param[in] gradient_operator
 * @param[in] evaluation_at_origin_operator
 * @param[in] exponential_at_origin
 * @param[in] solver
 *
 * @return Exponential vector.
 */
template <size_t dimension, typename operator_t, typename vector_t, typename data_t>
vector_t
eigel_exponential(const vector_t& exponent,
                  const ::boba::Array<operator_t, dimension>& gradient_operator,
                  const operator_t& evaluation_at_origin_operator,
                  const data_t& exponential_at_origin,
                  ::boba::Krylov<operator_t, vector_t>& solver)
{
  checkpoint();

  const auto sizes = exponent.sizes();

  // used to enforce the initial condition, w(0) = 0
  operator_t eigel_operator = evaluation_at_origin_operator.transpose() * evaluation_at_origin_operator;

  vector_t eigel_rhs(sizes);
  eigel_rhs.fill_with_zeros();

  for (size_t k = 0; k < dimension; ++k)
  {
    // discretization of one component of the gradient system,
    // \partial_{x_k} w(x) - w(x) \partial_{x_k} h(x) = \partial_{x_k} h(x),
    // for unknown w(x)
    const vector_t exponent_gradient_k = gradient_operator[k] * exponent;
    const operator_t diag_exponent_gradient_k = diagonalize(exponent_gradient_k);
    const operator_t eigel_pde_operator_k = gradient_operator[k] - diag_exponent_gradient_k;
    const operator_t eigel_pde_operator_k_transpose = eigel_pde_operator_k.transpose();

    // we are using a normal system to find the least square solution of the combined rectangular system
    eigel_operator += eigel_pde_operator_k_transpose * eigel_pde_operator_k;
    eigel_rhs += eigel_pde_operator_k_transpose * exponent_gradient_k;
  }

  checkpoint();
  eigel_operator.round();
  eigel_rhs.round();

  // solve the normal system
  vector_t eigel_solution(sizes);
  eigel_solution.fill_with(1.0);

  vector_t eigel_initial_guess(sizes);
  eigel_initial_guess.fill_with(1.0);

  solver.set_matrix(eigel_operator);

  checkpoint();
  solver.solve(eigel_rhs, eigel_initial_guess, eigel_solution);

  // compute the true exponential by scaling and shifting w(x)
  vector_t ones(sizes);
  ones.fill_with(1.0);

  checkpoint();
  vector_t exponential = exponential_at_origin * (eigel_solution + ones);
  exponential.round();

  return exponential;
}
