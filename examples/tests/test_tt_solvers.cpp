// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

/*
  Test tensor train specific solvers, such as AMeN and DMRG
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

struct input
{
  input()
  {
    boba::TensorTrainDMRG<double> get_solver_defaults;
    convergence_tolerance = get_solver_defaults.convergence_tolerance;
    max_direct_solve_size = get_solver_defaults.max_direct_solve_size;
    solve3d_2ml_method = get_solver_defaults.solve3d_2ml_options.method;
    solve3d_2ml_tolerance_relative = get_solver_defaults.solve3d_2ml_options.tolerance_relative;
  }

  size_t N = 4;
  double convergence_tolerance = 1.0e-4;
  int max_direct_solve_size = 1500;
  size_t solve3d_2ml_method = 2; // This is by far the fastest for DRMG at present
  double solve3d_2ml_tolerance_relative = 1.0e-3;
  size_t max_sweeps = 20;
  size_t minimum_sweeps = 1;
  size_t kickrank = 4;
  size_t max_allowed_ranks = 1000;
  size_t random_seed = 0;
  double dx = 1.0;
  bool timing_only = false;
  bool verbose = false;
  bool report_analytic_error_norms = false;
  bool no_row_norm_preconditioner = false;
};

enum class solver_selection : size_t
{
  amen = 0,
  dmrg = 1
};

template <size_t dimension>
boba::Tensor<dimension, space, double> make_exact_solution(
  boba::Array<size_t, dimension> sizes,
  double dx)
{
  BOBA_CALI_EXTERNAL_MARK

  ::boba::detail::ignore(dx);
  checkpoint();
  boba::Tensor<dimension, space, double> exact(sizes);
  exact.rename("exact");
  checkpoint();
  size_t exact_solution_expansion_terms = 7;
  boba::Multiindexer<dimension> expansion_terms(boba::filled_array<dimension>(exact_solution_expansion_terms));

  auto exact_view = exact.view();

  ::boba::loop<space, 1>(exact_view.size(),
                         [=] __boba_host_device__(size_t id)
  {
    auto mid = exact_view.multiindex(id);

    auto solution = 0.0;

    for (size_t k = 0; k < expansion_terms.size(); k++)
    {
      auto expansion_mid = expansion_terms.multiindex(k);
      auto odd_expansion_mid = 1_z + 2_z * expansion_mid;
      auto dbl_mid = boba::cast<double>(odd_expansion_mid);
      auto mid_pi = dbl_mid * boba::pi;
      auto sum_mid_pi_2 = boba::sum(mid_pi * mid_pi);
      auto prod_mid_pi = boba::product(mid_pi);
      auto coefficient = -boba::pow(4.0, double(dimension)) / (sum_mid_pi_2 * prod_mid_pi);
      boba::Array<double, dimension> x;
      for (size_t dim = 0; dim < dimension; dim++)
      {
        x[dim] = static_cast<double>(mid[dim] + 1) / static_cast<double>(sizes[dim] + 1);
      }
      auto mid_pi_x = boba::sin(mid_pi * x);
      solution += coefficient * boba::product(mid_pi_x);
    }
    exact_view(mid) = solution;
  });
  return exact;
}

/** @brief Builds the sine-series reference as a tensor train without a full spatial tensor. */
template <size_t dimension>
boba::TensorTrain<dimension, space, double> make_exact_solution_tt(
  boba::Array<size_t, dimension> sizes)
{
  constexpr size_t expansion_terms_per_dimension = 7;
  const auto expansion_sizes = boba::filled_array<dimension>(expansion_terms_per_dimension);
  boba::Multiindexer<dimension> expansion_terms(expansion_sizes);
  boba::Tensor<dimension, host_space, double> coefficients(expansion_sizes);
  auto coefficients_view = coefficients.view();

  for (size_t term = 0; term < expansion_terms.size(); term++)
  {
    const auto frequencies = expansion_terms.multiindex(term);
    double sum_frequency_squared = 0.0;
    double product_frequency = 1.0;
    for (size_t d = 0; d < dimension; d++)
    {
      const double frequency_pi = static_cast<double>(2 * frequencies[d] + 1) * ::boba::pi;
      sum_frequency_squared += frequency_pi * frequency_pi;
      product_frequency *= frequency_pi;
    }
    coefficients_view(frequencies) = -::boba::pow(4.0, static_cast<double>(dimension)) /
                                     (sum_frequency_squared * product_frequency);
  }

  auto coefficient_tt = boba::compress_to_TensorTrain(coefficients, 1.0e-14, 1.0e-14);
  boba::TensorTrain<dimension, space, double> exact(sizes);
  for (size_t d = 0; d < dimension; d++)
  {
    const auto coefficient_view = coefficient_tt.cores[d].const_view();
    const auto left_rank = coefficient_tt.cores[d].sizes(0);
    const auto right_rank = coefficient_tt.cores[d].sizes(2);
    boba::Tensor<3, host_space, double> lifted_core({left_rank, sizes[d], right_rank});
    auto lifted_view = lifted_core.view();

    for (size_t left = 0; left < left_rank; left++)
    {
      for (size_t i = 0; i < sizes[d]; i++)
      {
        for (size_t right = 0; right < right_rank; right++)
        {
          double value = 0.0;
          for (size_t frequency = 0; frequency < expansion_terms_per_dimension; frequency++)
          {
            const double angle = static_cast<double>(2 * frequency + 1) * ::boba::pi *
                                 static_cast<double>(i + 1) / static_cast<double>(sizes[d] + 1);
            value += coefficient_view({left, frequency, right}) * ::boba::sin(angle);
          }
          lifted_view({left, i, right}) = value;
        }
      }
    }
    exact.cores[d] = lifted_core;
  }
  return exact;
}

/**
 * @brief Computes the tensor-train solution error against the manufactured solution.
 * @note Copies the reference to the solution's execution space when they differ.
 */
template <size_t dimension, boba::execution_space solution_space, boba::execution_space exact_space>
std::pair<double, double> analytic_error_norms(
  const boba::TensorTrain<dimension, solution_space, double>& solution,
  const boba::TensorTrain<dimension, exact_space, double>& exact)
{
  if constexpr (solution_space != exact_space)
  {
    boba::TensorTrain<dimension, solution_space, double> exact_in_solution_space = exact;
    return analytic_error_norms(solution, exact_in_solution_space);
  }
  else
  {
    const double error_norm_abs = ::boba::norm_difference_frobenius(solution, exact);
    return {error_norm_abs, error_norm_abs / ::boba::norm_frobenius(exact)};
  }
}

template <typename solver_t>
void configure_solver(solver_t& tt_solver, const input& parameters)
{
  using solver_size_t = std::remove_cvref_t<decltype(tt_solver.max_direct_solve_size)>;
  if constexpr (std::is_unsigned_v<solver_size_t>)
  {
    boba_always_assert(parameters.max_direct_solve_size >= 0, "DMRG max_direct_solve_size must be nonnegative.");
  }
  tt_solver.max_direct_solve_size = static_cast<solver_size_t>(parameters.max_direct_solve_size);
  tt_solver.convergence_tolerance = parameters.convergence_tolerance;
  tt_solver.solve3d_2ml_options.method = parameters.solve3d_2ml_method;
  tt_solver.solve3d_2ml_options.tolerance_relative = parameters.solve3d_2ml_tolerance_relative;
  tt_solver.solve3d_2ml_options.use_row_norm_preconditioner = !parameters.no_row_norm_preconditioner;
  if constexpr (requires { tt_solver.max_sweeps; })
  {
    tt_solver.max_sweeps = parameters.max_sweeps;
    tt_solver.kickrank = parameters.kickrank;
    tt_solver.max_allowed_ranks = parameters.max_allowed_ranks;
  }
  if constexpr (requires { tt_solver.minimum_sweeps; })
  {
    boba_always_assert(parameters.minimum_sweeps <= parameters.max_sweeps,
                       "AMEn minimum_sweeps must not exceed max_sweeps.");
    tt_solver.minimum_sweeps = parameters.minimum_sweeps;
  }
  if constexpr (requires { tt_solver.verbose; })
  {
    tt_solver.verbose = parameters.verbose;
  }
}

template <size_t dimension>
boba::TensorTrainMatrix<dimension, space, double> make_operator(
  input& parameters,
  boba::Array<size_t, dimension> sizes)
{
  boba::TensorTrainMatrix<dimension, space, double> A_operator(sizes, sizes);
  A_operator.fill_with_zeros();

  for (size_t dim = 0; dim < dimension; dim++)
  {
    boba::Array<boba::Matrix<space, double>, dimension> array_of_matrices;
    for (size_t d = 0; d < dimension; d++)
    {
      auto N = sizes[d];
      auto dx = 1.0 / double(N + 1);
      boba::Matrix<host_space, double> L({N, N});
      boba::Matrix<host_space, double> I({N, N});

      if (N <= 2)
      {
        L.fill_with_zeros();
        I.fill_with_zeros();
        for (size_t row = 0; row < N; row++)
        {
          I({row, row}) = 1.0;
          L({row, row}) = 2.0;
          if (row > 0)
          {
            L({row, row - 1}) = -1.0;
          }
          if (row + 1 < N)
          {
            L({row, row + 1}) = -1.0;
          }
        }
        L *= -1.0 / (dx * dx);
      }
      else
      {
        common_ttm<2, space, double> operators(N, 1.0, false);
        L = boba::reshape_to_matrix(-operators.laplacian_1d_interior, {N, N});
        I = boba::reshape_to_matrix(operators.identity_1d, {N, N});
        L.fill_row(0, 0.0);
        L.fill_row(N - 1, 0.0);
        L({0, 0}) = 2.0;
        L({0, 1}) = -1.0;
        L({N - 1, N - 2}) = -1.0;
        L({N - 1, N - 1}) = 2.0;
        L *= -1.0 / (dx * dx);
      }
      parameters.dx = dx;

      boba::Matrix<space, double> L_device = L;
      boba::Matrix<space, double> I_device = I;

      array_of_matrices[d] = (dim == d) ? L_device : I_device;
    }
    A_operator += boba::make_ttm_from_matrices(array_of_matrices);
  }
  return A_operator;
}

template <size_t dimension>
void check_sparse_operator(
  bool& check,
  const boba::TensorTrainMatrix<dimension, space, double>& dense_operator,
  const boba::SparseTensorTrainMatrix<dimension, space, double>& sparse_operator,
  const boba::TensorTrain<dimension, space, double>& tt_initial_guess,
  bool timing_only)
{
  pass_or_fail_bool(check, sparse_operator.rows() == dense_operator.rows());
  pass_or_fail_bool(check, sparse_operator.cols() == dense_operator.cols());
  pass_or_fail_bool(check, sparse_operator.get_number_elements() > 0);

  if (not timing_only)
  {
    auto dense_application = dense_operator.TensorTrainMatrix_vector_multiply(tt_initial_guess);
    auto sparse_application = sparse_operator.TensorTrainMatrix_vector_multiply(tt_initial_guess);
    pass_or_fail(check, ::boba::norm_difference_frobenius(sparse_application, dense_application), 1.0e-10);
  }
}

template <size_t dimension, typename solver_t, bool use_sparse_operator>
void run_dimension_with_solver(
  bool& check,
  input parameters,
  const char* solver_label)
{
  auto sizes = ::boba::filled_array<dimension>(parameters.N);

  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] += d;
  }

  checkpoint();
  auto dense_operator = make_operator<dimension>(parameters, sizes);

  using dense_operator_t = boba::TensorTrainMatrix<dimension, space, double>;
  using sparse_operator_t = boba::SparseTensorTrainMatrix<dimension, space, double>;
  using operator_t = std::conditional_t<use_sparse_operator, sparse_operator_t, dense_operator_t>;
  operator_t operator_to_solve;

  if constexpr (use_sparse_operator)
  {
    operator_to_solve = boba::from_dense(dense_operator);
  }
  else
  {
    operator_to_solve = dense_operator;
  }

  checkpoint();

  auto exact_tt = make_exact_solution_tt<dimension>(sizes);
  auto ttb = operator_to_solve.TensorTrainMatrix_vector_multiply(exact_tt);
  boba::TensorTrain<dimension, space, double> tt_initial_guess(sizes);
  tt_initial_guess.fill_with(1.0);

  if constexpr (use_sparse_operator)
  {
    check_sparse_operator(check, dense_operator, operator_to_solve, tt_initial_guess, parameters.timing_only);
  }

  checkpoint();
  solver_t tt_solver;
  configure_solver(tt_solver, parameters);

  if (parameters.random_seed != 0)
  {
    boba::random::set_seed(parameters.random_seed);
  }

  boba::TicToc<tictoc_units> timer;
  timer.tic();
  auto solution = tt_solver.solve(operator_to_solve, ttb, tt_initial_guess);
  timer.end_and_print(solver_label);

  if (parameters.report_analytic_error_norms)
  {
    boba::TicToc<tictoc_units> error_timer;
    error_timer.tic();
    const auto [analytic_error_norm_abs, analytic_error_norm_relative] = analytic_error_norms(solution, exact_tt);
    error_timer.end_and_print("Analytic error norm time:");
    boba_print(analytic_error_norm_abs);
    boba_print(analytic_error_norm_relative);
  }

  if (parameters.timing_only)
  {
    return;
  }

  auto exact = make_exact_solution(sizes, parameters.dx);
  auto exact_tt_dense = exact_tt.decompress();
  auto exact_norm = ::boba::norm_frobenius(exact);
  auto manufactured_solution_error = ::boba::norm_difference_frobenius(exact_tt_dense, exact) / exact_norm;
  pass_or_fail(check, manufactured_solution_error, 1.0e-10);

  auto solution_decompress = solution.decompress();
  auto error = solution_decompress - exact;

  boba::TensorTrain<dimension, space, double> solution_in_execution_space = solution;
  auto residual = dense_operator * solution_in_execution_space;
  residual -= ttb;

  auto solution_norm = ::boba::norm_frobenius(solution);
  auto error_norm_abs = ::boba::norm_frobenius(error);
  auto error_norm_relative = error_norm_abs / exact_norm;
  auto residual_norm_relative = ::boba::norm_frobenius(residual) / ::boba::norm_frobenius(ttb);

  boba_print(solution_norm);
  boba_print(exact_norm);
  boba_print(error_norm_abs);
  boba_print(error_norm_relative);
  boba_print(residual_norm_relative);

  pass_or_fail(check, error_norm_relative, 0.3 * double(64 + 1) / double(parameters.N + 1));
  pass_or_fail(check, residual_norm_relative, boba::max(1000.0 * parameters.convergence_tolerance, 1.0e-6));
}

template <size_t dimension>
void run_dimension(
  bool& check,
  input parameters,
  size_t which)
{
  switch (static_cast<solver_selection>(which))
  {
  case solver_selection::amen:
    run_dimension_with_solver<dimension, boba::TensorTrainAMEN<double>, false>(check, parameters, "TensorTrainAMEN solver time:");
    break;
  case solver_selection::dmrg:
    run_dimension_with_solver<dimension, boba::TensorTrainDMRG<double>, false>(check, parameters, "TensorTrainDMRG solver time:");
    break;
  default:
    boba_error("Invalid solver selector. Use --which 0 for AMEn or 1 for DMRG.");
  }
}

template <size_t dimension>
void run_selected_dimension(
  bool& check,
  input parameters,
  size_t selected_dimension,
  size_t which,
  bool sparse_solver)
{
  if ((selected_dimension != 0) and (selected_dimension != dimension))
  {
    return;
  }

  boba_print("Running " + std::to_string(dimension) + "D test");
  if (sparse_solver)
  {
    run_dimension_with_solver<dimension, boba::TensorTrainAMEN<double>, true>(
      check, parameters, "TensorTrainAMEN sparse solver time:");
  }
  else
  {
    run_dimension<dimension>(check, parameters, which);
  }
}

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  boba_print("Test for TT solvers (AMEn and DMRG)");

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  size_t dimension = 0;
  size_t which = 1;
  bool sparse_solver = false;
  input parameters;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(parameters.N,
                             "-N",
                             "--resolution",
                             "Number of rows and columns.");

  args.add_optional_argument(parameters.convergence_tolerance,
                             "-T",
                             "--convergence_tolerance",
                             "DMRG Convergence tolerance.");

  args.add_optional_argument(dimension,
                             "-d",
                             "--dimension",
                             "Which problem dimension to run.");

  args.add_optional_argument(parameters.max_direct_solve_size,
                             "-M",
                             "--max_direct_solve_size",
                             "Maximum matrix rows or columns before which the direct solver will be used and after which the iterative solver will be used.");

  args.add_optional_argument(parameters.solve3d_2ml_method,
                             "-m",
                             "--solve3d_2ml_method",
                             "DMRG's iterative method.");

  args.add_optional_argument(parameters.solve3d_2ml_tolerance_relative,
                             "-t",
                             "--solve3d_2ml_tolerance_relative",
                             "DMRG's iterative method tolerance.");

  args.add_optional_argument(parameters.no_row_norm_preconditioner,
                             "",
                             "--no_row_norm_preconditioner",
                             "Disable row-norm left preconditioning in dense operator GMRES.");

  args.add_optional_argument(parameters.max_sweeps,
                             "",
                             "--max_sweeps",
                             "Maximum number of AMEn sweeps.");

  args.add_optional_argument(parameters.minimum_sweeps,
                             "",
                             "--minimum_sweeps",
                             "Minimum number of AMEn sweeps before checking convergence.");

  args.add_optional_argument(parameters.kickrank,
                             "",
                             "--kickrank",
                             "AMEn residual enrichment rank.");

  args.add_optional_argument(parameters.max_allowed_ranks,
                             "",
                             "--max_allowed_ranks",
                             "Maximum AMEn TT rank during enrichment.");

  args.add_optional_argument(parameters.random_seed,
                             "",
                             "--random_seed",
                             "Seed for solver randomness; zero keeps the default random seed.");

  args.add_optional_argument(parameters.timing_only,
                             "",
                             "--timing_only",
                             "Run only the solver timing path without reference-solution checks.");

  args.add_optional_argument(parameters.report_analytic_error_norms,
                             "",
                             "--report_analytic_error_norms",
                             "Compute absolute and relative error norms against the manufactured tensor-train solution.");

  args.add_optional_argument(parameters.verbose,
                             "",
                             "--verbose",
                             "Print AMEn sweep residual and rank diagnostics.");

  args.add_optional_argument(which,
                             "-w",
                             "--which",
                             "Which solver to run: 0=AMEn, 1=DMRG.");

  args.add_optional_argument(sparse_solver,
                             "",
                             "--sparse_solver",
                             "Use the sparse tensor-train operator path.");

  args.parse_check();

  run_selected_dimension<2>(check, parameters, dimension, which, sparse_solver);
  run_selected_dimension<3>(check, parameters, dimension, which, sparse_solver);
  run_selected_dimension<4>(check, parameters, dimension, which, sparse_solver);
  run_selected_dimension<5>(check, parameters, dimension, which, sparse_solver);
  run_selected_dimension<6>(check, parameters, dimension, which, sparse_solver);

  boba::finalize();
  return final_check(check);
}
