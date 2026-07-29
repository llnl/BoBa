// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

/*
  Test tensor train specific solvers, such as AMeN and DMRG
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::seconds;

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
  size_t max_direct_solve_size = 1500;
  size_t solve3d_2ml_method = 2; // This is by far the fastest for DRMG at present
  double solve3d_2ml_tolerance_relative = 1.0e-3;
  double dx = 1.0;
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

template <typename solver_t>
void configure_solver(solver_t& tt_solver, const input& parameters)
{
  tt_solver.max_direct_solve_size = parameters.max_direct_solve_size;
  tt_solver.convergence_tolerance = parameters.convergence_tolerance;
  tt_solver.solve3d_2ml_options.method = parameters.solve3d_2ml_method;
  tt_solver.solve3d_2ml_options.tolerance_relative = parameters.solve3d_2ml_tolerance_relative;
}

template <size_t dimension, typename solver_t>
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

  checkpoint();

  boba::TensorTrain<dimension, space, double> ttb(sizes);
  ttb.fill_with(1.0);
  auto tt_initial_guess = ttb;

  checkpoint();
  solver_t tt_solver;
  configure_solver(tt_solver, parameters);

  boba::TicToc<tictoc_units> timer;
  timer.tic();
  auto solution = tt_solver.solve(A_operator, ttb, tt_initial_guess);
  timer.end_and_print(solver_label);

  auto exact = make_exact_solution(sizes, parameters.dx);

  auto solution_decompress = solution.decompress();
  auto error = solution_decompress - exact;

  auto solution_norm = ::boba::norm_frobenius(solution);
  auto exact_norm = ::boba::norm_frobenius(exact);
  auto error_norm_abs = ::boba::norm_frobenius(error);
  auto error_norm_relative = error_norm_abs / exact_norm;

  boba_print(solution_norm);
  boba_print(exact_norm);
  boba_print(error_norm_abs);
  boba_print(error_norm_relative);

  pass_or_fail(check, error_norm_relative, 0.3 * double(64 + 1) / double(parameters.N + 1));
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
    run_dimension_with_solver<dimension, boba::TensorTrainAMEN<double>>(check, parameters, "TensorTrainAMEN solver time:");
    break;
  case solver_selection::dmrg:
    run_dimension_with_solver<dimension, boba::TensorTrainDMRG<double>>(check, parameters, "TensorTrainDMRG solver time:");
    break;
  default:
    boba_error("Invalid solver selector. Use --which 0 for AMEn or 1 for DMRG.");
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

  args.add_optional_argument(which,
                             "-w",
                             "--which",
                             "Which solver to run: 0=AMEn, 1=DMRG.");

  args.parse_check();

  if ((dimension == 0) or (dimension == 2))
  {
    //
    boba_print("Running 2D test");
    //
    run_dimension<2>(check, parameters, which);
  }
  if ((dimension == 0) or (dimension == 3))
  {
    //
    boba_print("Running 3D test");
    //
    run_dimension<3>(check, parameters, which);
  }
  if ((dimension == 0) or (dimension == 4))
  {
    //
    boba_print("Running 4D test");
    //
    run_dimension<4>(check, parameters, which);
  }
  if ((dimension == 0) or (dimension == 6))
  {
    //
    boba_print("Running 6D test");
    //
    run_dimension<6>(check, parameters, which);
  }

  boba::finalize();
  return final_check(check);
}
