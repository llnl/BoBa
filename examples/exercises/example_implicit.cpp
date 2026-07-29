// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../tests/common.hpp"
#include "../tests/common_ttm.hpp"

#ifdef BOBA_ENABLE_EIGEN
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/SparseLU"
#endif

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*

  The goal of this example is to solve a coupled operator equation implicitly,
  and thus demonstrates an implicit version of the explicit example.

  d/dt q + {c, c, c}*grad(q) + beta*Lap(q) = kappa*q

  Define
    spatial_operator = {c, c, c}*grad + beta*Lap - kappa

  Discretize in time
  q^{n+1} - q^{n} + dt*spatial_operator*q^{n+1} = 0
  (1 + dt*spatial_operator)*q^{n+1} = q^{n}

  // If beta = 0, we can compute an exact solution
  q_t + c*q_x + c*q_y + c*q_z = kappa*q

  Parametrize along a path s
  q_s = kappa*q, s = t
  q(s) = q(s=0)*exp(kappa*s)
  q(t) = q({x, y, z} - ct)*exp(kappa*t)
*/

enum schemes : size_t
{
  tt = 0,
  full,
  number_of_schemes,
};

static std::string const schemes_strings[schemes::number_of_schemes]{
  "tensor_train",
  "full",
};

struct parameters
{
  double advection = 1.0;
  double diffusion = 0.0;
  double source = -0.0;
  size_t number_points_1d = 20;
  double deltat = 0.1;
  size_t solver_steps = 20;
  int boundary_conditions = 0; // 0 = periodic (default), 1 = Dirichlet
  bool use_bandwidth = false;
};

namespace boundary_conditions
{
constexpr int periodic = 0;
constexpr int dirichlet = 1;
} // namespace boundary_conditions

inline bool use_dirichlet_boundary_conditions(parameters const& input)
{
  boba_always_assert(
    input.boundary_conditions == boundary_conditions::periodic or input.boundary_conditions == boundary_conditions::dirichlet,
    "Unsupported boundary_conditions. Use 0 for periodic or 1 for Dirichlet.");
  return input.boundary_conditions == boundary_conditions::dirichlet;
}

size_t linear_outer_iterations = 1;
size_t linear_inner_iterations = 20;
double linear_threshhold = 1.0e-4;
double svd_tolerance_relative = 1.0e-4;
double svd_tolerance_absolute = 1.0e-12;

enum solvers : size_t
{
  gmres,
  cg,
  bicgstab,
  grad_descent,
  ttm_split_inverse,
  number_of_solvers,
};

solvers solver_scheme = solvers::gmres;

#ifdef BOBA_ENABLE_EIGEN
inline Eigen::VectorXd conventional_linear_solve(
  Eigen::SparseMatrix<double>& A,
  Eigen::VectorXd b)
{
  if constexpr (not boba::is_gpu(space))
  {
    Eigen::VectorXd x(b.size());
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    x = solver.solve(b);
    return x;
  }
  else
  {
    boba::detail::ignore(A);
    boba_error("Not implemented for device code");
    return b;
  }
}

template <size_t dimension>
Eigen::SparseMatrix<double> form_conventional_LHS(
  ::boba::Array<size_t, dimension> sizes,
  double dx,
  parameters input)
{
  size_t N_full = ::boba::product(sizes);
  checkpoint();
  Eigen::SparseMatrix<double> full_sparse(static_cast<Eigen::Index>(N_full), static_cast<Eigen::Index>(N_full));
  checkpoint();
  using tensor_type = ::boba::Tensor<dimension, space, double>;
  using view_type = typename tensor_type::view_type;

  for (size_t row = 0; row < N_full; row++)
  {
    auto indices = view_type::multiindex(sizes, row);
    double diagonal_coeff = input.advection / dx * double(dimension) + input.source;
    full_sparse.insert(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(row)) = 1 + input.deltat * diagonal_coeff;
    for (size_t d = 0; d < dimension; d++)
    {
      auto index = indices;
      {
        index[d] = ::boba::mod((indices[d] + sizes[d]) - 1, sizes[d]);
        size_t col = view_type::index(view_type::precompute_strides(sizes), index);
        full_sparse.insert(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = -input.deltat * input.advection / dx;
      }
    }
  }
  checkpoint();
  full_sparse.makeCompressed();
  checkpoint();
  return full_sparse;
}
#endif

template <size_t dimension>
struct run
{

  // maps initial_guess to solution = LHS^{-1}SRHS

  void linear_solve(
    boba::TensorTrain<dimension, space, double>& initial_guess,
    boba::TensorTrain<dimension, space, double>& RHS,
    boba::TensorTrainMatrix<dimension, space, double>& LHS,
    boba::TensorTrain<dimension, space, double>& solution)
  {
    BOBA_CALI_EXTERNAL_MARK

    // linear solver
    checkpoint();
    using operator_t = boba::TensorTrainMatrix<dimension, space, double>;
    using vector_t = boba::TensorTrain<dimension, space, double>;
    boba::Krylov<operator_t, vector_t> solver(
      linear_outer_iterations,
      linear_inner_iterations,
      linear_threshhold);
    checkpoint();
    solver.method = solver_scheme;

    solver.set_matrix(LHS);
    solver.solve(RHS, initial_guess, solution);

    solution.round();
  }

  // maps previous solution to new solution

  void nonlinear_solve(
    boba::TensorTrain<dimension, space, double>& previous_solution,
    boba::TensorTrainMatrix<dimension, space, double>& LHS,
    boba::TensorTrain<dimension, space, double>& solution)
  {
    BOBA_CALI_EXTERNAL_MARK
    size_t nonlinear_maximum_iterations = 1;
    size_t nonlinear_iterations_used = 0;
    double nonlinear_solver_tolerance = 1.0e-12;
    bool nonlinear_iterating = true;
    bool verbose = true;
    while (nonlinear_iterating)
    {
      ::boba::TicToc<::boba::tictoc_units::milliseconds> iterate;
      //
      BOBA_CALI_EXTERNAL_BEGIN("adv_assembly");
      //

      // this is a no-op for linear operations
      // for nonlinear problems we would linearize here

      //
      BOBA_CALI_EXTERNAL_SWITCH("adv_assembly", "adv_run_solver");
      //

      {
        /*
        boba::TensorTrain<dimension, space, double> residual = LHS * previous_solution;
        residual.rename("residual");
        residual.fill_with_zeros();
        double convergence = ::boba::norm_difference_inf(residual, previous_solution);
        */
      }

      // guess = previous_solution
      boba::TensorTrain<dimension, space, double> guess(previous_solution);
      guess.rename("guess");

      linear_solve(
        guess,
        previous_solution,
        LHS,
        solution);

      //
      BOBA_CALI_EXTERNAL_SWITCH("adv_run_solver", "adv_compute_nonlinear_residual");
      //

      boba::TensorTrain<dimension, space, double> residual = LHS * solution;
      residual.rename("residual");
      residual.fill_with_zeros();

      double convergence = ::boba::norm_difference_frobenius(residual, previous_solution);

      //
      BOBA_CALI_EXTERNAL_SWITCH("adv_compute_nonlinear_residual", "adv_nonlinear_bookkeeping");
      //

      size_t iteration_time = iterate.timing();

      nonlinear_iterations_used++;

      size_t iteration_timeout = iterate.convert<::boba::tictoc_units::seconds>(60);

      if (iteration_time > iteration_timeout)
      {
        if (verbose)
        {
          std::cout << ::boba::write_indent(1) << " iteration timeout failure, "
                    << iteration_time << " > " << iteration_timeout
                    << " " << iterate.units_string << std::endl;
        }
        nonlinear_iterating = false;
      }
      if (convergence < nonlinear_solver_tolerance)
      {
        if (verbose)
        {
          std::cout << ::boba::write_indent(1) << "% convergence achieved = " << convergence << std::endl;
        }
        nonlinear_iterating = false;
      }
      if (nonlinear_iterations_used >= nonlinear_maximum_iterations)
      {
        if (verbose)
        {
          std::cout << ::boba::write_indent(1) << "% maximum iterations failure " << std::endl;
        }
        nonlinear_iterating = false;
      }
      if (verbose)
      {
        std::cout << ::boba::write_indent(1)
                  << "% " << nonlinear_iterations_used
                  << "  ||R|| " << convergence
                  << "  " << iterate.units_string << " " << iteration_time
                  << "  CR " << solution.get_full_size() / solution.get_number_elements() << "x  "
                  << solution.ranks_string()
                  << std::endl;
      }

      //
      BOBA_CALI_EXTERNAL_END("adv_nonlinear_bookkeeping");
      //

    } // end nonlinear iteration
  }

  run(
    double& error,
    parameters input)
  {
    run_test(
      error,
      input);
  }

  void run_test(
    double& error,
    parameters input)
  {
    BOBA_CALI_EXTERNAL_BEGIN("adv_setup")
    std::cout << "%---------------------------------------------------------" << std::endl;
    std::cout << "% Running test " << std::endl;
    std::cout << "% number_elements_1d = " << input.number_points_1d << std::endl;
    std::cout << "% dimension = " << dimension << std::endl;
    std::cout << "%---------------------------------------------------------" << std::endl;

    size_t N = input.number_points_1d;
    auto sizes = ::boba::filled_array<dimension>(N);

    if (boba::env_match("SOLVER", "gmres"))
    {
      solver_scheme = solvers::gmres;
    }
    if (boba::env_match("SOLVER", "gradient_descent"))
    {
      solver_scheme = solvers::grad_descent;
    }
    if (boba::env_match("SOLVER", "cg"))
    {
      solver_scheme = solvers::cg;
    }
    if (boba::env_match("SOLVER", "bicgstab"))
    {
      solver_scheme = solvers::bicgstab;
    }

    bool verbose = boba::is_env_nonempty("VERBOSE");
    bool file_dump = boba::is_env_nonempty("FILE_DUMP");

    // -------------------------------------------------
    // Define derivative and identity operators
    // -------------------------------------------------
    // see common_ttm.hpp
    common_ttm<dimension, space, double> operators(N, 2.0, false);
    double dx = operators.dx;
    double wavespeed = input.advection;

    //
    // advection example
    //
    double kappa = input.source;
    // TODO<feature> add diffusion
    // double beta = input.diffusion;
    double deltat = input.deltat;
    double time = 0.0;
    size_t time_steps = input.solver_steps;
    double end_time = deltat * double(time_steps);

    checkpoint();
    boba::TensorTrainMatrix<dimension, space, double> LHS(sizes, sizes);
    LHS.rename("LHS");
    LHS.fill_with_zeros();
    const bool use_dirichlet_bc = use_dirichlet_boundary_conditions(input);
    auto const& source_domain = use_dirichlet_bc ? operators.identity_interior : operators.identity;
    auto const& advection_operator = use_dirichlet_bc ? operators.advection_forward_interior : operators.advection_forward_periodic;
    {
      // Add time discretization and source term
      LHS += operators.identity;
      // Do not apply source on Dirichlet boundary rows.
      LHS += -1. * deltat * kappa * source_domain;
      // Add spatial operator
      LHS += (wavespeed * deltat / dx) * advection_operator;
    }
    if (input.use_bandwidth)
    {
      LHS.determine_bandwidth();
    }

    checkpoint();
    boba::TensorTrainMatrix<dimension, space, double> LHST(sizes, sizes);
    LHST.rename("LHS");

    LHST.fill_with_zeros();
    LHST = LHS.transpose();

    //
    // Initial Value
    //
    double blob_radius = 0.63;
    auto initial_function = [=] __boba_host_device__(double x)
    {
      double x_reference = ::boba::periodic(x, -1.0, 1.0);
      double x2 = ::boba::pow(x_reference, 2.);
      double r2 = ::boba::pow(blob_radius, 2.);
      double f = 0.0;
      if (x2 < r2)
      {
        f = boba::exp(1. / r2 + 1. / (x2 - r2));
      }
      return f;
    };

    boba::TensorTrain<dimension, space, double> initial_value(sizes);
    initial_value.rename("initial_value");
    initial_value.fill_with_zeros();
    {
      for (size_t d = 0; d < dimension; d++)
      {
        auto initial_value_view = initial_value.cores[d].view();
        ::boba::loop<space, 1>({N},
                               [=] __boba_host_device__(size_t i)
        {
          double x = -1.0 + dx * double(i);
          double value = initial_function(x);
          initial_value_view({0, i, 0}) = value;
        });
      }
    }

    if (file_dump)
    {
      boba::write_to_file(initial_value.decompress(), ::boba::name_flag("dump_initial_value_decompressed"));
    }

    boba::TensorTrain<dimension, space, double> solution(sizes);
    solution.rename("solution");
    solution = initial_value;
    solution.svd_tolerance_absolute = svd_tolerance_absolute;
    solution.svd_tolerance_relative = svd_tolerance_relative;

    boba::TensorTrain<dimension, space, double> previous_solution(sizes);
    previous_solution.rename("previous_solution");
    previous_solution = solution;

    std::cout << "% Initial conditions " << std::endl;
    std::cout << "%   Boba size              : " << solution.get_number_elements() << std::endl;
    std::cout << "%   Compression rate       : " << solution.compression_rate() << "x" << std::endl;
    std::cout << "%   Ranks                  : " << solution.ranks_string() << std::endl;
    std::cout << "%   Time steps: " << time_steps << std::endl
              << "% --------------------- " << std::endl;

    BOBA_CALI_EXTERNAL_SWITCH("adv_setup", "adv_tt_timestepping");

    //
    // Time-stepping loop
    //
    boba::TicToc<tictoc_units> timer_timestepping;

    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::tt]))
    {
      size_t cyclecount = 0;

      if (verbose)
      {
        std::cout << "% Current time: " << time << std::endl
                  << "% --------------------- " << std::endl;
        std::cout << "% Current index: " << cyclecount << std::endl
                  << "% --------------------- " << std::endl;
      }

      for (size_t step = 0; step < time_steps; step++)
      {
        time = time + deltat;
        cyclecount++;

        if (solver_scheme == gmres)
        {
          nonlinear_solve(
            previous_solution,
            LHS,
            solution);
        }
        else if (solver_scheme == cg)
        {
          nonlinear_solve(
            previous_solution,
            LHS,
            solution);
        }
        else if (solver_scheme == bicgstab)
        {
          nonlinear_solve(
            previous_solution,
            LHS,
            solution);
        }
        else
        {
          double tol = 1.0e-9;
          double step_error = 100;
          double lambda_descent_scaling_parameter = 1.1;
          size_t counter = 0;
          size_t max_count = 25;
          boba::TensorTrain<dimension, space, double> residual(solution.sizes());
          residual.rename("residual");
          residual.fill_with_zeros();
          boba::TensorTrain<dimension, space, double> intermediate(solution.sizes());
          intermediate.rename("intermediate");
          intermediate.fill_with_zeros();
          boba::TensorTrain<dimension, space, double> RHS = LHST * previous_solution;
          RHS.rename("RHS");
          while (step_error > tol && counter < max_count)
          {
            // LHS*solution - previous_solution
            auto intermediate_temp = LHS * solution;
            intermediate_temp.round();
            auto residual_temp = LHST * intermediate_temp;
            residual_temp -= RHS;
            step_error = ::boba::norm_frobenius(residual);
            if (verbose)
            {
              std::cout << "% Current error in gradient descent: " << step_error << std::endl
                        << "% --------------------- " << std::endl;
            }
            residual *= -1.0 * lambda_descent_scaling_parameter;
            solution += residual;
            solution.round();
            counter++;
          }
        }

        // Initialize next iteration
        previous_solution = solution;

        if (verbose)
        {
          std::cout << "% Current time: " << time << std::endl
                    << "% --------------------- " << std::endl;
          std::cout << "% Current index: " << cyclecount << std::endl
                    << "% --------------------- " << std::endl;
        }
        boba::detail::device_sync();

      } // end timestepping loop

      timer_timestepping.end();
      std::cout << "% Timestepping complete " << std::endl;

      std::cout << "% Results " << std::endl;
      std::cout << "%   time                   : " << timer_timestepping.timing() << std::endl;
      std::cout << "%   Boba size              : " << solution.get_number_elements() << std::endl;
      std::cout << "%   Compression rate       : " << solution.compression_rate() << "x" << std::endl;
      std::cout << "%   Ranks                  : " << solution.ranks_string() << std::endl;
      std::cout << std::endl;
    }

    //
    // Conventional solver
    //

    ::boba::Tensor<dimension, space, double> conventional_solution(sizes);
    conventional_solution.rename("conventional_solution");

#ifdef BOBA_ENABLE_EIGEN
    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::full]))
    {
      auto conventional_LHS = form_conventional_LHS<dimension>(sizes, dx, input);

      Eigen::VectorXd eigen_solution, previous_eigen_solution;
      {
        ::boba::Vector<space, double> initial_value_vector({::boba::product(sizes)});
        auto initial_value_tensor = initial_value.decompress();
        initial_value_vector.reshape(initial_value_tensor);
        auto initial_value_vector_map = ::boba::get_eigen_map(initial_value_vector);
        eigen_solution = initial_value_vector_map;
        previous_eigen_solution = initial_value_vector_map;
      }
      //
      BOBA_CALI_EXTERNAL_SWITCH("adv_tt_timestepping", "adv_conventional_timestepping");
      //
      boba::TicToc<tictoc_units> timer_conventional;

      double simulation_time = 0;
      size_t cyclecount = 0;
      for (size_t step = 0; step < time_steps; step++)
      {
        simulation_time = simulation_time + deltat;
        cyclecount++;

        eigen_solution = conventional_linear_solve(conventional_LHS, previous_eigen_solution);

        // Initialize next iteration
        previous_eigen_solution = eigen_solution;

        if (verbose)
        {
          std::cout << "% Current time: " << simulation_time << std::endl
                    << "% --------------------- " << std::endl;
          std::cout << "% Current index: " << cyclecount << std::endl
                    << "% --------------------- " << std::endl;
        }
        boba::detail::device_sync();
      }

      //
      BOBA_CALI_EXTERNAL_END("adv_conventional_timestepping");
      //

      timer_conventional.end();

      std::cout << "% Conventional Results " << std::endl;
      std::cout << "%   time                   : " << timer_conventional.timing() << std::endl;
      std::cout << std::endl;

      {
        ::boba::Vector<space, double> solution_vector({::boba::product(sizes)});
        auto solution_vector_map = ::boba::get_eigen_map(solution_vector);
        solution_vector_map = eigen_solution;
        conventional_solution.reshape(solution_vector);
      }
      if (file_dump)
      {
        boba::write_to_file(conventional_solution, ::boba::name_flag("dump_conventional_solution"));
      }
    }
#else
    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::full]))
    {
      boba_error("eigen not enabled....");
    }
#endif

    //
    // Compute exact solution
    //
    if (::boba::abs(input.diffusion) < 1.0e-16)
    {
      //
      // Exact Solution
      //
      boba::TensorTrain<dimension, space, double> exact_solution(sizes);
      exact_solution.rename("exact_solution");
      exact_solution.fill_with_zeros();

      {
        for (size_t d = 0; d < dimension; d++)
        {
          double kappa_scaled = kappa / double(dimension);
          auto exact_solution_view = exact_solution.cores[d].view();
          ::boba::loop<space, 1>({N},
                                 [=] __boba_host_device__(size_t i)
          {
            double x = -1.0 + dx * double(i);
            double advection_value = initial_function(x - end_time * wavespeed);
            double source_value = ::boba::exp(kappa_scaled * end_time);
            exact_solution_view({0, i, 0}) = advection_value * source_value;
          });
        }
      }

      //
      // Error in the TT format
      //
      double norm_tt_solution = ::boba::norm_frobenius(solution);
      double norm_tt_exact = ::boba::norm_frobenius(exact_solution);
      double norm_tt_error = ::boba::norm_difference_inf(solution.decompress(), exact_solution.decompress());
      double norm_tt_error_relative = norm_tt_error / norm_tt_exact;

      std::cout << "%               ||tt_solution|| = " << norm_tt_solution << std::endl;
      std::cout << "%                  ||tt_exact|| = " << norm_tt_exact << std::endl;
      std::cout << "%    ||tt_solution - tt_exact|| = " << norm_tt_error << std::endl;
      std::cout << "% || ... - ... ||/ ||tt_exact|| = " << norm_tt_error_relative << std::endl;

      //
      // Error in the tensor format
      //
      auto exact_tensor = exact_solution.decompress();
      exact_tensor.rename("exact_tensor");
      auto solution_tensor = solution.decompress();
      solution_tensor.rename("solution_tensor");

      if (file_dump)
      {
        boba::write_to_file(exact_tensor, ::boba::name_flag("dump_exact_tensor"));
        boba::write_to_file(solution.decompress(), ::boba::name_flag("dump_solution_decompressed"));
      }

      double norm_tensor_solution = ::boba::norm_frobenius(solution_tensor);
      double norm_tensor_exact = ::boba::norm_frobenius(exact_tensor);
      double norm_tensor_conventional = ::boba::norm_frobenius(conventional_solution);

      double norm_tensor_error = ::boba::norm_difference_inf(exact_tensor, solution_tensor);
      double norm_tensor_error_relative = norm_tensor_error / norm_tensor_exact;

      double norm_conventional_error = ::boba::norm_difference_inf(exact_tensor, conventional_solution);
      double norm_conventional_error_relative = norm_conventional_error / norm_tensor_exact;

      double norm_conventional_difference = ::boba::norm_difference_inf(solution_tensor, conventional_solution);
      double norm_conventional_difference_relative = norm_conventional_difference / (norm_tensor_solution + norm_tensor_conventional);

      std::cout << std::endl;
      std::cout << "%               ||solution|| = " << norm_tensor_solution << std::endl;
      std::cout << "%                  ||exact|| = " << norm_tensor_exact << std::endl;
      std::cout << "%           ||conventional|| = " << norm_tensor_conventional << std::endl;
      std::cout << std::endl;
      std::cout << "%       ||solution - exact|| = " << norm_tensor_error << std::endl;
      std::cout << "% || ... - ... ||/ ||exact|| = " << norm_tensor_error_relative << std::endl;
      std::cout << std::endl;
      std::cout << "%   ||conventional - exact|| = " << norm_conventional_error << std::endl;
      std::cout << "% || ... - ... ||/ ||exact|| = " << norm_conventional_error_relative << std::endl;
      std::cout << std::endl;
      std::cout << "%||conventional - solution|| = " << norm_conventional_difference << std::endl;
      std::cout << "%   || ... - ... ||/ ||avg|| = " << norm_conventional_difference_relative << std::endl;

      // For CI we just check consistency between the discretizations
      error = norm_conventional_difference_relative;
    }
  }
};

template <size_t dimension>
struct run_wrapper
{

  // runs the problem several times and averages the results

  run_wrapper(
    bool& check,
    size_t averages,
    parameters input)
  {
    double error = -1.0;
    for (size_t r = 0; r < averages; r++)
    {
      run<dimension> runner(
        error,
        input);
      pass_or_fail(check, error, 1.0e-0);
    }
  }
};

int main(int argc, char* argv[])
{

  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  size_t averages = 1;
  size_t dimension = 2;
  parameters input;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(dimension,
                             "-D",
                             "--dimension",
                             "Number of dimensions (2 or 3).");

  args.add_optional_argument(input.boundary_conditions,
                             "-bc",
                             "--boundary-conditions",
                             "BCs: 0 = periodic (default), 1 = Dirichlet");

  args.add_optional_argument(input.advection,
                             "-c",
                             "--advection",
                             "Coefficient for advection term.");

  args.add_optional_argument(input.diffusion,
                             "-d",
                             "--diffusion",
                             "Coefficient for diffusion term.");

  args.add_optional_argument(input.number_points_1d,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(input.source,
                             "-s",
                             "--source",
                             "Coefficient for source term.");

  args.add_optional_argument(input.deltat,
                             "-dt",
                             "--timestep",
                             "Time step size.");

  args.add_optional_argument(input.solver_steps,
                             "-nt",
                             "--timesteps",
                             "Number of solver time steps to take.");

  args.add_optional_argument(linear_inner_iterations,
                             "-GI",
                             "--linear_inner",
                             "Inner iterations for linear solver.");

  args.add_optional_argument(linear_outer_iterations,
                             "-GO",
                             "--linear_outer",
                             "Outer iterations for linear solver.");

  args.add_optional_argument(linear_threshhold,
                             "-GT",
                             "--linear_threshhold",
                             "Error tolerance for linear solver.");

  args.add_optional_argument(svd_tolerance_relative,
                             "-svd",
                             "--svd_tolerance_relative",
                             "Relative tolerance for truncating small singular values.");

  args.add_optional_argument(svd_tolerance_absolute,
                             "-svda",
                             "--svd_tolerance_absolute",
                             "Absolute tolerance floor for truncating small singular values.");

  args.add_optional_argument(input.use_bandwidth,
                             "-bt",
                             "--bandwidth",
                             "Enable tensor-train-matrix bandwidth acceleration.");

  args.parse_check();

  if (dimension == 2)
  {
    run_wrapper<2> runner(check, averages, input);
  }
  if (dimension == 3)
  {
    run_wrapper<3> runner(check, averages, input);
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
