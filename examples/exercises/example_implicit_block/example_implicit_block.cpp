// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../tests/common.hpp"
#include "../../tests/common_ttm.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*

  The goal of this example is to solve a linearized coupled system implicitly,
  which involves a system of equations as opposed to a single scalar equation.
  Given select initial conditions, we can form an analytic solution, so we can perform formal error analysis.

  d/dt h + H * d/dx h ​+ H * d/dy h = 0
  d/dt u ​+ g * d/dx h ​+      0     = 0
  d/dt v ​+      0     +​ g * d/dy h = 0

  characteristic speed c = sqrt(g*H)
  Let H = 1,
  so then g = c^2

  Rewriting,

  d/dt h     +  d/dx u ​+  d/dy v   = 0
  g * d/dx h ​+  d/dt u +     0     = 0
​  g * d/dy h +    0    +  d/dt v ​+ = 0

  Discretizing using backward Euler in time:

                h_{i + 1} + dt * d/dx u_{i + 1} + dt * d/dy v_{i + 1} = h_i
  g * dt * d/dx h_{i + 1} +           u_{i + 1} +                   0 = u_i
  g * dt * d/dy h_{i + 1}                     0 +           v_{i + 1} = v_i

  or equivalently as a block system:

  [           I    dt * Dx    dt * Dy ] [ h_{i + 1} ]   [ h_i ]
  [ g * dt * Dx          I          0 ] [ u_{i + 1} ] = [ u_i ]
  [ g * dt * Dy          0          I ] [ v_{i + 1} ]   [ v_i ]

*/

enum schemes : size_t
{
  tt = 0,
  number_of_schemes,
};

static std::string const schemes_strings[schemes::number_of_schemes]{
  "tensor_train",
};

struct parameters
{
  size_t number_points_1d = 20;
  double wavespeed = 0.3;
  double deltat = 0.01;
  size_t solver_steps = 50;
  int boundary_conditions = 1; // 0 = periodic, 1 = Dirichlet (default)
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
double svd_tolerance = 1.0e-4;

enum solvers : size_t
{
  gmres,
  number_of_solvers,
};

solvers solver_scheme = solvers::gmres;

template <size_t dimension>
struct run
{

  // Uses GMRES to solve linear system

  template <typename operator_t, typename vector_t>
  void linear_solve_gmres(
    vector_t& initial_guess,
    vector_t& RHS,
    operator_t& LHS,
    vector_t& solution)
  {
    BOBA_CALI_EXTERNAL_MARK

    // linear solver
    checkpoint();
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

  run(
    bool& check,
    parameters input)
  {
    run_test(
      check,
      input);
  }

  void run_test(
    bool& check,
    parameters input)
  {
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

    // -------------------------------------------------
    // Define derivative and identity operators
    // -------------------------------------------------
    // see common_ttm.hpp
    common_ttm<dimension, space, double> operators(N, 2.0, false);
    double dx = operators.dx;
    double wavespeed = input.wavespeed;
    auto param_g = boba::pow(wavespeed, 2.0);
    double deltat = input.deltat;
    size_t time_steps = input.solver_steps;
    double end_time = deltat * double(time_steps);

    // Don't allow the waves to propagate to the boundaries, since we are not dealing with
    // boundary conditions in this example
    // So we enforce that a slightly faster wave traveling at 0.1 + wavespeed does not x = 0.8
    double maximum_end_time = 0.8 / (0.1 + wavespeed);
    boba_always_assert_lt(end_time, maximum_end_time, "Proposed experiment runtime is too long, problems will happen at boundary conditions. Please take fewer time steps.");

    checkpoint();
    using tensor_t = boba::Tensor<dimension, space, double>;
    using tt_t = boba::TensorTrain<dimension, space, double>;
    using ttm_t = boba::TensorTrainMatrix<dimension, space, double>;
    constexpr size_t system_size = 1 + dimension;

    ::boba::BlockOperator<ttm_t> block_LHS(system_size, system_size);

    for (size_t r = 0; r < system_size; r++)
      for (size_t c = 0; c < system_size; c++)
      {
        block_LHS({r, c}).resize(sizes, sizes);
        block_LHS({r, c}).fill_with_zeros();
      }

    //
    // First row, h equation
    //

    const bool use_dirichlet_bc = use_dirichlet_boundary_conditions(input);
    auto const& height_gradients = use_dirichlet_bc
                                     ? operators.gradient_backward_interior
                                     : operators.gradient_backward_periodic;
    auto const& momentum_gradients = use_dirichlet_bc
                                       ? operators.gradient_forward_interior
                                       : operators.gradient_forward_periodic;

    block_LHS({0, 0}) = operators.identity;
    for (size_t c = 1; c < system_size; c++)
    {
      block_LHS({0, c}).resize(sizes, sizes);
      block_LHS({0, c}).fill_with_zeros();
      block_LHS({0, c}) = deltat / dx * height_gradients[c - 1];
    }

    // u, v, ... eqns
    for (size_t r = 1; r < system_size; r++)
    {
      block_LHS({r, 0}) = deltat / dx * param_g * momentum_gradients[r - 1];
      block_LHS({r, r}) = operators.identity;
    }

    //
    // Initial Value: 1 if inside a ball of radius hat_radius, 0 otherwise
    //
    double hat_radius = 0.2;

    tensor_t initial_value_tensor(sizes);
    initial_value_tensor.rename("initial_value_tensor");
    initial_value_tensor.fill_with_zeros();
    auto initial_value_view = initial_value_tensor.view();

    ::boba::loop<space, 1>(initial_value_view.size(),
                           [=] __boba_host_device__(size_t id)
    {
      auto mid = initial_value_view.multiindex(id);
      double radius_squared = 0.0;
      for (size_t d = 0; d < dimension; d++)
      {
        double xi = -1.0 + dx * static_cast<double>(mid[d]);
        radius_squared += xi * xi;
      }
      if (radius_squared < hat_radius * hat_radius)
      {
        initial_value_view(id) = 1.0;
      }
    });

    //
    // Initial condition is initial_function for h, 0 for u, v
    //
    ::boba::BlockVector<tensor_t> block_solution_full(system_size);
    block_solution_full(0) = initial_value_tensor;
    for (size_t r = 1; r < system_size; r++)
    {
      block_solution_full(r).resize(sizes);
      block_solution_full(r).fill_with_zeros();
    }

    ::boba::BlockVector<tt_t> block_solution(system_size);
    block_solution(0) = boba::compress_to_TensorTrain(initial_value_tensor);
    for (size_t r = 1; r < system_size; r++)
    {
      block_solution(r).resize(sizes);
      block_solution(r).fill_with_zeros();
    }

    boba::savetxt(block_solution_full(0), "full_initial_value.dump");
    boba::savetxt(block_solution(0).decompress(), "full_initial_value.dump");

    auto previous_block_solution = block_solution;
    auto previous_block_solution_full = block_solution_full;

    std::cout << "% Initial conditions " << std::endl;
    std::cout << "%   Boba size              : " << block_solution.get_number_elements() << std::endl;
    std::cout << "%   Compression rate       : " << block_solution.compression_rate() << "x" << std::endl;
    std::cout << "%   Time steps: " << time_steps << std::endl
              << "% --------------------- " << std::endl;

    //
    // Time-stepping loop
    //
    boba::TicToc<tictoc_units> timer_timestepping;
    double time = 0.0;

    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::tt]))
    {
      {
        size_t cyclecount = 0;

        for (size_t step = 0; step < time_steps; step++)
        {
          time = time + deltat;
          cyclecount++;

          if (solver_scheme == gmres)
          {
            linear_solve_gmres(
              previous_block_solution,
              previous_block_solution,
              block_LHS,
              block_solution);
          }
          boba_print(cyclecount);
          boba_print(time);

          previous_block_solution = block_solution;
        }

        timer_timestepping.end();
        std::cout << "% Timestepping complete " << std::endl;

        std::cout << "% Results " << std::endl;
        std::cout << "%   Time                   : " << timer_timestepping.timing() << std::endl;
        std::cout << "%   Boba size              : " << block_solution.get_number_elements() << std::endl;
        std::cout << "%   Compression rate       : " << block_solution.compression_rate() << "x" << std::endl;
        std::cout << std::endl;
      }

      {
        size_t cyclecount = 0;
        time = 0.0;
        for (size_t step = 0; step < time_steps; step++)
        {
          time = time + deltat;
          cyclecount++;

          if (solver_scheme == gmres)
          {
            linear_solve_gmres(
              previous_block_solution_full,
              previous_block_solution_full,
              block_LHS,
              block_solution_full);
          }
          boba_print(cyclecount);
          boba_print(time);

          previous_block_solution_full = block_solution_full;
        }

        timer_timestepping.end();
        std::cout << "% Timestepping complete " << std::endl;

        std::cout << "% Results " << std::endl;
        std::cout << "%   Time                   : " << timer_timestepping.timing() << std::endl;
        std::cout << "%   Full size              : " << block_solution_full.get_number_elements() << std::endl;
        std::cout << std::endl;
      }
    }

    //
    // Compute exact solution
    //
    {
      //
      // Exact Solution
      //
      auto disk_exact = [=] __boba_host_device__(double x, double y)
      {
        double R = wavespeed * time;

        if ((R <= 0.0) and (x * x + y * y < hat_radius * hat_radius))
        {
          return 1.0;
        }

        double d = std::hypot(x, y);
        if (d + R <= hat_radius) // circle fully inside disk
        {
          return 1.0;
        }
        if (d >= hat_radius + R) // no intersection
        {
          return 0.0;
        }

        // partial overlap
        double num = d * d + R * R - hat_radius * hat_radius;
        double den = 2.0 * d * R;
        double cosang = num / den;
        // clamp against roundoff
        if (cosang < -1.0)
        {
          cosang = -1.0;
        }
        if (cosang > 1.0)
        {
          cosang = 1.0;
        }
        return boba::acos(cosang) / boba::pi;
      };

      tensor_t exact_solution(sizes);
      exact_solution.rename("exact_solution");
      exact_solution.fill_with_zeros();
      auto exact_solution_view = exact_solution.view();

      ::boba::loop<space, 2>(exact_solution.sizes(),
                             [=] __boba_host_device__(boba::Array<size_t, 2> ij)
      {
        auto [i, j] = ij;
        double x = -1.0 + dx * double(i);
        double y = -1.0 + dx * double(j);
        double value = disk_exact(x, y);
        exact_solution_view({i, j}) = value;
      });

      auto solution = block_solution(0);
      auto solution_full = block_solution_full(0);

      boba::savetxt(exact_solution, "exact_solution.dump");
      boba::savetxt(solution.decompress(), "tt_solution.dump");
      boba::savetxt(solution_full, "full_solution.dump");

      //
      // Error in the TT format
      //
      double norm_tt_solution = ::boba::norm_frobenius(solution);
      double norm_tt_exact = ::boba::norm_frobenius(exact_solution);
      double norm_tt_error = ::boba::norm_difference_inf(solution.decompress(), exact_solution);
      double norm_tt_error_relative = norm_tt_error / norm_tt_exact;

      std::cout << "%               ||tt_solution|| = " << norm_tt_solution << std::endl;
      std::cout << "%                  ||tt_exact|| = " << norm_tt_exact << std::endl;
      std::cout << "%    ||tt_solution - tt_exact|| = " << norm_tt_error << std::endl;
      std::cout << "% || ... - ... ||/ ||tt_exact|| = " << norm_tt_error_relative << std::endl;

      pass_or_fail(check, norm_tt_error_relative, 5.0 * dx);
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
                             "BCs: 0 = periodic, 1 = Dirichlet (default)");

  args.add_optional_argument(input.wavespeed,
                             "-c",
                             "--wavespeed",
                             "Coefficient for wavespeed term.");

  args.add_optional_argument(input.number_points_1d,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

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

  args.add_optional_argument(svd_tolerance,
                             "-svd",
                             "--svd_tolerance",
                             "Tolerance for determining truncation of small singular values.");

  args.parse_check();

  if (dimension == 2)
  {
    run<2> runner(check, input);
  }
  /*
  Note - to support 3D or higher exact solution would need to be generalized
  */

  checkpoint();
  boba::finalize();
  return final_check(check);
}
