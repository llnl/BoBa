// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../tests/common.hpp"
#include "../tests/common_ttm.hpp"

static constexpr boba::execution_space space = boba::default_execution_space;
static constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
  This example provides one possible implementation of a solver for a three-dimensional
  nonlinear evolution problem

  u_{,t} + f(u)_{,x} + f(u)_{,y} + f(u)_{,z} = \nu(u_{,xx} + u_{,yy} + u_{,zz}) + S(x,y,z),

  where f(u) = u^2,
  subject to periodic boundary conditions on [-\pi,\pi]^3. Here, \nu is a diffusion rate
  assumed to be small and S(x,y,z) is a source term.

  The purpose of this test is to evaluate strategies for upwinding the
  derivatives for the flux terms.
*/

template <typename T>
concept is_TT = std::is_same_v<T, boba::TensorTrain<3, space, double>>;

template <typename T>
concept is_tensor = std::is_same_v<T, boba::Tensor<3, space, double>>;

enum class Scheme : size_t
{
  Tensor = 0,
  TT_cross_maxvol,
  TT_cross_deim,
  number_of_schemes,
};

/**
 * \brief Parameter structure to hold essential problem data.
 */

struct Parameters
{
  double end_time = 0.01;
  double nu = 1.0e-01;
  size_t number_points_1d = 64;
  double domain_begin = -boba::pi;
  double domain_end = boba::pi;
  double CFL_scaling = 0.1;
};

/**
 * \brief Computes the time step for the computation.
 */
double compute_dt(double max_u,
                  double dx,
                  double nu,
                  double CFL,
                  double current_time,
                  double final_time)
{
  double dx2 = dx * dx;

  double adv_limit = (max_u > 0.0) ? dx / max_u / 3.0 : 1e300;
  double diff_limit = (nu > 0.0) ? dx2 / nu / 3.0 : 1e300;

  double dt = CFL * boba::min(adv_limit, diff_limit);

  double remaining_time = final_time - current_time;
  if (dt > remaining_time)
  {
    dt = remaining_time;
  }

  return dt;
}

/**
 * \brief Full-grid solver for the 3D viscous Burgers equation.
 *
 * Uses finite differences with local Lax-Friedrichs splitting
 * and explicit time integration.
 */
template <typename _Tensor_t>
struct BurgersSolver
{
  Parameters params;

  double domain_size;
  double dx;
  double odx;
  double odx2;

  using Vector_t = boba::Vector<space, double>;
  using Tensor_t = _Tensor_t;

  Vector_t x;
  Vector_t y;
  Vector_t z;

  std::string scheme_label = is_TT<Tensor_t> ? "TT" : "Tensor";

  Tensor_t flux_plus;
  Tensor_t flux_minus;

  // Finite difference operators fot TT
  common_ttm<3, space, double> operators;

  using DMRGCross_t = boba::DMRGCross<double>;
  DMRGCross_t compute_fluxes;

  /**
   * \brief Constructor for the full-grid solver which builds the periodic mesh.
   */
  BurgersSolver(const Parameters& params_)
      : params(params_),
        x({params_.number_points_1d - 1}),
        y({params_.number_points_1d - 1}),
        z({params_.number_points_1d - 1})
  {
    fill_grid();
    setup_flux_objects();
  }

  /**
   * \brief Fills the x y z grid
   */
  void fill_grid()
  {
    const size_t N = params.number_points_1d;

    boba_always_assert_positive(N, "Grid size must be positive");

    // Compute the uniform mesh spacing and scalings for differences
    domain_size = params.domain_end - params.domain_begin;
    dx = domain_size / (double(N) - 1.0);
    odx = 1.0 / dx;
    odx2 = 1.0 / (dx * dx);

    auto x_view = x.view();
    auto y_view = y.view();
    auto z_view = z.view();

    const double domain_begin = params.domain_begin;
    const double dx_local = dx;

    // Fill the grid and drop the right boundary
    boba::loop<space, 1>(N - 1,
                         [=] __boba_host_device__(size_t i)
    {
      double xi = domain_begin + dx_local * double(i);
      x_view(i) = xi;
      y_view(i) = xi;
      z_view(i) = xi;
    });
  }

  /**
   * \brief Initializes flux_plus, flux_minus
   */

  void setup_flux_objects()
    requires is_tensor<Tensor_t>
  {
    flux_plus.resize({x.size(), y.size(), z.size()});
    flux_minus.resize({x.size(), y.size(), z.size()});
  }

  void setup_flux_objects()
    requires is_TT<Tensor_t>
  {
    flux_plus.resize({x.size(), y.size(), z.size()});
    flux_minus.resize({x.size(), y.size(), z.size()});

    flux_plus.fill_with_zeros();
    flux_minus.fill_with_zeros();

    flux_plus.svd_tolerance_relative = 1.0e-4;
    flux_plus.svd_tolerance_absolute = 1.0e-9;

    flux_minus.svd_tolerance_relative = 1.0e-4;
    flux_minus.svd_tolerance_absolute = 1.0e-9;

    compute_fluxes.tolerance = 1.0e-3;

    operators.generate(x.size(), domain_size, false);
  }

  /**
   * \brief Returns the initial condition as a tensor.
   */
  Tensor_t initialize_solution()
    requires is_tensor<Tensor_t>
  {
    return compute_exact_solution(0.0);
  }

  Tensor_t initialize_solution()
    requires is_TT<Tensor_t>
  {
    auto tensor_solution = compute_exact_solution(0.0);
    return boba::compress_to_TensorTrain(tensor_solution, 1.0e-4);
  }

  /**
   * \brief Estimate ||solution||_inf
   */

  double estimate_max_u(const Tensor_t& solution)
    requires is_tensor<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_MARK
    return ::boba::norm_inf(solution);
  }

  double estimate_max_u(const Tensor_t& solution)
    requires is_TT<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_MARK
    /* TODO - use this for now, think of something better later */
    return ::boba::norm_inf(solution.decompress());
  }

  /**
   * \brief Returns the manufactured solution at time t.
   */

  boba::Tensor<3, space, double> compute_exact_solution(double t)
  {
    const size_t nx = x.size();
    const size_t ny = y.size();
    const size_t nz = z.size();

    boba::Tensor<3, space, double> exact({nx, ny, nz});
    auto exact_view = exact.view();

    auto x_view = x.const_view();
    auto y_view = y.const_view();
    auto z_view = z.const_view();

    const double nu = params.nu;

    boba::loop<space, 3>({nx, ny, nz},
                         [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [i, j, k] = mid;
      const double angle = x_view(i) + y_view(j) + z_view(k);
      exact_view(mid) = boba::exp(-nu * t) * boba::sin(angle);
    });

    return exact;
  }

  /**
   * \brief Pointwise comparison of solution against the exact solution at t_now
   */

  boba::Array<double, 2> compute_errors(Tensor_t& solution, double t_now)
    requires is_tensor<Tensor_t>
  {
    auto exact = compute_exact_solution(t_now);
    double absolute_error = ::boba::norm_difference_frobenius(solution, exact);
    double relative_error = absolute_error / ::boba::norm_frobenius(exact);
    double volume_factor = boba::sqrt(dx * dx * dx);
    absolute_error *= volume_factor;

    return boba::Array<double, 2>{absolute_error, relative_error};
  }

  boba::Array<double, 2> compute_errors(Tensor_t& solution, double t_now)
    requires is_TT<Tensor_t>
  {
    auto exact = compute_exact_solution(t_now);
    double absolute_error = ::boba::norm_difference_frobenius(solution.decompress(), exact);
    double relative_error = absolute_error / ::boba::norm_frobenius(exact);
    double volume_factor = boba::sqrt(dx * dx * dx);
    absolute_error *= volume_factor;

    return boba::Array<double, 2>{absolute_error, relative_error};
  }

  /**
   * \brief Returns the source function at time t.
   */

  Tensor_t compute_source(const Tensor_t& solution, double t)
    requires is_tensor<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_MARK
    auto sizes = solution.sizes();
    auto solution_view = solution.const_view();

    Tensor_t source(sizes);
    auto source_view = source.view();

    auto x_view = x.const_view();
    auto y_view = y.const_view();
    auto z_view = z.const_view();

    const double nu = params.nu;

    boba::loop<space, 3>(sizes,
                         [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [i, j, k] = mid;

      const double angle = x_view(i) + y_view(j) + z_view(k);
      source_view(mid) = 2.0 * nu * solution_view(mid);
      source_view(mid) += 3.0 * solution_view(mid) * boba::exp(-nu * t) * boba::cos(angle);
    });

    return source;
  }

  /**
   * \brief Returns the source function at time t.
   */

  Tensor_t compute_source(const Tensor_t& solution, double t)
    requires is_TT<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_MARK
    auto sizes = solution.sizes();

    Tensor_t source(sizes);

    auto x_view = x.const_view();
    auto y_view = y.const_view();
    auto z_view = z.const_view();

    const double nu = params.nu;

    {
      Tensor_t source_guess(sizes);
      source_guess.fill_with_zeros();

      auto cross_func = [=](boba::Array<size_t, 3> ijk)
      {
        auto [i, j, k] = ijk;
        const double angle = x_view(i) + y_view(j) + z_view(k);

        // Factor of u handled below
        return 2.0 * nu + 3.0 * boba::exp(-nu * t) * boba::cos(angle);
      };

      source = compute_fluxes.apply(source_guess, cross_func);
    }

    return boba::elementwise_product(source, solution);
  }

  /**
   * \brief Evaluates the rhs of the viscous Burgers' equation at time t.
   */

  Tensor_t compute_rhs(const Tensor_t& solution, double t)
    requires is_tensor<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_MARK

    auto sizes = solution.sizes();
    auto u_view = solution.const_view();

    Tensor_t source = compute_source(solution, t);
    auto source_view = source.const_view();

    auto flux_plus_view = flux_plus.view();
    auto flux_minus_view = flux_minus.view();

    const double nu = params.nu;
    const double odx_local = odx;
    const double odx2_local = odx2;

    // Rusanov flux
    // Generate +/- fluxes as temporaries
    boba::loop<space, 3>(sizes,
                         [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      double uijk = u_view(mid);
      double f = 0.5 * uijk * uijk;
      double a = boba::abs(uijk);

      flux_plus_view(mid) = 0.5 * (f + a * uijk);
      flux_minus_view(mid) = 0.5 * (f - a * uijk);
    });

    Tensor_t rhs(sizes);
    auto rhs_view = rhs.view();

    boba::loop<space, 3>(sizes,
                         [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [i, j, k] = mid;

      size_t im1 = (i == 0) ? sizes[0] - 1 : i - 1;
      size_t ip1 = (i == sizes[0] - 1) ? 0 : i + 1;

      size_t jm1 = (j == 0) ? sizes[1] - 1 : j - 1;
      size_t jp1 = (j == sizes[1] - 1) ? 0 : j + 1;

      size_t km1 = (k == 0) ? sizes[2] - 1 : k - 1;
      size_t kp1 = (k == sizes[2] - 1) ? 0 : k + 1;

      double convection = 0.0;

      convection += (flux_plus_view(mid) - flux_plus_view({im1, j, k})) * odx_local;
      convection += (flux_minus_view({ip1, j, k}) - flux_minus_view(mid)) * odx_local;

      convection += (flux_plus_view(mid) - flux_plus_view({i, jm1, k})) * odx_local;
      convection += (flux_minus_view({i, jp1, k}) - flux_minus_view(mid)) * odx_local;

      convection += (flux_plus_view({mid}) - flux_plus_view({i, j, km1})) * odx_local;
      convection += (flux_minus_view({i, j, kp1}) - flux_minus_view(mid)) * odx_local;

      double laplacian =
        (u_view({im1, j, k}) - 2.0 * u_view(mid) + u_view({ip1, j, k}) + u_view({i, jm1, k}) - 2.0 * u_view(mid) + u_view({i, jp1, k}) + u_view({i, j, km1}) - 2.0 * u_view(mid) + u_view({i, j, kp1})) * odx2_local;

      rhs_view(mid) = -convection + nu * laplacian + source_view(mid);
    });

    return rhs;
  }

  /**
   * \brief Evaluates the rhs of the viscous Burgers' equation at time t.
   */

  Tensor_t compute_rhs(const Tensor_t& solution, double t)
    requires is_TT<Tensor_t>
  {
    BOBA_CALI_EXTERNAL_BEGIN("compute_rhs_setup");

    auto sizes = solution.sizes();

    Tensor_t source = compute_source(solution, t);

    Tensor_t rhs(sizes);

    const double nu = params.nu;
    const double odx_local = odx;
    const double odx2_local = odx2;

    BOBA_CALI_EXTERNAL_SWITCH("compute_rhs_setup", "compute_rhs_fluxes");
    {
      boba::DynamicTensorTrainView<3, double> u_view(solution);

      auto cross_func = [=](boba::Array<size_t, 3> ijk)
      {
        auto u = u_view(ijk);
        double f = 0.5 * u * u;
        double a = boba::abs(u);
        return 0.5 * (f + a * u);
      };

      flux_plus = compute_fluxes.apply(flux_plus, cross_func);
    }
    {
      boba::DynamicTensorTrainView<3, double> u_view(solution);

      auto cross_func = [=](boba::Array<size_t, 3> ijk)
      {
        auto u = u_view(ijk);
        double f = 0.5 * u * u;
        double a = boba::abs(u);
        return 0.5 * (f - a * u);
      };

      flux_minus = compute_fluxes.apply(flux_minus, cross_func);
    }

    BOBA_CALI_EXTERNAL_SWITCH("compute_rhs_fluxes", "compute_rhs_derivatives");

    auto N = sizes[0];
    boba_always_assert_equal(N, sizes[1], "Hard coded to be isotropic for now");
    boba_always_assert_equal(N, sizes[2], "Hard coded to be isotropic for now");

    constexpr size_t dimension = 3;

    auto convection = (odx_local) * (operators.advection_forward_periodic * flux_plus);

    convection += (odx_local) * (operators.advection_backward_periodic * flux_minus);

    auto laplacian = (odx2_local) * (operators.laplacian_periodic * solution);

    rhs = -convection + nu * laplacian + source;

    BOBA_CALI_EXTERNAL_END("compute_rhs_derivatives");
    return rhs;
  }

  /**
   * \brief Runs an explicit time integration test with explicit Euler.
   */
  void run_test(bool& check)
  {
    std::cout << "Beginning test with scheme: " << scheme_label << std::endl;

    BOBA_CALI_EXTERNAL_BEGIN("Setup");

    double t_now = 0.0;
    size_t num_steps = 0;

    Tensor_t solution = initialize_solution();

    BOBA_CALI_EXTERNAL_SWITCH("Setup", "Timestepping");

    boba::TicToc<tictoc_units> test_timer;
    test_timer.tic();

    while (t_now < params.end_time)
    {
      boba_print(t_now);
      double max_u = estimate_max_u(solution);

      double dt = compute_dt(max_u,
                             dx,
                             params.nu,
                             params.CFL_scaling,
                             t_now,
                             params.end_time);

      if (dt <= 0.0)
      {
        boba_warn("Computed non-positive time step. Stopping integration.");
        break;
      }

      Tensor_t rhs = compute_rhs(solution, t_now);
      solution += (dt)*rhs;
      solution.round();

      boba::detail::device_sync();

      t_now += dt;
      num_steps += 1;
    }

    // Print the timing data
    size_t elapsed = test_timer.toc();

    BOBA_CALI_EXTERNAL_END("Timestepping");

    double converted_elapsed = double(elapsed) / 1.0e+03;
    double time_per_step = converted_elapsed / double(num_steps);
    double update_frequency = double(boba::product(solution.sizes())) / time_per_step;

    std::cout << "Test complete" << std::endl;
    std::cout << "Solve time (s): " << std::scientific << converted_elapsed << std::endl;
    std::cout << "Time/step (s): " << std::scientific << time_per_step << std::endl;
    std::cout << "Update frequency (DoF/step/s): " << std::scientific << update_frequency << std::endl;
    std::cout << "Compression rate:" << std::scientific << solution.compression_rate() << std::endl;

    // Now check the error
    auto [absolute_error, relative_error] = compute_errors(solution, t_now);

    std::cout << " L2 error (abs) = " << absolute_error << std::endl;
    std::cout << " L2 error (rel) = " << relative_error << std::endl;
    double error_factor = double(64) / double(solution.sizes(0));
    pass_or_fail(check, relative_error, 1.0e-2 * boba::pow(error_factor, 2.0));
  }
};

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();

  bool check = true;
  checkpoint();

  Parameters input;

  size_t solver_index = static_cast<size_t>(Scheme::Tensor);

  ::boba::argparser args(argc, argv);
  args.add_optional_argument(input.number_points_1d, "-N", "--resolution", "Points per dimension.");
  args.add_optional_argument(input.nu, "-nu", "--viscosity", "Viscosity.");
  args.add_optional_argument(input.end_time, "-T", "--end_time", "End time.");
  args.add_optional_argument(input.CFL_scaling, "-CFL", "--CFLnumber", "CFL number for the time stepping.");
  args.add_optional_argument(solver_index, "-s", "--solver_type", "Type of solver to use for the test (0=Tensor, 1=TT-cross w/maxvol, 2=TT-cross with DEIM).");
  args.parse_check();

  boba_always_assert_lt(solver_index, static_cast<size_t>(Scheme::number_of_schemes), "Invalid solver_type.");

  Scheme solver_type = static_cast<Scheme>(solver_index);

  switch (solver_type)
  {
  case Scheme::Tensor:
  {
    using Tensor_t = boba::Tensor<3, space, double>;
    BurgersSolver<Tensor_t> solver(input);
    solver.run_test(check);
    break;
  }

  case Scheme::TT_cross_maxvol:
  {
    using Tensor_t = boba::TensorTrain<3, space, double>;
    BurgersSolver<Tensor_t> solver(input);
    solver.compute_fluxes.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::MAXVOL;
    solver.scheme_label = "TT-cross maxvol";
    solver.run_test(check);
    break;
  }

  case Scheme::TT_cross_deim:
  {
    using Tensor_t = boba::TensorTrain<3, space, double>;
    BurgersSolver<Tensor_t> solver(input);
    solver.compute_fluxes.submatrix_selection_type =
      boba::DMRGCross<double>::SubmatrixSelectionType::DEIM;
    solver.scheme_label = "TT-cross DEIM";
    solver.run_test(check);
    break;
  }

  default:
    std::abort();
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
