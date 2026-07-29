// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../../tests/common.hpp"
#include "../../tests/common_ttm.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
  This example demonstrates one possible implementation of explicit time-stepping in the tensor-train format for a structured operator equation

  u_{,t} + \vec c \cdot grad(u) - \beta*Lap(u) + \kappa*u = 0

  with periodic boundary conditions.
  Note that the parameters can be chosen at runtime via argparser.

*/

enum schemes : int
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
  double approximate_end_time = 2.0;
  double advection = 1.0;
  double diffusion = 0.0;
  double source = 0.0;
  size_t number_points_1d = 20;
  double domain_begin = -1.0;
  double domain_end = 1.0;
  double CFL_scaling = 0.5;
};

template <size_t dimension>
struct run
{

  boba::TensorTrain<dimension, space, double> compute_update(
    boba::TensorTrainMatrix<dimension, space, double>& spatial_operator,
    boba::TensorTrain<dimension, space, double>& previous_solution)
  {
    BOBA_CALI_EXTERNAL_MARK
    boba::TensorTrain<dimension, space, double> update = spatial_operator * previous_solution;
    return update;
  }

  run(
    bool& check,
    double& error,
    parameters input)
  {
    run_test(
      check,
      error,
      input);
  }

  void run_test(
    bool& check,
    double& error,
    parameters input)
  {
    BOBA_CALI_EXTERNAL_BEGIN("adv_setup")
    std::cout << "%---------------------------------------------------------" << std::endl;
    std::cout << "% Running test " << std::endl;
    std::cout << "% number_points_1d = " << input.number_points_1d << std::endl;
    std::cout << "% dimension = " << dimension << std::endl;
    std::cout << "%---------------------------------------------------------" << std::endl;

    size_t N = input.number_points_1d;
    auto sizes = ::boba::filled_array<dimension>(N);

    bool verbose = boba::is_env_nonempty("VERBOSE");
    bool file_dump = boba::is_env_nonempty("FILE_DUMP");

    // -------------------------------------------------
    // Define operators
    // -------------------------------------------------
    double wavespeed = input.advection;
    double kappa = input.source;
    double beta = input.diffusion;
    double domain_size = input.domain_end - input.domain_begin;

    // see common_ttm.hpp
    common_ttm<dimension, space, double> operators(N, domain_size, false);
    double dx = operators.dx;
    double dx2 = dx * dx;
    double odx = 1.0 / dx;
    double odx2 = 1.0 / dx2;

    boba::TensorTrainMatrix<dimension, space, double> spatial_operator(sizes, sizes);
    spatial_operator.rename("spatial_operator");
    //
    // advection term
    //
    if (::boba::abs(wavespeed) > 0)
    {
      spatial_operator += (-wavespeed * odx) * operators.advection_forward_periodic;
    }
    //
    // diffusion term
    //
    if (::boba::abs(beta) > 0)
    {
      spatial_operator += (beta * odx2) * operators.laplacian_periodic;
    }
    //
    // source term
    //
    if (::boba::abs(kappa) > 0)
    {
      spatial_operator += (-kappa) * operators.identity;
    }
    boba_print(spatial_operator.compression_rate());
    boba_print(spatial_operator.number_nonzeros(1.0e-10));
    boba_print(spatial_operator.sparsity(1.0e-10));
    spatial_operator.round();
    boba_print(spatial_operator.compression_rate());
    boba_print(spatial_operator.number_nonzeros(1.0e-10));
    boba_print(spatial_operator.sparsity(1.0e-10));

    //
    // Determine the time-step size
    //

    double diff_limit = (beta > 0) ? dx * dx / beta / double(dimension) : 1.0;
    double adv_limit = (::boba::abs(wavespeed) > 0) ? dx / ::boba::abs(wavespeed) / double(dimension) : 1.0;
    double source_limit = (::boba::abs(kappa) > 0) ? dx / ::boba::abs(kappa) / double(dimension) : 1.0;
    double cfl_dimension_factor = input.CFL_scaling / double(dimension);
    double deltat = cfl_dimension_factor * boba::min(boba::min(diff_limit, adv_limit), source_limit);
    double time = 0.0;

    //
    // Initial Value
    //
    boba::TensorTrain<dimension, space, double> initial_value(sizes);
    initial_value.rename("initial_value");
    initial_value.fill_with_zeros();
    double blob_radius = 0.63;
    auto initial_function = [=] __boba_host_device__(double x)
    {
      double x_reference = ::boba::periodic(x, input.domain_begin, input.domain_end);
      double x2 = ::boba::pow(x_reference, 2.);
      double r2 = ::boba::pow(blob_radius, 2.);
      double f = 0.0;
      if (x2 < r2)
      {
        f = boba::exp(1. / r2 + 1. / (x2 - r2));
      }
      return f;
    };
    {
      for (size_t d = 0; d < dimension; d++)
      {
        auto initial_value_view = initial_value.cores[d].view();
        ::boba::loop<space, 1>({N},
                               [=] __boba_host_device__(size_t i)
        {
          double x = input.domain_begin + dx * double(i);
          initial_value_view({0, i, 0}) = initial_function(x);
        });
      }
    }
    auto initial_value_full = initial_value.decompress();

    if (file_dump)
    {
      initial_value_full.rename(::boba::name_flag("initial_value_full"));
      boba::write_to_file(initial_value_full);
    }

    boba::TensorTrain<dimension, space, double> solution(sizes);
    solution.rename("solution");
    solution = initial_value;
    //
    // full
    //
    boba::Tensor<dimension, space, double> solution_full(solution.sizes());
    solution_full.rename("solution_full");
    solution_full = initial_value_full;

    std::cout << "% Initial conditions " << std::endl;
    std::cout << "%   Boba size              : " << solution.get_number_elements() << std::endl;
    std::cout << "%   Compression rate       : " << solution.compression_rate() << "x" << std::endl;
    std::cout << "%   Ranks                  : " << solution.ranks_string() << std::endl;
    std::cout << "%   deltat                  : " << deltat << std::endl;
    {
      auto soln_tensor = solution.decompress();
      soln_tensor.rename("solution_tensor");

      double norm_solution = ::boba::norm_frobenius(soln_tensor);
      std::cout << "%   Norm of solution = " << norm_solution << std::endl;
      double norm_full = ::boba::norm_frobenius(solution_full);
      std::cout << "%   Norm of full solution = " << norm_full << std::endl;
      double norm_error = ::boba::norm_difference_frobenius(soln_tensor, solution_full);
      std::cout << "%   Norm of difference, L2 = " << norm_error << std::endl;
      double norm_error_relative = norm_error / norm_full;
      std::cout << "%   Norm of difference, relative = " << norm_error_relative << std::endl;
    }

    BOBA_CALI_EXTERNAL_SWITCH("adv_setup", "adv_timestepping");

    //
    // Time-stepping loop
    //
    solution.svd_tolerance_absolute = 1.e-09;
    solution.svd_tolerance_relative = 1.e-12;

    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::tt]))
    {
      boba::TicToc<tictoc_units> timer_timestepping;
      bool timestepping = true;
      size_t cyclecount = 0;
      while (timestepping)
      {
        time = time + deltat;
        cyclecount++;

        // u_{n+1} = u_{n} + operator(u_{n});
        auto update = compute_update(spatial_operator, solution);
        update *= deltat;
        solution += update;
        solution.round();

        if (time > input.approximate_end_time)
        {
          timestepping = false;
        }
        if (verbose)
        {
          std::cout << "% Current time: " << time << std::endl
                    << "% --------------------- " << std::endl;
          std::cout << "% Current index: " << cyclecount << std::endl
                    << "% --------------------- " << std::endl;
        }
        boba::detail::device_sync();

      } // end timestepping loop

      //
      BOBA_CALI_EXTERNAL_END("adv_timestepping");
      //

      timer_timestepping.end();
      std::cout << "% Timestepping complete " << std::endl;

      std::cout << "% Results " << std::endl;
      std::cout << "%   Boba size              : " << solution.get_number_elements() << " : Time : " << timer_timestepping.timing() << std::endl;
      std::cout << "%   Compression rate       : " << solution.compression_rate() << "x" << std::endl;
      std::cout << "%   Ranks                  : " << solution.ranks_string() << std::endl;

      if (file_dump)
      {
        auto solution_decompressed = solution.decompress();
        std::string filename = ::boba::name_flag("examples/explicit_solution");
        boba::write_to_file(solution_decompressed, filename);
        auto solution_decompressed_copy = solution_decompressed;
        boba::read_from_file(solution_decompressed, filename);
        pass_or_fail(check, ::boba::norm_difference_inf(solution_decompressed, solution_decompressed_copy), 1.0e-10);
      }
    }

    //
    // Time-stepping loop - full tensor
    //
    if (boba::env_match_or_empty("SCHEME", schemes_strings[schemes::full]))
    {
      boba::Tensor<dimension, space, double> previous_solution(solution.sizes());
      boba::TicToc<tictoc_units> timer_timestepping;
      bool timestepping = true;
      size_t cyclecount = 0;
      time = 0;
      while (timestepping)
      {
        time = time + deltat;
        cyclecount++;

        previous_solution = solution_full;
        auto u_view = previous_solution.const_view();
        auto solution_view = solution_full.view();

        ::boba::loop<space, dimension>(::boba::filled_array<dimension>(N),
                                       [=] __boba_host_device__(::boba::Array<size_t, dimension> ids)
        {
          double fijk = u_view(ids);

          double diffusion_update = 0;
          for (size_t d = 0; d < dimension; d++)
          {
            auto stencil_ids_p1 = ids;
            auto stencil_ids_m1 = ids;
            stencil_ids_m1[d] = boba::mod((ids[d] + N) - 1, N);
            stencil_ids_p1[d] = boba::mod((ids[d] + N) + 1, N);

            diffusion_update += u_view(stencil_ids_m1);
            diffusion_update += -2.0 * fijk;
            diffusion_update += u_view(stencil_ids_p1);
          }
          diffusion_update *= beta * odx2;

          double advection_update = 0;
          for (size_t d = 0; d < dimension; d++)
          {
            auto stencil_ids_m1 = ids;
            stencil_ids_m1[d] = boba::mod((ids[d] + N) - 1, N);

            advection_update += -1.0 * u_view(stencil_ids_m1);
            advection_update += fijk;
          }
          advection_update *= wavespeed * odx;

          double source_update = kappa * fijk;

          double update = diffusion_update - source_update - advection_update;

          solution_view(ids) = fijk + deltat * update;
        });

        if (time > input.approximate_end_time)
        {
          timestepping = false;
        }
        if (verbose)
        {
          std::cout << "% Current time: " << time << std::endl
                    << "% --------------------- " << std::endl;
          std::cout << "% Current index: " << cyclecount << std::endl
                    << "% --------------------- " << std::endl;
        }
        boba::detail::device_sync();
      } // end timestepping loop

      if (file_dump)
      {
        solution_full.rename(::boba::name_flag("solution_full"));
        boba::write_to_file(solution_full);
      }

      //
      BOBA_CALI_EXTERNAL_END("adv_timestepping");
      //

      timer_timestepping.end();
      std::cout << "% Timestepping (full) complete " << std::endl;

      std::cout << "% Results " << std::endl;
      std::cout << "%   : : Time : " << timer_timestepping.timing() << std::endl;
      std::cout << "% --------------------- " << std::endl;
    }

    //
    // Error
    //
    auto soln_tensor = solution.decompress();
    soln_tensor.rename("solution_tensor");

    double norm_solution = ::boba::norm_frobenius(soln_tensor);
    std::cout << "%   Norm of solution = " << norm_solution << std::endl;
    double norm_full = ::boba::norm_frobenius(solution_full);
    std::cout << "%   Norm of full solution = " << norm_full << std::endl;
    double norm_error = ::boba::norm_difference_frobenius(soln_tensor, solution_full);
    std::cout << "%   Norm of difference, L2 = " << norm_error << std::endl;
    double norm_error_relative = norm_error / norm_full;
    std::cout << "%   Norm of difference, relative = " << norm_error_relative << std::endl;

    // Error used for CI check
    error = norm_error_relative;
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
    double error = -1;
    for (size_t r = 0; r < averages; r++)
    {
      run<dimension> runner(
        check,
        error,
        input);
      pass_or_fail(check, error, 1.0e-07);
    }
  }
};

int main(int argc, char* argv[])
{

  boba::splash();
  std::cout << "Example explicit timestepper" << std::endl;
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  parameters input;

  size_t dimension = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(input.number_points_1d,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(input.advection,
                             "-c",
                             "--advection",
                             "Coefficient for advection term.");

  args.add_optional_argument(input.diffusion,
                             "-d",
                             "--diffusion",
                             "Coefficient for diffusion term.");

  args.add_optional_argument(input.source,
                             "-s",
                             "--source",
                             "Coefficient for source term.");

  args.add_optional_argument(input.approximate_end_time,
                             "-T",
                             "--end_time",
                             "The time-stepper will end once it has passed this time, but will not change dt to exactly match it.");

  args.add_optional_argument(dimension,
                             "-D",
                             "--dimension",
                             "Number of dimensions (2 or 3).");

  args.add_optional_argument(input.domain_begin,
                             "-db",
                             "--domainBegin",
                             "Starting value of domain.");

  args.add_optional_argument(input.domain_end,
                             "-de",
                             "--domainEnd",
                             "Ending value of domain.");

  args.add_optional_argument(input.CFL_scaling,
                             "-CS",
                             "--CFLscaling",
                             "Scalar Multiple on CFL term.");

  args.parse_check();

  size_t averages = 1;
  if ((dimension == 2) or (dimension == 0))
  {
    run_wrapper<2> runner(
      check,
      averages,
      input);
  }
  if ((dimension == 3) or (dimension == 0))
  {
    run_wrapper<3> runner(
      check,
      averages,
      input);
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
