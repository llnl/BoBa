// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*
 *
 * Tests the exp(tt) function using finite difference discretization for a few
 * exponent functions.
 *
 */

#include "../../tests/common.hpp"
#include "BOBA/boba.hpp"
#include "eigel_exponential.hpp"

struct parameter_struct
{
  std::size_t exponent_id = 1;
  std::size_t number_elements = boba::is_boba_debug_mode() ? 6 : (boba::is_ci_mode() ? 10 : 6);
  std::size_t inner_iterations = 20;
  std::size_t outer_iterations = 1;
  double tolerance = 1.0e-09;
  bool compare_with_full = false;
  bool verbose = false;
};

// essentially, minimum of absolute and relative errors
inline double compute_error(double abserr, double refval)
{
  if (refval < 1.0)
  {
    return abserr;
  }

  return abserr / refval;
}

template <std::size_t dimension, ::boba::execution_space space = ::boba::default_execution_space>
struct test_dimension
{
  using vector_t = ::boba::Vector<space, double>;
  using matrix_t = ::boba::Matrix<space, double>;

  using tt_vector_t = ::boba::TensorTrain<dimension, space, double>;
  using tt_matrix_t = ::boba::TensorTrainMatrix<dimension, space, double>;

  static void print_tt_stats(const tt_vector_t& tt, const std::string& name)
  {
    std::cout << name << " : size = " << tt.sizes_string() << ", rank = " << tt.ranks_string() << std::endl;
  }

  test_dimension(parameter_struct parameters, bool& check)
  {
    this->run_dimension(parameters, check);
  }

  void run_dimension(parameter_struct parameters, bool& check)
  {
    const std::size_t exponent_id = parameters.exponent_id;
    const std::size_t num_element = parameters.number_elements;
    const std::size_t inner_iterations = parameters.inner_iterations;
    const std::size_t outer_iterations = parameters.outer_iterations;
    const double tolerance = parameters.tolerance;
    const bool compare_with_full = parameters.compare_with_full;
    const bool verbose = parameters.verbose;

    boba_always_assert(exponent_id <= 3, "unknown exponent function ID");
    boba_always_assert(num_element >= 1, "number of elements must be at least one");
    boba_always_assert(tolerance > 1.0e-15, "tolerance is too small");

    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "D = " << dimension << ",  N = " << num_element << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;

    const auto sizes = ::boba::filled_array<dimension>(num_element);

    //
    // ========================
    // Function discretizations
    // ========================
    //
    // Assume domain [0, 1] x ... x [0, 1]
    //
    // Assume h(boundary) = 0. Clearly
    //
    //   h(0, ..., 0) = 0,
    //
    // hence
    //
    //   w(boundary) = exp(h(boundary)) / exp(h(0, ..., 0)) - 1 = 0
    //
    // and there is no need to explicitly enforce the initial condition on w.
    //
    //
    // These are the two functions we apply the derivative operation to. Since
    // these functions vanish on the boundary, there is no need to explicitly
    // track them. In 1D domain discretization, we can devote all N points to
    // interior discretization points. Thus, we divide [0, 1] using N + 2 nodes
    // (N explicit nodes in interior, 2 implicit nodes on boundary) with uniform
    // spacing h = 1.0 / (N + 1). The node points are
    //
    //   x(i) = (i + 1) * h,    0 <= i <= N - 1
    //
    // Thus x(0) = h and x(N - 1) = 1 - h
    //

    ::boba::Tensor<dimension, space, double> exponent_tensor(sizes);
    ::boba::Tensor<dimension, space, double> exponential_tensor(sizes);

    checkpoint();
    {
      auto exponent_tensor_view = exponent_tensor.view();

      const double spacing = 1.0 / (num_element + 1);

      ::boba::loop<space, 1>(exponent_tensor_view.size(), [=] __boba_host_device__(std::size_t i)
      {
        double value = 0.0;

        if (exponent_id == 0)
        {
          //
          // h(x) = 0.0
          //
          value = 0.0;
        }
        else if (exponent_id == 1)
        {
          //
          // h(x) = 4^d * prod_d x_d (1 - x_d)
          //
          const auto mid = exponent_tensor_view.multiindex(i);

          double prod = 1.0;
          for (std::size_t d = 0; d < dimension; d++)
          {
            const std::size_t id = mid[d];
            const double xd = (id + 1) * spacing;
            prod *= 4.0 * xd * (1.0 - xd);
          }

          value = prod;
        }
        else if (exponent_id == 2)
        {
          //
          // h(x) = prod_d sin(pi * x_d)
          //
          const auto mid = exponent_tensor_view.multiindex(i);

          double prod = 1.0;
          for (std::size_t d = 0; d < dimension; d++)
          {
            const std::size_t id = mid[d];
            const double xd = (id + 1) * spacing;
            prod *= ::boba::sin(::boba::pi * xd);
          }

          value = prod;
        }
        else if (exponent_id == 3)
        {
          //
          // h(x) = prod_d sin((d + 1) * pi * x_d)
          //
          const auto mid = exponent_tensor_view.multiindex(i);

          double prod = 1.0;
          for (std::size_t d = 0; d < dimension; d++)
          {
            const std::size_t id = mid[d];
            const double xd = (id + 1) * spacing;
            prod *= ::boba::sin((d + 1) * ::boba::pi * xd);
          }

          value = prod;
        }

        exponent_tensor_view(i) = value;
      });

      // u(x) = exp(h(x))
      auto exponential_tensor_view = exponential_tensor.view();
      ::boba::loop<space, 1>(exponent_tensor_view.size(), [=] __boba_host_device__(std::size_t i)
      {
        exponential_tensor_view(i) = ::boba::exp(exponent_tensor_view(i));
      });
    }

    //
    // =================
    // Gradient operator
    // =================
    //
    // We use central-difference scheme. Since functions are assumed to be
    // zero on boundary, and we only track the interior values, there is no need
    // to modify the finite difference matrix.
    //

    // non-zero value for the 1D gradient matrix: 1 / (2 * h)
    const double value = 0.5 * (static_cast<double>(num_element) + 1.0);

    checkpoint();

    ::boba::Array<tt_matrix_t, dimension> gradient_tt;
    {
      checkpoint();
      ::boba::Tensor<4, space, double> identity_tt_core({1, num_element, num_element, 1});

      identity_tt_core.fill_with_zeros();
      auto identity_tt_core_view = identity_tt_core.view();

      ::boba::Tensor<4, space, double> gradient_tt_core({1, num_element, num_element, 1});
      gradient_tt_core.fill_with_zeros();
      auto gradient_tt_core_view = gradient_tt_core.view();

      ::boba::loop<space, 1>(num_element, [=] __boba_host_device__(std::size_t i)
      {
        identity_tt_core_view({0, i, i, 0}) = 1.0;
      });

      ::boba::loop<space, 1>(num_element - 1, [=] __boba_host_device__(std::size_t i)
      {
        gradient_tt_core_view({0, i, i + 1, 0}) = 1.0;
        gradient_tt_core_view({0, i + 1, i, 0}) = -1.0;
      });

      for (size_t k = 0; k < dimension; ++k)
      {
        gradient_tt[k].resize(sizes, sizes);
        for (size_t d = 0; d < dimension; ++d)
        {
          gradient_tt[k].cores[d] = (k == d) ? gradient_tt_core : identity_tt_core;
        }
      }
    }

    checkpoint();

    ::boba::Array<matrix_t, dimension> gradient_full;
    if (compare_with_full)
    {
      matrix_t identity_matrix({num_element, num_element});
      identity_matrix.fill_with_zeros();
      auto identity_matrix_view = identity_matrix.view();

      matrix_t gradient_matrix({num_element, num_element});
      gradient_matrix.fill_with_zeros();
      auto gradient_matrix_view = gradient_matrix.view();

      ::boba::loop<space, 1>(num_element, [=] __boba_host_device__(std::size_t i)
      {
        identity_matrix_view({i, i}) = 1.0;
      });

      ::boba::loop<space, 1>(num_element - 1, [=] __boba_host_device__(std::size_t i)
      {
        gradient_matrix_view({i, i + 1}) = 1.0;
        gradient_matrix_view({i + 1, i}) = -1.0;
      });

      for (size_t k = 0; k < dimension; ++k)
      {
        gradient_full[k] = matrix_t({1, 1});
        gradient_full[k].fill_with(1.0);
        for (size_t d = 0; d < dimension; ++d)
        {
          gradient_full[k] = tensor_product(gradient_full[k], (k == d) ? gradient_matrix : identity_matrix);
        }
      }
    }

    checkpoint();

    //
    // ===================
    // Evaluation operator
    // ===================
    //
    // Evaluates the function discretization at the origin. This is used to
    // enforce the initial condition. In this case, the evaluated value assumed
    // to be zero, so zero matrix suffices.
    //

    tt_matrix_t evaluation_tt(::boba::filled_array<dimension>(num_element), sizes);
    evaluation_tt.fill_with_zeros();

    matrix_t evaluation_full;
    if (compare_with_full)
    {
      evaluation_full = matrix_t({1, ::boba::pow(num_element, dimension)});
      evaluation_full.fill_with_zeros();
    }

    //
    // ================
    // Exponent vectors
    // ================
    //

    checkpoint();

    tt_vector_t exponent_tt = ::boba::compress_to_TensorTrain(exponent_tensor);

    vector_t exponent_full;
    if (compare_with_full)
    {
      exponent_full = flatten(exponent_tensor);
    }

    //
    // =================
    // Linear Solver: CG
    // =================
    //

    ::boba::Krylov<tt_matrix_t, tt_vector_t> solver_tt(outer_iterations, inner_iterations, tolerance);
    solver_tt.verbose = verbose;

    ::boba::Krylov<matrix_t, vector_t> solver_full(outer_iterations, inner_iterations, tolerance);
    solver_full.verbose = verbose;

    solver_tt.method = ::boba::KrylovMethods::conjugate_gradient;
    solver_full.method = ::boba::KrylovMethods::conjugate_gradient;

    //
    // =======================
    // Exponential Computation
    // =======================
    //

    ::boba::TicToc<::boba::tictoc_units::microseconds> timer_tt;
    tt_vector_t exponential_eigel_tt = eigel_exponential(exponent_tt, gradient_tt, evaluation_tt, 1.0, solver_tt);
    timer_tt.end();
    size_t elapsed_tt = timer_tt.timing();

    size_t elapsed_full = 0;
    vector_t exponential_eigel_full;
    if (compare_with_full)
    {
      ::boba::TicToc<::boba::tictoc_units::microseconds> timer_full;
      exponential_eigel_full = eigel_exponential(exponent_full, gradient_full, evaluation_full, 1.0, solver_full);
      timer_full.end();
      elapsed_full = timer_full.timing();
    }

    //
    // =============
    // Quality Check
    // =============
    //

    std::cout << std::scientific;

    double exponent_norm = ::boba::norm_frobenius(exponent_tensor);
    double exponential_norm = ::boba::norm_frobenius(exponential_tensor);

    double exponent_tt_error = compute_error(::boba::norm_frobenius(exponent_tensor - exponent_tt.decompress()), exponent_norm);

    print_tt_stats(exponent_tt, " exponent_tt                       ");
    std::cout << " exponent_tt_error                  : " << exponent_tt_error << std::endl
              << std::endl;

    double exponential_eigel_tt_error = compute_error(::boba::norm_frobenius(exponential_tensor - exponential_eigel_tt.decompress()), exponential_norm);

    print_tt_stats(exponential_eigel_tt, " exponential_eigel_tt              ");
    std::cout << " exponential_eigel_tt_error         : " << exponential_eigel_tt_error << std::endl;
    std::cout << " exponential_eigel_tt_time          : " << elapsed_tt << " us" << std::endl
              << std::endl;

    tt_vector_t exponential_tt = ::boba::compress_to_TensorTrain(exponential_tensor);
    double exponential_tt_error = compute_error(::boba::norm_frobenius(exponential_tensor - exponential_tt.decompress()), exponential_norm);

    print_tt_stats(exponential_tt, " exponential_tt                    ");
    std::cout << " exponential_tt_error               : " << exponential_tt_error << std::endl
              << std::endl;

    if (compare_with_full)
    {
      vector_t exponential_full = flatten(exponential_tensor);
      double exponential_eigel_full_error = compute_error(::boba::norm_frobenius(exponential_full - exponential_eigel_full), exponential_norm);

      std::cout << " exponential_eigel_full_error       : " << exponential_eigel_full_error << std::endl;
      std::cout << " exponential_eigel_full_time        : " << elapsed_full << " us" << std::endl
                << std::endl;

      std::cout << " comparing TT Eigel exponential against full Eigel exponential:" << std::endl;
      pass_or_fail(check, ::boba::norm_frobenius(exponential_eigel_full - flatten(exponential_eigel_tt.decompress())), tolerance);
    }

    std::cout << std::defaultfloat;
  }
};

int main(int argc, char* argv[])
{
  BOBA_CALI_EXTERNAL_MARK
  ::boba::splash();
  ::boba::init();

  parameter_struct parameters;

  std::size_t dimension = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(parameters.exponent_id, "-e", "--exponent", "Exponent function ID (0: constant, 1: quadratic, 2: trigonometric)");
  args.add_optional_argument(dimension, "-d", "--dimensions", "Dimensions in which to run test.");
  args.add_optional_argument(parameters.number_elements, "-n", "--resolution", "Number of elements in each dimension.");
  args.add_optional_argument(parameters.outer_iterations, "-oi", "--outer_iters", "Number of outer iterations to take.");
  args.add_optional_argument(parameters.inner_iterations, "-ii", "--inner_iters", "Number of inner iterations to take.");
  args.add_optional_argument(parameters.tolerance, "-t", "--tolerance", "Krylov solver tolerance.");
  args.add_optional_argument(parameters.compare_with_full, "-ct", "--compare_with_full", "Compare TT Eigel exponential against full version.");
  args.add_optional_argument(parameters.verbose, "-vt", "--verbose", "Krylov verbose output");

  args.parse_check();

  checkpoint();

  bool check = true;

  std::cout << std::endl;

  if (dimension == 0)
  {
    test_dimension<2> runner2(parameters, check);
    test_dimension<3> runner3(parameters, check);
  }
  if (dimension == 2)
  {
    test_dimension<2> runner(parameters, check);
  }
  if (dimension == 3)
  {
    test_dimension<3> runner(parameters, check);
  }
  if (dimension == 4)
  {
    test_dimension<4> runner(parameters, check);
  }
  if (dimension == 5)
  {
    test_dimension<5> runner(parameters, check);
  }
  if (dimension == 6)
  {
    test_dimension<6> runner(parameters, check);
  }

  checkpoint();
  ::boba::finalize();

  return final_check(check);
}
