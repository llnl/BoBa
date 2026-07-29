// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

enum solver_schemes : size_t
{
  tt = 0,
  tensor = 1,
  vec = 2,
  tucker = 3,
  block_tt = 4,
  qtt = 5,
  number_of_schemes,
};

struct parameter_struct
{
  size_t number_elements_1d = boba::is_boba_debug_mode() ? 6 : 10;
  size_t outer_iterations = 20;
  size_t inner_iterations = 40;
  size_t solver = 0;
  size_t krylov_method = 0;
  bool use_bandwidth = false;
};

template <size_t dimension, typename operator_t, typename vector_t, typename solver_t>
struct test_dimension
{

  test_dimension(
    parameter_struct parameters,
    bool& check)
  {
    this->run_dimension(
      parameters,
      check);
  }

  //
  // These converter functions will convert the initial conditions to the type we will actually use in the solver
  //

  template <typename solver_operator_t>
    requires std::same_as<solver_operator_t, ::boba::TensorTrainMatrix<dimension, space, double>>
  operator_t solver_operator(::boba::TensorTrainMatrix<dimension, space, double>& A)
  {
    return A;
  }

  template <typename solver_operator_t>
    requires std::same_as<solver_operator_t, boba::BlockOperator<boba::TensorTrainMatrix<dimension, space, double>>>
  operator_t solver_operator(::boba::TensorTrainMatrix<dimension, space, double>& A)
  {
    boba::BlockOperator<::boba::TensorTrainMatrix<dimension, space, double>> block_A(2, 2);
    block_A({0, 0}) = A;
    block_A({0, 1}) = 0.0 * A;
    block_A({1, 0}) = 0.0 * A;
    block_A({1, 1}) = A;

    block_A({0, 1}).round();
    block_A({1, 0}).round();
    return block_A;
  }

  template <typename solver_operator_t>
    requires std::same_as<solver_operator_t, ::boba::TuckerMatrix<dimension, space, double>>
  operator_t solver_operator(::boba::TensorTrainMatrix<dimension, space, double>& A)
  {
    return ::boba::TensorTrainMatrix_to_TuckerMatrix(A);
  }

  template <typename solver_operator_t>
    requires std::same_as<solver_operator_t, ::boba::TensorTrainMatrix<dimension, space, double>>
  operator_t solver_operator(::boba::TuckerMatrix<dimension, space, double>& A)
  {
    return ::boba::TuckerMatrix_to_TensorTrainMatrix(A);
  }

  template <typename solver_operator_t>
    requires std::same_as<solver_operator_t, ::boba::QuantizedTensorTrainMatrix<space, double>>
  operator_t solver_operator(::boba::TensorTrainMatrix<dimension, space, double>& A)
  {
    auto exponent = A.get_number_cores();
    auto rows_base = A.core_rows(0);
    auto cols_base = A.core_cols(0);
    boba::QuantizedTensorTrainMatrix<space, double> qA(rows_base, cols_base, exponent);
    for (size_t d = 0; d < dimension; d++)
    {
      qA.cores[d] = A.cores[d];
    }
    return qA;
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::TensorTrain<dimension, space, double>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    return v;
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::BlockVector<::boba::TensorTrain<dimension, space, double>>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    ::boba::BlockVector<::boba::TensorTrain<dimension, space, double>> block_v(2);
    block_v(0) = v;
    block_v(1) = v;
    return block_v;
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::Vector<space, double>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    auto solver_tensor = v.decompress();
    ::boba::Vector<space, double> solver_vector({solver_tensor.size()});
    solver_vector.reshape(solver_tensor);
    return solver_vector;
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::Tensor<dimension, space, double>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    return v.decompress();
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::Tucker<dimension, space, double>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    return tt_to_tucker(v);
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::TensorTrain<dimension, space, double>>
  vector_t solver_vector(::boba::Tucker<dimension, space, double>& v)
  {
    return tucker_to_TensorTrain(v);
  }

  template <typename solver_vector_t>
    requires std::same_as<solver_vector_t, ::boba::QuantizedTensorTrain<space, double>>
  vector_t solver_vector(::boba::TensorTrain<dimension, space, double>& v)
  {
    auto exponent = v.get_number_cores();
    auto base = v.sizes(0);
    boba::QuantizedTensorTrain<space, double> solver_vector(base, exponent);
    for (size_t d = 0; d < dimension; d++)
    {
      solver_vector.cores[d] = v.cores[d];
    }
    return solver_vector;
  }

  //
  //
  //

  void run_dimension(
    parameter_struct parameters,
    bool& check)
  {
    size_t outer_iterations = parameters.outer_iterations;
    size_t inner_iterations = parameters.inner_iterations;
    size_t number_elements_1d = parameters.number_elements_1d;
    boba_always_assert(outer_iterations > 0, "Nonpositive outer_iterations");
    boba_always_assert(inner_iterations > 0, "Nonpositive inner_iterations");

    double tolerance = 1.0e-6;
    size_t N = number_elements_1d;
    auto sizes = ::boba::filled_array<dimension>(N);

    // Generate the TTM operators
    common_ttm<dimension, space, double> operators(N, 1.0, false);

    bool laplacian_matrix = true;
    boba::TensorTrainMatrix<dimension, space, double> A_operator(sizes, sizes);
    A_operator.fill_with_zeros();
    if (laplacian_matrix)
    {
      A_operator += operators.laplacian_interior;
    }
    else
    {
      A_operator += operators.advection_forward_interior;
    }
    A_operator += operators.boundaries;

    checkpoint();
    boba::TensorTrain<dimension, space, double> exact(sizes);
    exact.rename("exact");
    exact.resize(sizes);
    for (size_t d = 0; d < dimension; d++)
    {
      auto core_view = exact.cores[d].view();

      ::boba::loop<space, 1>(sizes[d],
                             [=] __boba_host_device__(size_t id)
      {
        double x = double(id) / double(number_elements_1d - 1);
        double r2 = ::boba::pow(x - 0.5, 2);
        double R2 = 0.3;
        double y = 0.0;
        if (r2 < R2)
        {
          y = ::boba::exp(1.0 / (r2 - R2) + 1.0 / R2);
        }
        core_view({0, id, 0}) = y;
      });
    }

    boba::TensorTrain<dimension, space, double> b(sizes);
    b.rename("b");

    checkpoint();
    // b = A*x
    if (parameters.use_bandwidth)
    {
      A_operator.determine_bandwidth();
    }

    b = A_operator * exact;
    // Set to wrong answer
    checkpoint();
    boba::TensorTrain<dimension, space, double> initial_guess(sizes);
    initial_guess.rename("initial_guess");
    initial_guess.fill_with_zeros();

    boba::TensorTrain<dimension, space, double> solution(sizes);
    solution.rename("solution");

    checkpoint();

    double initial_residual_norm = -1.0;
    double residual_norm = -1.0;

    //
    // From the tt and ttm types used in setup, now convert to the types the solver will use
    //

    auto solver_A_operator = solver_operator<operator_t>(A_operator);
    auto solver_initial_guess = solver_vector<vector_t>(initial_guess);
    auto solver_b = solver_vector<vector_t>(b);
    auto solver_solution = solver_vector<vector_t>(solution);
    auto solver_exact = solver_vector<vector_t>(exact);

    size_t operator_compressed_size = solver_A_operator.get_number_elements();

    solver_t solver(
      outer_iterations,
      inner_iterations,
      tolerance);
    solver.method = parameters.krylov_method;
    solver.set_matrix(solver_A_operator);

    std::cout << "Running solver" << std::endl;
    initial_residual_norm = ::boba::norm_frobenius(solver.compute_residual(solver_initial_guess, solver_b));
    boba::TicToc<tictoc_units> solve_system;
    solver.solve(solver_b, solver_initial_guess, solver_solution);
    solve_system.end();
    auto used_outer_iterations = solver.used_outer_iterations;
    auto used_inner_iterations = solver.used_inner_iterations;

    std::cout << "Complete, computing residual_norm" << std::endl;
    residual_norm = ::boba::norm_frobenius(solver.compute_residual(solver_solution, solver_b));

    size_t matrix_size = boba::pow(number_elements_1d, 2 * dimension);
    size_t dofs = boba::pow(number_elements_1d, dimension);
    size_t solver_timing = solve_system.timing();
    size_t compressed_dofs = solver_solution.get_number_elements();
    double rhs_norm = ::boba::norm_frobenius(solver_b);
    double error = 0.0;
    if constexpr (std::same_as<vector_t, ::boba::QuantizedTensorTrain<space, double>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution.decompress(), solver_exact.decompress());
    }
    else if constexpr (std::same_as<vector_t, ::boba::TensorTrain<dimension, space, double>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution, solver_exact);
    }
    else if constexpr (std::same_as<vector_t, ::boba::Tucker<dimension, space, double>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution, solver_exact);
    }
    else if constexpr (std::same_as<vector_t, ::boba::BlockVector<::boba::TensorTrain<dimension, space, double>>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution, solver_exact);
    }
    else if constexpr (std::same_as<vector_t, ::boba::Tensor<dimension, space, double>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution, solver_exact);
    }
    else if constexpr (std::same_as<vector_t, ::boba::Vector<space, double>>)
    {
      error = ::boba::norm_difference_frobenius(solver_solution, solver_exact);
    }
    else
    {
      static_assert(std::same_as<vector_t, void>, "Unhandled Krylov vector type for error computation.");
    }
    double throughput = (double(dofs) / 1.e9) * double(used_inner_iterations) / (double(solver_timing) / 1.e6);

    std::cout << "effective  dofs           : " << dofs << std::endl;
    std::cout << "real dofs                 : " << compressed_dofs << std::endl;
    std::cout << "dofs compression rate     : " << divide_string(dofs, compressed_dofs) << "x" << std::endl;
    std::cout << "Matrix size               : " << matrix_size << std::endl;
    std::cout << "Compressed operator size  : " << operator_compressed_size << std::endl;
    std::cout << "Matrix compression rate   : " << divide_string(matrix_size, operator_compressed_size) << " x" << std::endl;
    std::cout << "Total inner iterations    : " << used_inner_iterations << std::endl;
    std::cout << "Outer iterations          : " << used_outer_iterations << "  of max " << outer_iterations << std::endl;
    std::cout << "Initial residual norm     : " << initial_residual_norm << std::endl;
    std::cout << "Rhs l2 norm               : " << rhs_norm << std::endl;
    std::cout << "Residual l2 norm          : " << residual_norm << std::endl;
    std::cout << "Residual begin/end        : " << initial_residual_norm / residual_norm << std::endl;
    std::cout << "Error l2 norm             : " << error << std::endl;
    std::cout << "Solver time               : " << solver_timing << " " << solve_system.units_string << std::endl;
    std::cout << "Throughput                : " << std::scientific << throughput << " gigadofs*iters/node/second " << std::endl;

    pass_or_fail(check, error, 1.e-03);
  }
};

void running(std::string scheme_name, size_t dimension, size_t number_elements_1d, bool use_bandwidth)
{
  std::cout << "---------------------------------------------------------------------" << std::endl;
  std::cout << scheme_name << " for dimension = " << dimension << ",  N = " << number_elements_1d << std::endl;
  std::cout << "bandwidth mode = " << (use_bandwidth ? "on" : "off") << std::endl;
  std::cout << "---------------------------------------------------------------------" << std::endl;
}

template <size_t dimension>
void run(bool& check, parameter_struct parameters)
{
  auto krylov_method = parameters.krylov_method;
  boba_always_assert_lt(krylov_method, static_cast<size_t>(solver_schemes::number_of_schemes), "Invalid scheme selection.");
  if (parameters.solver == solver_schemes::tt)
  {
    using operator_t = boba::TensorTrainMatrix<dimension, space, double>;
    using vector_t = boba::TensorTrain<dimension, space, double>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("ttm*tt " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
  if (parameters.solver == solver_schemes::tucker)
  {
    using operator_t = boba::TuckerMatrix<dimension, space, double>;
    using vector_t = boba::Tucker<dimension, space, double>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("TuckerMatrix*Tucker " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
  if (parameters.solver == solver_schemes::tensor)
  {
    using operator_t = boba::TensorTrainMatrix<dimension, space, double>;
    using vector_t = boba::Tensor<dimension, space, double>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("ttm*tensor " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
  if (parameters.solver == solver_schemes::vec)
  {
    using operator_t = boba::TensorTrainMatrix<dimension, space, double>;
    using vector_t = boba::Vector<space, double>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("ttm*vector " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
  if (parameters.solver == solver_schemes::block_tt)
  {
    using operator_t = ::boba::BlockOperator<boba::TensorTrainMatrix<dimension, space, double>>;
    using vector_t = boba::BlockVector<boba::TensorTrain<dimension, space, double>>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("ttm*vector " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
  if (parameters.solver == solver_schemes::qtt)
  {
    using operator_t = boba::QuantizedTensorTrainMatrix<space, double>;
    using vector_t = boba::QuantizedTensorTrain<space, double>;
    using solver_t = boba::Krylov<operator_t, vector_t>;
    running("qttm*qtt " + solver_t::method_string(krylov_method), dimension, parameters.number_elements_1d, parameters.use_bandwidth);
    test_dimension<dimension, operator_t, vector_t, solver_t> runner(parameters, check);
  }
}

int main(int argc, char* argv[])
{
  BOBA_CALI_EXTERNAL_MARK
  boba::splash();
  boba::init();
  std::cout << "Tests for BoBa Krylov solvers " << std::endl;

  parameter_struct parameters;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(parameters.number_elements_1d,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(parameters.solver,
                             "-s",
                             "--solver",
                             "Choice of solver scheme.");

  args.add_optional_argument(parameters.outer_iterations,
                             "-oi",
                             "--outer_iters",
                             "Number of outer iterations to take.");

  args.add_optional_argument(parameters.inner_iterations,
                             "-ii",
                             "--inner_iters",
                             "Number of inner iterations to take.");

  args.add_optional_argument(parameters.krylov_method,
                             "-m",
                             "--krylov_method",
                             "Choice of krylov method.");

  args.add_optional_argument(parameters.use_bandwidth,
                             "-bt",
                             "--bandwidth",
                             "Enable tensor-train-matrix bandwidth acceleration.");

  args.parse_check();

  checkpoint();

  bool check = true;

  run<1>(check, parameters);
  run<2>(check, parameters);
  run<3>(check, parameters);

  checkpoint();
  boba::finalize();

  return final_check(check);
}
