// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::seconds;

/*
  This test solves a 2-block linear system whose diagonal blocks are tensor-product
  discretizations of Poisson operators on different tensor grids:

    -Delta u_0 + C_01[u_1] = f_0
    -Delta u_1 + C_10[u_0] = f_1

  where each off-diagonal coupling C_ij is a rectangular separable operator of the
  form

    C_ij = sum_{k=0}^{q-1} sigma_k \bigotimes_d M^{(d)}_{ij,k},

  with q = coupling_rank, a scalar coefficient sigma_k applied once per
  separable term, and one-dimensional factors

    M^{(d)}_{ij,k}(m, n)
      = 1 / sqrt(n_i^{(d)} n_j^{(d)}) * a^{(d)}_{ij,k}(m) * b^{(d)}_{ij,k}(n),

    a^{(d)}_{ij,k}(m) = sin(omega^L_{ij,k,d} * pi * x_m),
    b^{(d)}_{ij,k}(n) = sin(omega^R_{ij,k,d} * pi * y_n),

    sigma_k = coupling_strength * 0.3^k,

  where x_m = (m+1)/(n_i^{(d)}+1), y_n = (n+1)/(n_j^{(d)}+1), and the left/right
  frequencies depend on the source block, target block, coupling term, and
  dimension. Each M^{(d)}_{ij,k} is rank-1 as a matrix, each k-term is TT-rank 1
  across dimensions, and the full k-term has overall size O(sigma_k), not
  O(sigma_k^dimension), because the coupling strength is carried only by sigma_k.
  The summed off-diagonal block has TT rank at most q. Since the factors are built
  with separate row and column sizes, the coupling can map between blocks with
  different mode sizes.

  The right-hand side is manufactured from a known exact solution. For each block,
  the exact solution is a sum of separable sinusoidal terms,

    u_exact = sum_{k=0}^{r-1} alpha_k prod_d sin(omega_{k,d} * pi * x_d + phi_{k,d}),

  with block-, term-, and dimension-dependent amplitudes, frequencies, and phase
  shifts. The input parameter exact_solution_rank controls the number of
  separable terms in that sum. The test forms f = A * u_exact, solves the block
  system in TT form, and checks both the explicit residual and the relative error
  against the manufactured solution.
*/

template <size_t dimension>
boba::Tensor<dimension, space, double> make_exact_block_solution(
  const boba::Array<size_t, dimension>& sizes,
  size_t block_id,
  size_t target_rank)
{
  boba_always_assert(target_rank > 0, "exact solution rank must be positive");

  boba::Tensor<dimension, space, double> exact_tensor(sizes);
  auto exact_view = exact_tensor.view();

  ::boba::loop<space, 1>(exact_tensor.size(),
                         [=] __boba_host_device__(size_t i)
  {
    auto midx = exact_view.multiindex(i);
    double value = 0.0;
    for (size_t term = 0; term < target_rank; term++)
    {
      auto term_value = boba::pow(0.2, static_cast<double>(term));
      for (size_t dim = 0; dim < dimension; dim++)
      {
        auto x = static_cast<double>(midx[dim] + 1) / static_cast<double>(sizes[dim] + 1);
        auto frequency = static_cast<double>(
          1 + ((3 * term + 2 * dim + 5 * block_id + term * dim) % (sizes[dim] + 1)));
        auto phase_shift = 0.17 * static_cast<double>((block_id + 1) * (term + 1) * (dim + 2));
        auto phase = frequency * boba::pi * x + phase_shift;
        auto amplitude = 1.0 + static_cast<double>((block_id + 1) * (dim + 1)) / (8.0 * static_cast<double>(term + 1));
        term_value *= amplitude * boba::sin(phase);
      }
      value += term_value;
    }
    exact_view(i) = value;
  });

  return exact_tensor;
}

struct input
{
  input()
  {
    boba::TensorTrainAMENBlock<double> get_solver_defaults;
    convergence_tolerance = get_solver_defaults.convergence_tolerance;
    max_direct_solve_size = get_solver_defaults.max_direct_solve_size;
    solve3d_2ml_method = get_solver_defaults.solve3d_2ml_options.method;
    solve3d_2ml_tolerance_relative = get_solver_defaults.solve3d_2ml_options.tolerance_relative;
  }

  size_t N = 4;
  double convergence_tolerance = 1.0e-4;
  int max_direct_solve_size = 1500;
  size_t solve3d_2ml_method = 0;
  double solve3d_2ml_tolerance_relative = 1.0e-3;
  double coupling_strength = 1.0e-2;
  size_t coupling_rank = 1;
  size_t exact_solution_rank = 3;
  bool verbose = false;
  double dx = 1.0;
};

//
// Test for Block TT-AMEn with variable-size blocks
//

template <size_t dimension, size_t n_blocks>
void run_block_test(
  bool& check,
  input parameters,
  double coupling_strength)
{
  using tt_t = boba::TensorTrain<dimension, space, double>;
  using ttm_t = boba::TensorTrainMatrix<dimension, space, double>;
  using tensor_t = boba::Tensor<dimension, space, double>;
  using block_tt_t = boba::BlockVector<tt_t>;
  using block_tensor_t = boba::BlockVector<tensor_t>;
  using block_ttm_t = boba::BlockOperator<ttm_t>;

  std::cout << "Testing " << n_blocks << " blocks in " << dimension << "D"
            << " with coupling_strength = " << coupling_strength
            << ", coupling_rank = " << parameters.coupling_rank
            << " and exact_solution_rank = " << parameters.exact_solution_rank << std::endl;

  // Create blocks with different sizes
  boba::Array<boba::Array<size_t, dimension>, n_blocks> block_sizes;
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    // Each block has different size: N + blk
    for (size_t d = 0; d < dimension; d++)
    {
      block_sizes[blk][d] = parameters.N + blk + d;
    }
    std::cout << "Block " << blk << " sizes: ";
    for (size_t d = 0; d < dimension; d++)
    {
      std::cout << block_sizes[blk][d] << " ";
    }
    std::cout << std::endl;
  }

  // Create block operator with variable-size diagonal Poisson blocks.
  block_ttm_t A(n_blocks, n_blocks);

  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    auto sizes = block_sizes[blk];

    // Create Poisson operator for this block
    ttm_t A_blk(sizes, sizes);
    A_blk.fill_with_zeros();

    for (size_t dim = 0; dim < dimension; dim++)
    {
      boba::Array<boba::Matrix<space, double>, dimension> array_of_matrices;
      for (size_t d = 0; d < dimension; d++)
      {
        auto N = sizes[d];
        auto dx = 1.0 / double(N + 1);
        boba::Matrix<host_space, double> L({N, N});
        boba::Matrix<host_space, double> I({N, N});

        // Special handling of very small sizes to avoid issues with the standard 3-point stencil.
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

        boba::Matrix<space, double> L_device = L;
        boba::Matrix<space, double> I_device = I;

        array_of_matrices[d] = (dim == d) ? L_device : I_device;
      }
      A_blk += boba::make_ttm_from_matrices(array_of_matrices);
    }

    A({blk, blk}) = A_blk;
  }

  if (coupling_strength > 0.0)
  {
    for (size_t row = 0; row < n_blocks; row++)
    {
      for (size_t col = 0; col < n_blocks; col++)
      {
        if (row == col)
        {
          continue;
        }

        ttm_t A_coupling;
        bool initialized = false;

        for (size_t term = 0; term < parameters.coupling_rank; term++)
        {
          boba::Array<boba::Matrix<space, double>, dimension> array_of_matrices;
          auto term_weight = boba::pow(0.3, static_cast<double>(term));

          for (size_t d = 0; d < dimension; d++)
          {
            auto rows = block_sizes[row][d];
            auto cols = block_sizes[col][d];
            boba::Matrix<host_space, double> coupling({rows, cols});

            auto left_frequency = static_cast<double>(term + 1 + ((row + d) % 2));
            auto right_frequency = static_cast<double>(term + 1 + ((col + d + 1) % 2));
            for (size_t i = 0; i < rows; i++)
            {
              auto x = static_cast<double>(i + 1) / static_cast<double>(rows + 1);
              auto left_value = boba::sin(left_frequency * boba::pi * x);
              for (size_t j = 0; j < cols; j++)
              {
                auto y = static_cast<double>(j + 1) / static_cast<double>(cols + 1);
                auto right_value = boba::sin(right_frequency * boba::pi * y);
                coupling({i, j}) = left_value * right_value / static_cast<double>(boba::sqrt(rows * cols));
              }
            }
            array_of_matrices[d] = boba::Matrix<space, double>(coupling);
          }

          auto A_term = boba::make_ttm_from_matrices(array_of_matrices);
          A_term *= term_weight;
          if (!initialized)
          {
            A_coupling = A_term;
            initialized = true;
          }
          else
          {
            A_coupling += A_term;
          }
        }

        auto diagonal_scale = boba::sqrt(
          ::boba::norm_frobenius(A({row, row})) *
          ::boba::norm_frobenius(A({col, col})));
        auto coupling_norm = ::boba::norm_frobenius(A_coupling);
        if (coupling_norm > 0.0)
        {
          A_coupling *= coupling_strength * diagonal_scale / coupling_norm;
        }

        A({row, col}) = A_coupling;
      }
    }
  }

  checkpoint();

  // Create exact manufactured solution
  block_tensor_t exact_solution(n_blocks);
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    exact_solution(blk) = make_exact_block_solution<dimension>(
      block_sizes[blk],
      blk,
      parameters.exact_solution_rank);
  }

  block_tt_t exact_solution_tt(n_blocks);
  boba::BlockVector<boba::TensorTrain<dimension, host_space, double>> exact_solution_tt_host(n_blocks);
  double max_exact_solution_compression_error = 0.0;
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    exact_solution_tt(blk) = boba::compress_to_TensorTrain(exact_solution(blk), 1.0e-14, 0.0);
    boba::Tensor<dimension, host_space, double> exact_solution_host = exact_solution(blk);
    exact_solution_tt_host(blk) = boba::compress_to_TensorTrain(exact_solution_host, 1.0e-14, 0.0);
    auto exact_solution_norm = ::boba::norm_frobenius(exact_solution_tt_host(blk));
    auto exact_solution_compression_error =
      boba::norm_difference_frobenius(exact_solution_tt(blk), exact_solution_tt_host(blk));
    if (exact_solution_norm > 1.0e-14)
    {
      exact_solution_compression_error /= exact_solution_norm;
    }
    max_exact_solution_compression_error =
      boba::max(max_exact_solution_compression_error, exact_solution_compression_error);
  }

  // Create RHS from the manufactured exact solution
  block_tensor_t rhs_tensor_dense(n_blocks);
  block_tt_t rhs(n_blocks);
  double max_rhs_compression_error = 0.0;
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    tensor_t rhs_tensor(block_sizes[blk]);
    rhs_tensor.fill_with_zeros();
    for (size_t src = 0; src < n_blocks; src++)
    {
      if (A({blk, src}).get_number_elements() == 0)
      {
        continue;
      }
      rhs_tensor += A({blk, src}) * exact_solution(src);
    }
    rhs_tensor_dense(blk) = rhs_tensor;
    rhs(blk) = boba::compress_to_TensorTrain(rhs_tensor, 1.0e-14, 0.0);
    tensor_t rhs_compressed_tensor = rhs(blk).decompress();
    auto rhs_tensor_norm = ::boba::norm_frobenius(rhs_tensor);
    auto rhs_compression_error =
      ::boba::norm_frobenius(rhs_compressed_tensor - rhs_tensor);
    if (rhs_tensor_norm > 1.0e-14)
    {
      rhs_compression_error /= rhs_tensor_norm;
    }
    max_rhs_compression_error =
      boba::max(max_rhs_compression_error, rhs_compression_error);
  }
  std::cout << "Max relative compression error in exact solution blocks: "
            << max_exact_solution_compression_error << std::endl;
  std::cout << "Max relative compression error in RHS blocks: "
            << max_rhs_compression_error << std::endl;

  // Create initial guess
  block_tt_t initial_guess(n_blocks);
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    initial_guess(blk).resize(block_sizes[blk]);
    initial_guess(blk).fill_with(1.0);
  }

  constexpr double tiny_norm = 1.0e-14;
  block_tt_t reference_residual(n_blocks);
  block_tt_t compressed_exact_matvec(n_blocks);
  double max_operator_application_error_from_compressed_exact = 0.0;
  double dense_residual_from_compressed_terms_norm_squared = 0.0;
  auto block_rhs_norm = ::boba::norm_frobenius(rhs);
  for (size_t row = 0; row < n_blocks; row++)
  {
    bool initialized = false;
    tt_t compressed_exact_matvec_row;
    for (size_t src = 0; src < n_blocks; src++)
    {
      if (A({row, src}).get_number_elements() == 0)
      {
        continue;
      }
      if (initialized)
      {
        compressed_exact_matvec_row += A({row, src}) * exact_solution_tt(src);
      }
      else
      {
        compressed_exact_matvec_row = A({row, src}) * exact_solution_tt(src);
        initialized = true;
      }
    }
    boba_always_assert(initialized, "Reference residual requires at least one nonempty block per row.");
    tensor_t compressed_exact_matvec_tensor = compressed_exact_matvec_row.decompress();
    auto dense_rhs_norm = ::boba::norm_frobenius(rhs_tensor_dense(row));
    auto operator_application_error =
      ::boba::norm_frobenius(compressed_exact_matvec_tensor - rhs_tensor_dense(row));
    if (dense_rhs_norm > tiny_norm)
    {
      operator_application_error /= dense_rhs_norm;
    }
    max_operator_application_error_from_compressed_exact =
      boba::max(max_operator_application_error_from_compressed_exact, operator_application_error);
    compressed_exact_matvec(row) = compressed_exact_matvec_row;
    tensor_t compressed_rhs_tensor = rhs(row).decompress();
    auto dense_residual_from_compressed_terms =
      compressed_exact_matvec_tensor - compressed_rhs_tensor;
    const auto dense_residual_from_compressed_terms_norm =
      ::boba::norm_frobenius(dense_residual_from_compressed_terms);
    dense_residual_from_compressed_terms_norm_squared +=
      dense_residual_from_compressed_terms_norm * dense_residual_from_compressed_terms_norm;
    reference_residual(row) = compressed_exact_matvec_row;
    reference_residual(row) -= rhs(row);
  }
  std::cout << "Max relative error in A*compressed(exact solution) vs dense A*exact solution: "
            << max_operator_application_error_from_compressed_exact << std::endl;
  auto dense_residual_from_compressed_terms_relative =
    (block_rhs_norm > tiny_norm)
      ? (::boba::sqrt(dense_residual_from_compressed_terms_norm_squared) / block_rhs_norm)
      : ::boba::sqrt(dense_residual_from_compressed_terms_norm_squared);
  std::cout << "Dense relative residual from decompressed A*compressed(exact solution) minus decompressed compressed RHS: "
            << dense_residual_from_compressed_terms_relative << std::endl;
  auto reference_residual_norm = ::boba::norm_frobenius(reference_residual);
  auto reference_relative_residual = (block_rhs_norm > tiny_norm) ? (reference_residual_norm / block_rhs_norm) : reference_residual_norm;
  std::cout << "Pre-solve compressed reference relative residual: "
            << reference_relative_residual << std::endl;
  auto rounded_reference_residual = reference_residual;
  rounded_reference_residual.round();
  auto rounded_reference_residual_norm = ::boba::norm_frobenius(rounded_reference_residual);
  auto rounded_reference_relative_residual =
    (block_rhs_norm > tiny_norm) ? (rounded_reference_residual_norm / block_rhs_norm) : rounded_reference_residual_norm;
  std::cout << "Pre-solve rounded compressed reference relative residual: "
            << rounded_reference_relative_residual << std::endl;

  checkpoint();

  // Solve
  boba::TensorTrainAMENBlock<double> tt_amen_block;
  tt_amen_block.max_direct_solve_size = parameters.max_direct_solve_size;
  tt_amen_block.convergence_tolerance = parameters.convergence_tolerance;
  tt_amen_block.solve3d_2ml_options.method = parameters.solve3d_2ml_method;
  tt_amen_block.solve3d_2ml_options.tolerance_relative = parameters.solve3d_2ml_tolerance_relative;
  tt_amen_block.verbose = parameters.verbose;

  boba::TicToc<tictoc_units> timer;
  timer.tic();
  auto solution = tt_amen_block.solve(A, rhs, initial_guess);
  timer.end_and_print("Block TT AMEn solver time:");

  // Compute residual for each block
  double max_residual = 0.0;
  double max_solution_error = 0.0;
  bool saw_rank_growth = false;
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    auto residual_vec = rhs(blk);
    residual_vec *= -1.0;
    for (size_t src = 0; src < n_blocks; src++)
    {
      if (A({blk, src}).get_number_elements() == 0)
      {
        continue;
      }
      residual_vec += A({blk, src}) * solution(src);
    }
    residual_vec.round();
    auto residual_norm = ::boba::norm_frobenius(residual_vec);
    auto rhs_norm = ::boba::norm_frobenius(rhs(blk));
    auto exact_norm = ::boba::norm_frobenius(exact_solution_tt(blk));
    auto solution_error_norm = boba::norm_difference_frobenius(solution(blk), exact_solution_tt_host(blk));
    auto relative_residual = (rhs_norm > tiny_norm) ? (residual_norm / rhs_norm) : residual_norm;
    auto relative_solution_error = (exact_norm > tiny_norm) ? (solution_error_norm / exact_norm) : solution_error_norm;
    if (!std::isfinite(relative_residual))
    {
      relative_residual = 0.0;
    }
    if (!std::isfinite(relative_solution_error))
    {
      relative_solution_error = 0.0;
    }

    std::cout << "Block " << blk << " relative residual: " << relative_residual << std::endl;
    std::cout << "Block " << blk << " relative solution error: " << relative_solution_error << std::endl;
    std::cout << "Block " << blk << " solution ranks: " << solution(blk).ranks_string() << std::endl;

    max_residual = boba::max(max_residual, relative_residual);
    max_solution_error = boba::max(max_solution_error, relative_solution_error);

    for (size_t d = 1; d < dimension; d++)
    {
      saw_rank_growth = saw_rank_growth || (solution(blk).ranks(d) > 1);
    }
  }

  std::cout << "Max relative residual across all blocks: " << max_residual << std::endl;
  std::cout << "Max relative solution error across all blocks: " << max_solution_error << std::endl;

  // Check convergence
  double residual_tolerance = boba::max(1000.0 * parameters.convergence_tolerance, 1.0e-8);
  double solution_tolerance = boba::max(100.0 * parameters.convergence_tolerance, 1.0e-10);
  pass_or_fail(check, max_residual, residual_tolerance);
  pass_or_fail(check, max_solution_error, solution_tolerance);
  if (parameters.exact_solution_rank > 1)
  {
    pass_or_fail_bool(check, saw_rank_growth);
  }
}

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  boba_print("Test for Block TT-AMEn with variable-size blocks");

  bool check = true;

  size_t dimension = 0;
  input parameters;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(
    parameters.N,
    "-N",
    "--resolution",
    "Base resolution for blocks.");

  args.add_optional_argument(
    parameters.convergence_tolerance,
    "-T",
    "--convergence_tolerance",
    "AMEn Convergence tolerance.");

  args.add_optional_argument(
    dimension,
    "-d",
    "--dimension",
    "Which problem dimension to run (0 runs all supported cases: 2, 3, and 4).");

  args.add_optional_argument(
    parameters.max_direct_solve_size,
    "-M",
    "--max_direct_solve_size",
    "Maximum size before switching to iterative solver.");

  args.add_optional_argument(
    parameters.solve3d_2ml_method,
    "-m",
    "--solve3d_2ml_method",
    "Iterative method (0=Eigen GMRES, 1=Boba GMRES matrix, 2=Boba GMRES operator).");

  args.add_optional_argument(
    parameters.solve3d_2ml_tolerance_relative,
    "-t",
    "--solve3d_2ml_tolerance_relative",
    "Iterative method relative tolerance.");

  args.add_optional_argument(
    parameters.coupling_strength,
    "-c",
    "--coupling_strength",
    "Off-diagonal block coupling strength.");

  args.add_optional_argument(
    parameters.coupling_rank,
    "-q",
    "--coupling_rank",
    "TT rank target for the manufactured off-diagonal block couplings.");

  args.add_optional_argument(
    parameters.exact_solution_rank,
    "-r",
    "--exact_solution_rank",
    "Target rank for the manufactured exact block solution.");

  args.add_optional_argument(
    parameters.verbose,
    "-v",
    "--verbose",
    "Print per-sweep Block TT-AMEn residual history.");

  args.parse_check();

  boba_always_assert(
    (dimension == 0) || (dimension == 2) || (dimension == 3) || (dimension == 4),
    "test_amen_block currently supports only dimensions 2, 3, and 4, or 0 for all.");

  if ((dimension == 0) || (dimension == 2))
  {
    boba_print("Running 2D test with 2 blocks");
    run_block_test<2, 2>(check, parameters, 0.0);
    if (parameters.coupling_strength > 0.0)
    {
      boba_print("Running 2D coupled test with 2 blocks");
      run_block_test<2, 2>(check, parameters, parameters.coupling_strength);
    }

    boba_print("Running 2D test with 3 blocks");
    run_block_test<2, 3>(check, parameters, 0.0);
    if (parameters.coupling_strength > 0.0)
    {
      boba_print("Running 2D coupled test with 3 blocks");
      run_block_test<2, 3>(check, parameters, parameters.coupling_strength);
    }
  }
  if ((dimension == 0) || (dimension == 3))
  {
    boba_print("Running 3D test with 2 blocks");
    run_block_test<3, 2>(check, parameters, 0.0);
    if (parameters.coupling_strength > 0.0)
    {
      boba_print("Running 3D coupled test with 2 blocks");
      run_block_test<3, 2>(check, parameters, parameters.coupling_strength);
    }
  }
  if ((dimension == 0) || (dimension == 4))
  {
    boba_print("Running 4D test with 2 blocks");
    run_block_test<4, 2>(check, parameters, 0.0);
    if (parameters.coupling_strength > 0.0)
    {
      boba_print("Running 4D coupled test with 2 blocks");
      run_block_test<4, 2>(check, parameters, parameters.coupling_strength);
    }
  }

  boba::finalize();
  return final_check(check);
}
