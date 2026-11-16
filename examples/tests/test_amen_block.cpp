// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include <random>
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
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    exact_solution_tt(blk) = boba::compress_to_TensorTrain(exact_solution(blk), 1.0e-14, 0.0);
    boba::Tensor<dimension, host_space, double> exact_solution_host = exact_solution(blk);
    exact_solution_tt_host(blk) = boba::compress_to_TensorTrain(exact_solution_host, 1.0e-14, 0.0);
  }

  // Create RHS from the manufactured exact solution
  block_tt_t rhs(n_blocks);
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
    rhs(blk) = boba::compress_to_TensorTrain(rhs_tensor, 1.0e-14, 0.0);
  }

  // Create initial guess
  block_tt_t initial_guess(n_blocks);
  for (size_t blk = 0; blk < n_blocks; blk++)
  {
    initial_guess(blk).resize(block_sizes[blk]);
    initial_guess(blk).fill_with(1.0);
  }

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
  constexpr double tiny_norm = 1.0e-14;
  double residual_tolerance = boba::max(1000.0 * parameters.convergence_tolerance, 1.0e-8);
  double solution_tolerance = boba::max(100.0 * parameters.convergence_tolerance, 1.0e-10);
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

    std::cout << "Block " << blk << " relative residual: " << relative_residual << std::endl;
    std::cout << "Block " << blk << " relative solution error: " << relative_solution_error << std::endl;
    std::cout << "Block " << blk << " solution ranks: " << solution(blk).ranks_string() << std::endl;

    pass_or_fail(check, relative_residual, residual_tolerance);
    pass_or_fail(check, relative_solution_error, solution_tolerance);

    max_residual = boba::max(max_residual, relative_residual);
    max_solution_error = boba::max(max_solution_error, relative_solution_error);

    for (size_t d = 1; d < dimension; d++)
    {
      saw_rank_growth = saw_rank_growth || (solution(blk).ranks(d) > 1);
    }
  }

  std::cout << "Max relative residual across all blocks: " << max_residual << std::endl;
  std::cout << "Max relative solution error across all blocks: " << max_solution_error << std::endl;

  if (parameters.exact_solution_rank > 1)
  {
    pass_or_fail_bool(check, saw_rank_growth);
  }

  // Packed operator (packed_operator): the same coupled system, matrix-free GMRES site solves, once with the dense
  // cores and once with packed cores and rank-one zero placeholder cores; the solutions must agree.
  if constexpr (space == boba::execution_space::CPU)
  {
    if (coupling_strength > 0.0)
    {
      std::vector<std::vector<std::vector<boba::CoreSliceSparsity<double>>>> packed(
        n_blocks, std::vector<std::vector<boba::CoreSliceSparsity<double>>>(n_blocks));
      block_ttm_t placeholder(n_blocks, n_blocks);
      for (size_t r = 0; r < n_blocks; r++)
        for (size_t c = 0; c < n_blocks; c++)
        {
          placeholder({r, c}) = A({r, c});
          if (A({r, c}).get_number_elements() == 0) continue;
          for (size_t d = 0; d < dimension; d++)
          {
            packed[r][c].push_back(boba::core_slice_sparsity(A({r, c}).cores[d], 1.0));
            const auto n = A({r, c}).cores[d].sizes(1), m = A({r, c}).cores[d].sizes(2);
            placeholder({r, c}).cores[d].resize({1, n, m, 1});
            placeholder({r, c}).cores[d].fill_with_zeros();
          }
        }
      auto configured = [&]() {
        boba::TensorTrainAMENBlock<double> s;
        s.max_direct_solve_size = parameters.max_direct_solve_size;
        s.convergence_tolerance = parameters.convergence_tolerance;
        s.solve3d_2ml_options.method = parameters.solve3d_2ml_method;
        s.solve3d_2ml_options.tolerance_relative = 1e-12;
        s.use_boba_gmres_for_coupled_site_solve = true;
        s.site_jacobi_preconditioner = true;
        s.scale_coupled_site_system = false;
        s.verbose = false;
        return s;
      };
      auto denseSolver = configured();
      auto packedSolver = configured();
      packedSolver.packed_operator = &packed;
      const auto denseSolution = denseSolver.solve(A, rhs, initial_guess);
      const auto packedSolution = packedSolver.solve(placeholder, rhs, initial_guess);
      for (size_t blk = 0; blk < n_blocks; blk++)
      {
        const double ref = boba::norm_frobenius(denseSolution(blk));
        double packed_vs_dense = boba::norm_difference_frobenius(packedSolution(blk), denseSolution(blk)) / (ref > 0 ? ref : 1.0);
        double packed_error = boba::norm_difference_frobenius(packedSolution(blk), exact_solution_tt_host(blk)) /
                              boba::max(boba::norm_frobenius(exact_solution_tt_host(blk)), 1e-14);
        std::cout << "Block " << blk << " packed vs dense: " << packed_vs_dense << ", packed error " << packed_error << std::endl;
        pass_or_fail(check, packed_vs_dense, 1e-8);
        pass_or_fail(check, packed_error, solution_tolerance);
      }
    }
  }
}

/// Packed (nonzero-slice) operator cores against the dense kernels: compute_next_Phi in both sweep directions,
/// bfun3, and the transpose.  The cores are cell-local, as the explicit Schur complement of a DPG system: block
/// diagonal (a field block) and block bidiagonal (a field-trace block in the trace's normal direction).
void run_packed_core_test(bool& check)
{
  using T3 = boba::Tensor<3, boba::execution_space::CPU, double>;
  using T4 = boba::Tensor<4, boba::execution_space::CPU, double>;
  std::mt19937 gen(4242);
  std::normal_distribution<double> nd;
  auto randomT3 = [&](boba::index_t a, boba::index_t b, boba::index_t c) {
    T3 t({a, b, c}); for (boba::index_t k = 0; k < a * b * c; ++k) t.data()[k] = nd(gen); return t; };
  const boba::index_t cells = 5, na = 3, nbF = 3, ra1 = 4, ra2 = 6, rx1 = 5, rx2 = 3, ry1 = 4, ry2 = 2;
  for (int pattern = 0; pattern < 2; ++pattern)
  {
    // pattern 0: rows and columns (cell, basis), block diagonal; pattern 1: columns are the interior faces
    const boba::index_t N = cells * na, M = pattern == 0 ? cells * nbF : cells - 1;
    T4 A({ra1, N, M, ra2}); A.fill_with_zeros();
    for (boba::index_t c = 0; c < cells; ++c)
      for (boba::index_t i = 0; i < na; ++i)
      {
        std::vector<boba::index_t> cols;
        if (pattern == 0) for (boba::index_t j = 0; j < nbF; ++j) cols.push_back(c * nbF + j);
        else { if (c >= 1) cols.push_back(c - 1); if (c + 1 < cells) cols.push_back(c); }
        for (boba::index_t m : cols)
          for (boba::index_t r2 = 0; r2 < ra2; ++r2) for (boba::index_t r1 = 0; r1 < ra1; ++r1)
            A.data()[r1 + ra1 * ((c * na + i) + N * (m + M * r2))] = nd(gen);
      }
    const auto packed = boba::core_slice_sparsity(A, 1.0);
    const double denseNorm = boba::norm_frobenius(A);
    double densifyError = boba::norm_frobenius(boba::densify_core_slices(packed) - A) / denseNorm;
    pass_or_fail(check, densifyError, 1e-15);
    for (bool leftRight : {true, false})
    {
      const T3 x = randomT3(rx1, N, rx2), y = randomT3(ry1, M, ry2);
      const T3 Phi = leftRight ? randomT3(rx1, ry1, ra1) : randomT3(ry2, ra2, rx2);
      for (double externalNorm : {0.0, 3.0})
      {
        auto dense = boba::compute_next_Phi<4>(Phi, x, A, y, leftRight, externalNorm);
        auto sparse = boba::compute_next_Phi_packed<4>(Phi, x, packed, y, leftRight, externalNorm);
        double phiError = boba::norm_frobenius(dense.first - sparse.first) / boba::norm_frobenius(dense.first);
        double normError = std::abs(dense.second - sparse.second) / std::max(1.0, std::abs(dense.second));
        pass_or_fail(check, phiError, 1e-13);
        pass_or_fail(check, normError, 1e-13);
      }
    }
    // bfun3: Phi1(ry1, rx1, ra1), Phi2(rx2, ra2, ry2), x(rx1, m, rx2) -> (ry1, n, ry2)
    {
      const T3 Phi1 = randomT3(ry1, rx1, ra1), Phi2 = randomT3(rx2, ra2, ry2), x = randomT3(rx1, M, rx2);
      auto dense = boba::bfun3(Phi1, A, Phi2, x);
      auto sparse = boba::bfun3_sparse(Phi1, packed, Phi2, x);
      double bfunError = boba::norm_frobenius(dense - sparse) / boba::norm_frobenius(dense);
      pass_or_fail(check, bfunError, 1e-13);
    }
    // transpose: densify(transpose(packed)) == A with rows and columns swapped
    {
      const auto packedT = boba::transpose_core_slices(packed);
      const auto D = boba::densify_core_slices(packedT);
      double transposeError = 0;
      for (boba::index_t r2 = 0; r2 < ra2; ++r2) for (boba::index_t m = 0; m < M; ++m) for (boba::index_t n = 0; n < N; ++n)
        for (boba::index_t r1 = 0; r1 < ra1; ++r1)
          transposeError = std::max(transposeError, std::abs(D.data()[r1 + ra1 * (m + M * (n + N * r2))] -
                                                             A.data()[r1 + ra1 * (n + N * (m + M * r2))]));
      pass_or_fail(check, transposeError, 1e-15);
      bool columnMajor = true;
      for (size_t q = 1; q < packedT.pairs.size(); ++q)
        if (packedT.pairs[q].second < packedT.pairs[q - 1].second ||
            (packedT.pairs[q].second == packedT.pairs[q - 1].second && packedT.pairs[q].first <= packedT.pairs[q - 1].first))
          columnMajor = false;
      pass_or_fail_bool(check, columnMajor);
    }
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

  boba_print("Running packed operator-core kernel test");
  run_packed_core_test(check);

  boba::finalize();
  return final_check(check);
}
