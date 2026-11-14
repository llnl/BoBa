#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <array>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <map>
#include <tuple>

namespace boba
{

/// Nonzero (row, column) slices of a TT-matrix core A(ra1, n, m, ra2), packed: for each nonzero slice the ra1 x ra2
/// matrix A(:, n, m, :).  Operator cores that are (nearly) block diagonal in a cell index -- e.g. element-local
/// operators in a (cell x basis) mode -- have few nonzero slices, and the site operator can skip the zero ones exactly.
template <typename data_t>
struct CoreSliceSparsity
{
  index_t ra1 = 0, n = 0, m = 0, ra2 = 0;
  std::vector<std::pair<index_t, index_t>> pairs;   // (n, m) of the nonzero slices
  std::vector<data_t> blocks;                        // ra1 x ra2 per pair, column-major, in the order of pairs
};

/// AxPhi2(ra1, n, rx1, ry2) = sum_{m, ra2} A(ra1, n, m, ra2) xPhi2(rx1, m, ra2, ry2) over the nonzero slices of A only
/// (host tensors; first index fastest).  Same result as the dense contraction in bfun3.
template <execution_space space, typename data_t>
boba::Tensor<3, space, data_t> bfun3_sparse(
  const boba::Tensor<3, space, data_t>& Phi1,
  const CoreSliceSparsity<data_t>& A,
  const boba::Tensor<3, space, data_t>& Phi2,
  const boba::Tensor<3, space, data_t>& x)
{
  using MatX = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  auto xPhi2 = boba::tensor_contraction<1>(
    {"rx1", "m", "rx2"}, x, {"rx2", "ra2", "ry2"}, Phi2, {"rx1", "m", "ra2", "ry2"});
  const index_t X1 = xPhi2.sizes(0), M = xPhi2.sizes(1), R2 = xPhi2.sizes(2), Y2 = xPhi2.sizes(3);
  boba_always_assert(M == A.m && R2 == A.ra2, "bfun3_sparse: size mismatch");
  // Y_m(ra2, rx1 + X1 ry2) = xPhi2(rx1, m, ra2, ry2), packed contiguously per m
  std::vector<MatX> Y(static_cast<size_t>(M));
  {
    const data_t* xp = xPhi2.data();
    for (index_t m = 0; m < M; ++m)
    {
      MatX Ym(R2, X1 * Y2);
      for (index_t ry2 = 0; ry2 < Y2; ++ry2)
        for (index_t ra2 = 0; ra2 < R2; ++ra2)
          for (index_t rx1 = 0; rx1 < X1; ++rx1)
            Ym(ra2, rx1 + X1 * ry2) = xp[rx1 + X1 * (m + M * (ra2 + R2 * ry2))];
      Y[static_cast<size_t>(m)] = std::move(Ym);
    }
  }
  boba::Tensor<4, space, data_t> AxPhi2({A.ra1, A.n, X1, Y2});
  AxPhi2.fill_with_zeros();
  data_t* out = AxPhi2.data();
  const index_t R1 = A.ra1, N = A.n;
  for (size_t q = 0; q < A.pairs.size(); ++q)
  {
    const index_t n = A.pairs[q].first, m = A.pairs[q].second;
    Eigen::Map<const MatX> Anm(A.blocks.data() + q * static_cast<size_t>(R1 * R2), R1, R2);
    Eigen::Map<MatX, 0, Eigen::OuterStride<>> On(out + R1 * n, R1, X1 * Y2, Eigen::OuterStride<>(R1 * N));
    On.noalias() += Anm * Y[static_cast<size_t>(m)];
  }
  return boba::tensor_contraction<2>(
    {"ry1", "rx1", "ra1"}, Phi1, {"ra1", "n", "rx1", "ry2"}, AxPhi2, {"ry1", "n", "ry2"});
}

/// The nonzero-slice pattern of a host core, or an empty pattern if more than maxFraction of the slices are nonzero.
template <typename data_t>
CoreSliceSparsity<data_t> core_slice_sparsity(const boba::Tensor<4, ::boba::execution_space::CPU, data_t>& A,
                                              double maxFraction)
{
  CoreSliceSparsity<data_t> S;
  S.ra1 = A.sizes(0); S.n = A.sizes(1); S.m = A.sizes(2); S.ra2 = A.sizes(3);
  const data_t* a = A.data();
  const index_t R1 = S.ra1, N = S.n, M = S.m, R2 = S.ra2;
  for (index_t m = 0; m < M; ++m)
    for (index_t n = 0; n < N; ++n)
    {
      bool nonzero = false;
      for (index_t ra2 = 0; ra2 < R2 && !nonzero; ++ra2)
        for (index_t ra1 = 0; ra1 < R1; ++ra1)
          if (a[ra1 + R1 * (n + N * (m + M * ra2))] != data_t{0}) { nonzero = true; break; }
      if (nonzero) S.pairs.push_back({n, m});
    }
  if (double(S.pairs.size()) > maxFraction * double(N * M)) { S.pairs.clear(); return S; }
  S.blocks.resize(S.pairs.size() * static_cast<size_t>(R1 * R2));
  for (size_t q = 0; q < S.pairs.size(); ++q)
  {
    const index_t n = S.pairs[q].first, m = S.pairs[q].second;
    for (index_t ra2 = 0; ra2 < R2; ++ra2)
      for (index_t ra1 = 0; ra1 < R1; ++ra1)
        S.blocks[q * static_cast<size_t>(R1 * R2) + ra1 + R1 * ra2] = a[ra1 + R1 * (n + N * (m + M * ra2))];
  }
  return S;
}

/**
 * Matrix-free action of a coupled block-AMEn site operator.
 *
 * The input and output are concatenated dense vectors, but each nonzero block
 * is applied through bfun3 (or, for cores with few nonzero slices, bfun3_sparse) rather than materializing its
 * projected matrix.
 */
template <execution_space space, typename data_t>
struct CoupledSolve3DGMRESOperator
{
  using tensor3_t = ::boba::Tensor<3, space, data_t>;
  using tensor4_t = ::boba::Tensor<4, space, data_t>;

  std::vector<std::vector<const tensor3_t*>> phi_left;
  std::vector<std::vector<const tensor4_t*>> operator_cores;
  std::vector<std::vector<const CoreSliceSparsity<data_t>*>> sparse_cores;   // optional; nullptr or empty: dense
  std::vector<std::vector<const tensor3_t*>> phi_right;
  std::vector<::boba::Array<index_t, 3>> input_sizes;
  std::vector<index_t> offsets;
  std::vector<data_t> row_factors;
  std::vector<::boba::Vector<space, data_t>> scalar_row_factors;
  /// Diagonal mode: when set and no blocks are given, the operator is diag(diagonal) (a Jacobi preconditioner).
  ::boba::Vector<space, data_t> diagonal;

  [[nodiscard]]
  ::boba::Vector<space, data_t> operator*(
    const ::boba::Vector<space, data_t>& input) const
  {
    BOBA_CALI_MARK
    if (operator_cores.empty() && diagonal.size() > 0)
    {
      ::boba::Vector<space, data_t> output({static_cast<index_t>(input.size())});
      auto out_view = output.view();
      auto in_view = input.const_view();
      auto d_view = diagonal.const_view();
      ::boba::loop<space, 1>(static_cast<index_t>(input.size()), [=] __boba_host_device__(index_t idx)
      {
        out_view({idx}) = d_view({idx}) * in_view({idx});
      });
      return output;
    }
    boba_always_assert_equal(offsets.size(), input_sizes.size() + 1,
                             "Coupled operator offsets are inconsistent.");
    boba_always_assert_equal(input.size(), static_cast<size_t>(offsets.back()),
                             "Coupled operator input has the wrong size.");

    const auto block_count = input_sizes.size();
    std::vector<tensor3_t> input_blocks(block_count);
    for (size_t col = 0; col < block_count; col++)
    {
      const auto block_size = offsets[col + 1] - offsets[col];
      ::boba::Vector<space, data_t> input_block({block_size});
      auto input_block_view = input_block.view();
      auto input_view = input.const_view();
      const auto offset = offsets[col];
      ::boba::loop<space, 1>(block_size, [=] __boba_host_device__(index_t idx)
      {
        input_block_view({idx}) = input_view({offset + idx});
      });
      input_blocks[col] = reshape<3>(input_block, input_sizes[col]);
    }

    ::boba::Vector<space, data_t> output({offsets.back()});
    output.fill_with_zeros();
    for (size_t row = 0; row < block_count; row++)
    {
      tensor3_t output_block;
      bool initialized = false;
      for (size_t col = 0; col < block_count; col++)
      {
        if (operator_cores[row][col] == nullptr)
        {
          continue;
        }
        const CoreSliceSparsity<data_t>* sparse =
          sparse_cores.empty() ? nullptr : sparse_cores[row][col];
        tensor3_t contribution;
        if constexpr (space == ::boba::execution_space::CPU)
        {
          if (sparse != nullptr && !sparse->pairs.empty())
            contribution = bfun3_sparse(*phi_left[row][col], *sparse, *phi_right[row][col], input_blocks[col]);
          else
            contribution = bfun3(*phi_left[row][col], *operator_cores[row][col], *phi_right[row][col], input_blocks[col]);
        }
        else
        {
          contribution = bfun3(*phi_left[row][col], *operator_cores[row][col], *phi_right[row][col], input_blocks[col]);
        }
        if (initialized)
        {
          output_block += contribution;
        }
        else
        {
          output_block = std::move(contribution);
          initialized = true;
        }
      }
      boba_always_assert(initialized,
                         "Coupled site operator requires a nonempty block in every row.");

      auto output_block_flat = flatten(output_block);
      auto output_view = output.view();
      auto output_block_view = output_block_flat.const_view();
      const auto offset = offsets[row];
      auto scalar_factors_view = scalar_row_factors.empty()
                                   ? output_block_flat.const_view()
                                   : scalar_row_factors[row].const_view();
      const bool use_scalar_factors = !scalar_row_factors.empty();
      const auto block_factor = row_factors.empty() ? data_t{1} : row_factors[row];
      ::boba::loop<space, 1>(static_cast<index_t>(output_block_flat.size()),
                             [=] __boba_host_device__(index_t idx)
      {
        const auto factor = use_scalar_factors ? scalar_factors_view({idx}) : block_factor;
        output_view({offset + idx}) = factor * output_block_view({idx});
      });
    }
    return output;
  }
};

template <typename data_t>
struct TensorTrainAMENBlock
{
  /*
    This class reimplements Oseledets' amen_block_solve using the BoBa framework.
    See https://github.com/oseledets/TT-Toolbox/blob/master/solve/amen_block_solve.m
    Unlike the MATLAB version, it supports block systems whose nonzero blocks
    may have different mode sizes as long as each block maps between compatible
    TensorTrain dimensions. The operator is represented as a
    BlockOperator<TensorTrainMatrix<...>> and the vectors as
    BlockVector<TensorTrain<...>>.
  */

  data_t convergence_tolerance = 1.0e-10;
  index_t kickrank = 4;
  data_t resid_damp = 2.0;
  int max_direct_solve_size = 1500;
  index_t max_allowed_ranks = 1000;
  size_t max_sweeps = 20;
  size_t minimum_sweeps = 1;
  /// Relative SVD tolerance used to truncate residual-enrichment bases.
  real_type_t<data_t> residual_enrichment_svd_tolerance_relative = 0.0;
  /// Absolute SVD tolerance used to truncate residual-enrichment bases.
  real_type_t<data_t> residual_enrichment_svd_tolerance_absolute = 0.0;
  /// Relative SVD tolerance used to split an updated local solution.
  real_type_t<data_t> local_solution_svd_tolerance_relative = 0.0;
  /// Absolute SVD tolerance used to split an updated local solution.
  real_type_t<data_t> local_solution_svd_tolerance_absolute = 0.0;
  // Scalar AMEn-style intermediate normalization is algebraically safe for
  // diagonal block systems. For genuinely coupled block systems the current
  // per-block scalar bookkeeping does not preserve the off-diagonal projected
  // equations, so coupled solves use unnormalized environments.
  bool normalize_coupled_intermediate_quantities = false;
  // Scale coupled site equations explicitly before the local solve. Dense
  // paths currently use one Frobenius-norm factor per block row; matrix-free
  // paths compute every scalar row norm by tensor contraction and apply true
  // row equilibration. Both are algebraic left scalings of B*x=rhs; residual
  // diagnostics and enrichment continue to use the original unscaled system.
  // Unknown/column scaling is intentionally avoided because projected coupled
  // systems may have gauge/nullspace freedom, and reweighting unknowns can
  // change the least-squares representative selected by the local solve.
  bool scale_coupled_site_system = true;
  /// Use Eigen GMRES instead of complete orthogonal decomposition for coupled site systems.
  bool use_eigen_gmres_for_coupled_site_solve = false;
  /// Use matrix-free BoBa GMRES for coupled site systems.
  bool use_boba_gmres_for_coupled_site_solve = false;
  /// Assemble a reference matrix and compare it with the matrix-free action.
  bool validate_coupled_matrix_free_operator = false;
  /// Memory budget (bytes) for reusing the operator row Grams of the matrix-free site scaling across sweeps
  /// (bfun3_operator_row_gram, (rank_left rank_right)^2 x rows per block pair and site); 0 disables the reuse.
  size_t row_norm_gram_cache_bytes = size_t(4) << 30;
  /// Matrix-free site solve: use the nonzero-slice (n, m) pattern of an operator core when at most this fraction of its
  /// slices are nonzero (exact; host only); 0 disables.
  double sparse_site_operator_max_fraction = 0.5;
  /// Evaluate the global residual certificate after every sweep (from minimum_sweeps on), not only once the largest
  /// local residual max_res is below convergence_tolerance.  max_res can stall above the tolerance while the global
  /// residual is already below it.  Meant for a cheap caller-supplied global_residual (see solve()).
  bool certify_every_sweep = false;
  /// With certify_every_sweep: stop (without certifying) once the global residual has fallen by less than this fraction
  /// over the last two certified sweeps.  The floor of |A x - y| / |y| is set by the TT rounding of the iterate and,
  /// for an approximated operator, by that approximation; a fixed target below the floor is never reached.  0: off.
  data_t stagnation_reduction = 0;
  /// Matrix-free site GMRES: right-precondition by the inverse diagonal of the site operator (host only).
  bool site_jacobi_preconditioner = false;
  /// Direct CPU solver: 0=complete orthogonal decomposition, 1=pivoted QR, 2=partial-pivot LU, 3=sparse LU (dense LU above the fill threshold),
  /// 4=sparse LU, with dense minimum-norm (COD) solves above the fill threshold.
  size_t coupled_site_direct_solver_method = 0;
  /// Methods 3 and 4: sites with fill above this use a dense solve (LU for 3, COD for 4); > 1 means always sparse LU
  /// (with the dense minimum-norm fallback when the sparse factorization fails).
  double coupled_site_dense_fill_threshold = 1.1;
  /// \brief Enable per-sweep progress output from `solve()`.
  bool verbose = false;
  /// Saddle-point (inf-sup) enrichment: index of the block whose left factors are also kicked with the
  /// projected image of the other blocks, sum_{j != k} A_{kj} x_j (at most kickrank SVD directions per site).
  /// For a mixed DPG system [G B; B^T 0] with the Riesz block k = 0, this keeps directions of B u in the
  /// Riesz basis, so the projected local systems inherit the discrete inf-sup condition.  -1 disables it.
  int inf_sup_enrichment_block = -1;
  /// Optional per-block relative SVD tolerances for splitting the updated local solution; an entry <= 0 (or a
  /// missing entry) falls back to local_solution_svd_tolerance_relative.  For mixed DPG, the Riesz block is the
  /// representer of the residual, so a looser tolerance there truncates relative to the DPG error.
  std::vector<real_type_t<data_t>> local_solution_svd_tolerance_relative_per_block;

  Solve3D2MLOptions<data_t> solve3d_2ml_options;

  /**
   * \brief Reject unsupported operator/vector combinations for block AMEn.
   *
   * This overload exists only to provide a consistent error when the caller
   * passes a block type combination that this solver does not handle.
   *
   * \param[in] crA Input operator.
   * \param[in] cry Right-hand side.
   * \param[in] tt_initial_guess Initial iterate.
   * \return Unreachable; the function always throws.
   */
  template <typename unsupported_operator, typename unsupported_vector>
  unsupported_vector solve(
    const unsupported_operator& crA,
    const unsupported_vector& cry,
    const unsupported_vector& tt_initial_guess)
  {
    detail::ignore(crA);
    detail::ignore(cry);
    detail::ignore(tt_initial_guess);
    boba_error("Block AMEn not yet supported for this operator/vector combination.");
    return cry;
  }

  /**
   * \brief Solve a square block TT linear system with block AMEn.
   *
   * The operator and vectors are organized as block containers whose entries are
   * TT matrices and TT vectors, respectively. Diagonal-only systems reduce to
   * independent local AMEn updates, while systems with off-diagonal blocks use a
   * coupled site solve and a final block-scalar correction. Unlike the MATLAB
   * block AMEn implementation, block mode sizes are allowed to differ across
   * rows and columns as long as each nonzero block maps between compatible sizes.
   *
   * \tparam random_generator_t Type satisfying the C++
   *          UniformRandomBitGenerator requirements.
   * \param[in] crA Square block operator.
   * \param[in] cry Block right-hand side.
   * \param[in] tt_initial_guess Initial block TT iterate.
   * \param[in,out] random_generator Generator used for residual enrichment.
   * \return Final block TT solution, returned in host space.
   */
  template <size_t dimension, execution_space space, typename random_generator_t>
  ::boba::BlockVector<::boba::TensorTrain<dimension, host_space, data_t>> solve(
    const ::boba::BlockOperator<::boba::TensorTrainMatrix<dimension, space, data_t>>& crA,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& cry,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& tt_initial_guess,
    random_generator_t& random_generator)
  {
    return solve(crA, cry, tt_initial_guess, random_generator,
                 [](const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>&) { return data_t(-1); });
  }

  /**
   * \brief Block AMEn with a caller-supplied global residual.
   *
   * The convergence certificate (and the verbose per-sweep report) needs
   * |A x - y| / |y|.  By default this is formed as the TT product A x, whose
   * ranks are rank(A) * rank(x) before any rounding; for a dense-rank operator
   * this exact product dominates the solve (a few hundred ranks per block
   * pair).  A caller with a cheaper exact apply (for example matrix-free on full
   * vectors) passes global_residual(x) returning |A x - y| / |y|; a negative
   * return value selects the built-in TT product.
   */
  template <size_t dimension, execution_space space, typename random_generator_t, typename residual_fn_t>
  ::boba::BlockVector<::boba::TensorTrain<dimension, host_space, data_t>> solve(
    const ::boba::BlockOperator<::boba::TensorTrainMatrix<dimension, space, data_t>>& crA,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& cry,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& tt_initial_guess,
    random_generator_t& random_generator,
    residual_fn_t&& global_residual)
  {
    BOBA_CALI_MARK
    checkpoint();

    boba_always_assert_equal(crA.block_rows, crA.block_cols, "To use Block AMEn, the operator must be square");
    boba_always_assert_equal(cry.block_size, crA.block_rows, "RHS block size must match operator block size");
    boba_always_assert_equal(tt_initial_guess.block_size, crA.block_rows, "Initial guess block size must match operator block size");

    size_t Kay = crA.block_rows;
    bool has_offdiag_blocks = false;
    data_t norm_cry = ::boba::norm_frobenius(cry);
    if (::boba::is_tiny(norm_cry))
    {
      norm_cry = 1.0;
    }

    // Validate the block layout and detect whether any off-diagonal couplings are present.
    for (size_t ki = 0; ki < Kay; ki++)
    {
      for (size_t kj = 0; kj < Kay; kj++)
      {
        if (crA({ki, kj}).get_number_elements() > 0)
        {
          has_offdiag_blocks = has_offdiag_blocks || (ki != kj);
          // Matrix A{ki,kj} maps from block kj to block ki
          // So: A{ki,kj}.core_cols() should match cry(kj).sizes()
          //     A{ki,kj}.core_rows() should match cry(ki).sizes()
          for (size_t d = 0; d < dimension; d++)
          {
            boba_always_assert_equal(crA({ki, kj}).core_cols(d), cry(kj).sizes(d), "Matrix block column size must match RHS size");
            boba_always_assert_equal(crA({ki, kj}).core_rows(d), cry(ki).sizes(d), "Matrix block row size must match RHS size");
          }
        }
      }
    }

    auto crx = tt_initial_guess;
    // The scalar-AMEn normalization below is valid only when each block row is
    // an independent diagonal solve. In a coupled block system, normalizing a
    // row by quantities derived from its diagonal block changes the relative
    // scale of the off-diagonal projected operators and therefore changes the
    // local equation. Coupled systems instead keep the environments
    // unnormalized. The assembled path applies block-row scaling. The
    // matrix-free path applies finer scalar-row equilibration, obtaining every
    // projected row norm by contracting the tensor network with itself.
    if (has_offdiag_blocks && normalize_coupled_intermediate_quantities)
    {
      boba_warn("TensorTrainAMENBlock coupled intermediate normalization is disabled because the current scaling does not preserve off-diagonal projected equations.");
    }
    const bool normalize_intermediate_quantities = !has_offdiag_blocks;

    checkpoint();

    // Allocate the AMEn environments and the enrichment tensors for each block row/column pair.
    std::vector<std::vector<::boba::TensorTrain<dimension + 1, space, data_t>>> phia(Kay);
    std::vector<::boba::TensorTrain<dimension + 1, space, data_t>> phiy(Kay);
    std::vector<std::vector<::boba::TensorTrain<dimension + 1, space, data_t>>> phiza(Kay);
    std::vector<::boba::TensorTrain<dimension + 1, space, data_t>> phizy(Kay);
    std::vector<::boba::TensorTrain<dimension, space, data_t>> crz(Kay);

    checkpoint();
    for (size_t ki = 0; ki < Kay; ki++)
    {
      phia[ki].resize(Kay);
      phiza[ki].resize(Kay);

      for (size_t kj = 0; kj < Kay; kj++)
      {
        if (crA({ki, kj}).get_number_elements() > 0)
        {
          phia[ki][kj].resize(::boba::filled_array<dimension + 1>(1_z));
          phia[ki][kj].fill_with(1);

          phiza[ki][kj].resize(::boba::filled_array<dimension + 1>(1_z));
          phiza[ki][kj].fill_with(1);
        }
      }

      phiy[ki].resize(::boba::filled_array<dimension + 1>(1_z));
      phiy[ki].fill_with(1);

      phizy[ki].resize(::boba::filled_array<dimension + 1>(1_z));
      phizy[ki].fill_with(1);

      crz[ki].resize(cry(ki).sizes());
      for (size_t d = 0; d < dimension; d++)
      {
        auto z_rank = boba::max<index_t>(kickrank, 1);
        auto rank_left = (d == 0) ? 1 : z_rank;
        auto rank_right = (d == dimension - 1) ? 1 : z_rank;

        crz[ki].cores[d].resize({rank_left, cry(ki).sizes(d), rank_right});
        crz[ki].cores[d].fill_with_random(random_generator);
      }
    }

    checkpoint();

    // Track per-interface normalization factors so the left/right environment
    // contractions stay numerically stable across sweeps.
    std::vector<::boba::Array<data_t, dimension - 1>> nrmsx(Kay);
    std::vector<std::vector<::boba::Array<data_t, dimension - 1>>> nrmsa(Kay);
    std::vector<::boba::Array<data_t, dimension - 1>> nrmsy(Kay);
    std::vector<data_t> nrmsc(Kay, 1.0);
    std::vector<::boba::Array<size_t, dimension + 1>> rz(Kay);
    std::vector<index_t> rznew(Kay, 0);
    std::vector<boba::Matrix<space, data_t>> crznew(Kay);
    std::vector<boba::Tensor<3, space, data_t>> crznew_tensor(Kay);

    for (size_t ki = 0; ki < Kay; ki++)
    {
      nrmsx[ki] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      nrmsy[ki] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      nrmsa[ki].resize(Kay);
      for (size_t kj = 0; kj < Kay; kj++)
      {
        nrmsa[ki][kj] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      }
      rz[ki] = crz[ki].ranks();
    }

    checkpoint();
    // Main AMEn sweeps: first move right-to-left to update environments and enrichment data.
    row_norm_gram_cache_.clear(); row_norm_gram_cache_used_ = 0;
    core_sparsity_cache_.clear();   // keyed by (block row, block column, site) of this operator
    bool last_sweep = false;
    bool converged = false;
    bool stagnated = false;
    std::vector<data_t> certified_history;
    for (size_t swp = 0; swp < max_sweeps; swp++)
    {
      timing_.start_sweep();
      for (size_t ip1 = dimension; ip1 > 1; ip1--)
      {
        size_t i = ip1 - 1;
        // Snapshot the current site cores before any block is updated so
        // off-diagonal couplings at this site all see a consistent iterate.
        std::vector<boba::Tensor<3, space, data_t>> site_x_cores(Kay);
        for (size_t kj = 0; kj < Kay; kj++)
        {
          site_x_cores[kj] = crx(kj).cores[i];
        }

        for (size_t ki = 0; ki < Kay; ki++)
        {
          if ((kickrank > 0) && (!last_sweep))
          {
            // Build or propagate the AMEn enrichment basis z for this site.
            // On later sweeps this comes from the local residual; on the first
            // sweep we simply orthogonalize the random initialization.
            bool propagate_z_factor = false;
            auto current_z_right_rank = rz[ki][i + 1];
            if (swp > 0)
            {
              // Contract all operator blocks against the current block vector to
              // form A*x in the z basis at this site, then subtract the projected
              // right-hand side to obtain a residual-like enrichment candidate.
              boba::Tensor<3, space, data_t> crzAt({rz[ki][i], cry(ki).sizes(i), rz[ki][i + 1]});
              crzAt.fill_with_zeros();

              for (size_t kj = 0; kj < Kay; kj++)
              {
                if (crA({ki, kj}).get_number_elements() == 0)
                {
                  continue;
                }

                auto x_core = site_x_cores[kj];
                crzAt += bfun3_block(
                  phiza[ki][kj].cores[i],
                  crA({ki, kj}).cores[i],
                  phiza[ki][kj].cores[i + 1],
                  x_core,
                  ki,
                  kj,
                  i,
                  "rl-crzAt");
              }

              auto crzy2 = project_phizy_block(
                phizy[ki].cores[i],
                cry(ki).cores[i],
                phizy[ki].cores[i + 1],
                ki,
                i,
                "rl-project_phizy");
              if (normalize_intermediate_quantities)
              {
                crzy2 *= nrmsc[ki];
              }
              crzy2 -= crzAt;
              crznew[ki] = boba::reshape_to_matrix(crzy2, {rz[ki][i], cry(ki).sizes(i) * rz[ki][i + 1]});

              ::boba::SVD<space, data_t> svd;
              svd.tolerance_relative = residual_enrichment_svd_tolerance_relative;
              svd.tolerance_absolute = residual_enrichment_svd_tolerance_absolute;
              svd.max_kept_singular_values = kickrank;
              svd(crznew[ki]);
              crznew[ki] = svd.V;
            }
            else
            {
              propagate_z_factor = true;
              auto crznew_permute = crz[ki].cores[i];
              current_z_right_rank = crznew_permute.sizes(2);
              crznew[ki] = boba::reshape_to_matrix(
                             crznew_permute,
                             {crznew_permute.sizes(0), crznew_permute.sizes(1) * crznew_permute.sizes(2)})
                             .transpose();
            }

            checkpoint();
            // Re-orthogonalize the enrichment directions and push any leftover
            // factor into the neighboring core so the TT structure remains valid.
            QR<space, data_t> qr;
            qr(crznew[ki]);
            crznew[ki] = qr.Q;
            auto rv = qr.R;

            rznew[ki] = crznew[ki].cols();
            if ((swp == 0) && propagate_z_factor)
            {
              auto crz_prev = crz[ki].cores[i - 1];
              try
              {
                crz[ki].cores[i - 1] = ::boba::tensor_contraction<1>(
                  {"rzm1", "n", "rz"}, crz_prev, {"k", "rz"}, rv, {"rzm1", "n", "k"});
              }
              catch (const std::exception& error)
              {
                std::ostringstream msg;
                msg << "TensorTrainAMENBlock rl-z-propagate failed at sweep " << swp
                    << ", site " << i
                    << " for block " << ki
                    << " with crz_prev " << crz_prev.sizes()
                    << ", rv " << rv.sizes()
                    << ": " << error.what();
                throw std::runtime_error(msg.str());
              }
            }
            crz[ki].cores[i] = boba::reshape_from_matrix<3>(crznew[ki].transpose(), {rznew[ki], cry(ki).sizes(i), current_z_right_rank});
          }

          if (swp > 0)
          {
            // Undo the normalization that was applied while building the right
            // environments before re-factorizing the current solution core.
            if (normalize_intermediate_quantities)
            {
              nrmsc[ki] = nrmsc[ki] / (nrmsy[ki][i - 1] / (nrmsa[ki][ki][i - 1] * nrmsx[ki][i - 1]));
            }
          }

          auto cr = crx(ki).cores[i];
          auto unfold = ::boba::compute_unfold_right(cr).transpose();

          // Right-orthogonalize the current x core and propagate the triangular
          // factor into the left neighbor so the sweep invariant is preserved.
          checkpoint();
          QR<space, data_t> qr;
          qr(unfold);
          cr = boba::reshape_from_matrix<3>(qr.Q.transpose(), {qr.Q.cols(), crx(ki).sizes(i), crx(ki).ranks(i + 1)});
          auto rv = qr.R;

          checkpoint();
          auto cr2 = crx(ki).cores[i - 1];
          Tensor<3, space, data_t> new_cr2mat;
          try
          {
            new_cr2mat = ::boba::tensor_contraction<1>(
              {"rxm1", "n", "rx"}, cr2, {"k", "rx"}, rv, {"rxm1", "n", "k"});
          }
          catch (const std::exception& error)
          {
            std::ostringstream msg;
            msg << "TensorTrainAMENBlock rl-x-propagate failed at site " << i
                << " for block " << ki
                << " with cr2 " << cr2.sizes()
                << ", rv " << rv.sizes()
                << ": " << error.what();
            throw std::runtime_error(msg.str());
          }

          auto curnorm = ::boba::sqrt(::boba::abs(::boba::inner_product(new_cr2mat, new_cr2mat)));
          if (normalize_intermediate_quantities && !::boba::is_tiny(curnorm))
          {
            new_cr2mat *= (1.0 / curnorm);
          }
          else
          {
            curnorm = 1;
          }

          if (normalize_intermediate_quantities)
          {
            nrmsx[ki][i - 1] *= curnorm;
          }
          crx(ki).cores[i - 1] = new_cr2mat;
          crx(ki).cores[i] = cr;
        }

        // Update right-to-left operator and RHS environments after every block
        // has contributed its newly orthogonalized site core.
        for (size_t row = 0; row < Kay; row++)
        {
          for (size_t col = 0; col < Kay; col++)
          {
            if (crA({row, col}).get_number_elements() == 0)
            {
              continue;
            }
            auto phia_external_norm = normalize_intermediate_quantities
                                        ? ((row == col) ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0))
                                        : static_cast<data_t>(1.0);

            std::tie(phia[row][col].cores[i], nrmsa[row][col][i - 1]) =
              compute_next_phi_block<dimension>(
                phia[row][col].cores[i + 1],
                crx(row).cores[i],
                crA({row, col}).cores[i],
                crx(col).cores[i],
                false,
                phia_external_norm,
                row,
                col,
                i,
                "rl-phia");

            if (row != col || !normalize_intermediate_quantities)
            {
              nrmsa[row][col][i - 1] = 1.0;
            }
          }

          std::tie(phiy[row].cores[i], nrmsy[row][i - 1]) =
            compute_next_phi_block<dimension>(
              phiy[row].cores[i + 1],
              crx(row).cores[i],
              cry(row).cores[i],
              false,
              normalize_intermediate_quantities ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0),
              row,
              row,
              i,
              "rl-phiy");

          if (normalize_intermediate_quantities)
          {
            nrmsc[row] = nrmsc[row] * (nrmsy[row][i - 1] / (nrmsa[row][row][i - 1] * nrmsx[row][i - 1]));
          }
        }

        if ((kickrank > 0) && (!last_sweep))
        {
          // Mirror the same environment propagation for the enrichment tensor z.
          for (size_t row = 0; row < Kay; row++)
          {
            rz[row][i] = rznew[row];

            for (size_t col = 0; col < Kay; col++)
            {
              if (crA({row, col}).get_number_elements() == 0)
              {
                continue;
              }

              auto phiza_external_norm = static_cast<data_t>(1.0);
              if (normalize_intermediate_quantities && (row == col))
              {
                phiza_external_norm = nrmsa[row][col][i - 1];
              }

              std::tie(phiza[row][col].cores[i], std::ignore) =
                compute_next_phi_block<dimension>(
                  phiza[row][col].cores[i + 1],
                  crz[row].cores[i],
                  crA({row, col}).cores[i],
                  crx(col).cores[i],
                  false,
                  phiza_external_norm,
                  row,
                  col,
                  i,
                  "rl-phiza");
            }

            std::tie(phizy[row].cores[i], std::ignore) =
              compute_next_phi_block<dimension>(
                phizy[row].cores[i + 1],
                crz[row].cores[i],
                cry(row).cores[i],
                false,
                normalize_intermediate_quantities ? nrmsy[row][i - 1] : static_cast<data_t>(1.0),
                row,
                row,
                i,
                "rl-phizy");
          }
        }
      }

      checkpoint();

      data_t max_res = 0.0;
      data_t max_dx = 0.0;

      timing_.end_right_to_left();
      // Sweep left-to-right, solve the local problems, and adapt the TT ranks.
      for (size_t i = 0; i < dimension; i++)
      {
        std::vector<boba::Tensor<3, space, data_t>> site_x_cores(Kay);
        for (size_t kj = 0; kj < Kay; kj++)
        {
          site_x_cores[kj] = crx(kj).cores[i];
        }

        // Temporary storage for the site-local linear system. For diagonal block
        // systems each block is solved independently; otherwise all blocks are
        // stacked into one coupled dense solve at this site.
        std::vector<boba::Vector<space, data_t>> coupled_rhs(Kay);
        std::vector<boba::Vector<space, data_t>> coupled_sol(Kay);
        std::vector<data_t> coupled_norm_rhs(Kay, 0.0);
        std::vector<data_t> coupled_res_prev(Kay, 0.0);
        std::vector<data_t> coupled_res_new(Kay, 0.0);
        std::vector<index_t> coupled_offsets(Kay + 1, 0);
        auto use_coupled_site_solve = has_offdiag_blocks;
        auto real_tol = (convergence_tolerance / ::boba::sqrt(static_cast<data_t>(dimension))) / resid_damp;
        bool skip_solution_enrichment = false;
        data_t coupled_prev_res_for_convergence = 0.0;
        data_t coupled_new_res_for_convergence = 0.0;

        timing_.mark(0);
        // For coupled systems, assemble one local block system spanning all blocks at this site.
        if (use_coupled_site_solve)
        {
          for (size_t ki = 0; ki < Kay; ki++)
          {
            // Project each block RHS into the current local basis so every block
            // contributes one contiguous segment to the coupled site system.
            auto y1 = cry(ki).cores[i];
            if (normalize_intermediate_quantities)
            {
              y1 *= nrmsc[ki];
            }

            auto rhs_tensor = project_phizy_block(
              phiy[ki].cores[i],
              y1,
              phiy[ki].cores[i + 1],
              ki,
              i,
              "lr-rhs-project");
            coupled_rhs[ki] = flatten(rhs_tensor);
            coupled_norm_rhs[ki] = ::boba::norm_frobenius(coupled_rhs[ki]);
            coupled_offsets[ki + 1] = coupled_offsets[ki] + static_cast<index_t>(coupled_rhs[ki].size());
          }

          auto total_local_size = coupled_offsets[Kay];
          if (std::getenv("BOBA_AMEN_LOG_SITE_SIZES"))
          {
            std::cout << "TensorTrainAMENBlock: site " << i << " coupled_local_size " << total_local_size << " blocks";
            for (size_t ki = 0; ki < Kay; ki++)
            {
              std::cout << ' ' << crx(ki).ranks(i) << 'x' << cry(ki).sizes(i) << 'x' << crx(ki).ranks(i + 1);
            }
            std::cout << std::endl;
          }
          const bool use_matrix_free_coupled_solve =
            use_boba_gmres_for_coupled_site_solve;
          const bool assemble_coupled_matrix =
            !use_matrix_free_coupled_solve || validate_coupled_matrix_free_operator;
          boba::Matrix<space, data_t> coupled_B;
          if (assemble_coupled_matrix)
          {
            coupled_B.resize({total_local_size, total_local_size});
            coupled_B.fill_with_zeros();
          }

          boba::Vector<space, data_t> coupled_rhs_full({total_local_size});
          coupled_rhs_full.fill_with_zeros();

          boba::Vector<space, data_t> coupled_sol_prev_full({total_local_size});
          coupled_sol_prev_full.fill_with_zeros();

          std::vector<data_t> coupled_row_norm_squared(Kay, 0.0);
          std::vector<boba::Vector<space, data_t>> coupled_scalar_row_norm_squared(Kay);
          for (size_t ki = 0; ki < Kay; ki++)
          {
            // Assemble the dense site matrix block-by-block from the projected
            // TT operator environments and the old local iterate.
            copy_vector_block(coupled_rhs_full, coupled_offsets[ki], coupled_rhs[ki]);
            copy_vector_block(coupled_sol_prev_full, coupled_offsets[ki], flatten(site_x_cores[ki]));
            coupled_row_norm_squared[ki] += coupled_norm_rhs[ki] * coupled_norm_rhs[ki];
            coupled_scalar_row_norm_squared[ki].resize(coupled_rhs[ki].sizes());
            auto rhs_norm_squared_view = coupled_scalar_row_norm_squared[ki].view();
            auto rhs_view = coupled_rhs[ki].const_view();
            ::boba::loop<space, 1>(static_cast<index_t>(coupled_rhs[ki].size()),
                                   [=] __boba_host_device__(index_t idx)
            {
              auto rhs_abs = ::boba::abs(rhs_view({idx}));
              rhs_norm_squared_view({idx}) = rhs_abs * rhs_abs;
            });

            for (size_t kj = 0; kj < Kay; kj++)
            {
              if (crA({ki, kj}).get_number_elements() == 0)
              {
                continue;
              }

              data_t matrix_free_block_norm_squared = 0.0;
              // the row norms only feed the site row scaling: skip them when it is off
              if (use_matrix_free_coupled_solve && scale_coupled_site_system)
              {
                // The operator part of the row norms depends only on the operator core, which does not change between
                // sweeps: reuse it across sweeps, within row_norm_gram_cache_bytes.
                auto row_norms = [&]() {
                  if constexpr (space == ::boba::execution_space::CPU)
                  {
                    if (const auto* gram = cached_operator_row_gram(crA({ki, kj}).cores[i], ki, kj, i))
                      return ::boba::bfun3_matrix_row_norms_squared_from_gram(phia[ki][kj].cores[i], *gram, phia[ki][kj].cores[i + 1]);
                  }
                  return ::boba::bfun3_matrix_row_norms_squared(phia[ki][kj].cores[i], crA({ki, kj}).cores[i], phia[ki][kj].cores[i + 1]);
                };
                auto scalar_block_norm_squared = flatten(row_norms());
                matrix_free_block_norm_squared = scalar_block_norm_squared.sum_reduce();
                coupled_row_norm_squared[ki] += matrix_free_block_norm_squared;
                auto row_norm_squared_view = coupled_scalar_row_norm_squared[ki].view();
                auto block_norm_squared_view = scalar_block_norm_squared.const_view();
                ::boba::loop<space, 1>(
                  static_cast<index_t>(scalar_block_norm_squared.size()),
                  [=] __boba_host_device__(index_t idx)
                {
                  row_norm_squared_view({idx}) += block_norm_squared_view({idx});
                });
              }

              if (assemble_coupled_matrix)
              {
                auto B_block = bfun3_matrix(
                  phia[ki][kj].cores[i],
                  crA({ki, kj}).cores[i],
                  phia[ki][kj].cores[i + 1]);

                auto B_block_norm = ::boba::norm_frobenius(B_block);
                if (!use_matrix_free_coupled_solve)
                {
                  coupled_row_norm_squared[ki] += B_block_norm * B_block_norm;
                }
                else if (validate_coupled_matrix_free_operator)
                {
                  auto dense_norm_squared = B_block_norm * B_block_norm;
                  auto norm_denominator = boba::max(
                    ::boba::abs(dense_norm_squared), ::boba::epsilon<data_t>());
                  std::cout << "TensorTrainAMENBlock: matrix-free norm validation site " << i
                            << ", block (" << ki << ", " << kj << ")"
                            << ", relative_squared_norm_difference: "
                            << ::boba::abs(matrix_free_block_norm_squared - dense_norm_squared) /
                                 norm_denominator << "\n";
                }

                coupled_B.replace_submatrix(
                  {coupled_offsets[ki], coupled_offsets[ki + 1]},
                  {coupled_offsets[kj], coupled_offsets[kj + 1]},
                  B_block);
              }
            }
            if (use_matrix_free_coupled_solve && validate_coupled_matrix_free_operator)
            {
              auto scalar_norm_sum = coupled_scalar_row_norm_squared[ki].sum_reduce();
              auto denominator = boba::max(
                ::boba::abs(coupled_row_norm_squared[ki]), ::boba::epsilon<data_t>());
              std::cout << "TensorTrainAMENBlock: scalar-row norm validation site " << i
                        << ", block row " << ki
                        << ", relative_sum_difference: "
                        << ::boba::abs(scalar_norm_sum - coupled_row_norm_squared[ki]) /
                             denominator << "\n";
            }
          }

          CoupledSolve3DGMRESOperator<space, data_t> coupled_operator;
          if (use_matrix_free_coupled_solve)
          {
            coupled_operator.phi_left.resize(Kay, std::vector<const boba::Tensor<3, space, data_t>*>(Kay, nullptr));
            coupled_operator.operator_cores.resize(Kay, std::vector<const boba::Tensor<4, space, data_t>*>(Kay, nullptr));
            coupled_operator.phi_right.resize(Kay, std::vector<const boba::Tensor<3, space, data_t>*>(Kay, nullptr));
            coupled_operator.input_sizes.resize(Kay);
            coupled_operator.offsets = coupled_offsets;
            coupled_operator.row_factors.assign(Kay, data_t{1});
            for (size_t kj = 0; kj < Kay; kj++)
            {
              coupled_operator.input_sizes[kj] = site_x_cores[kj].sizes();
            }
            for (size_t ki = 0; ki < Kay; ki++)
            {
              for (size_t kj = 0; kj < Kay; kj++)
              {
                if (crA({ki, kj}).get_number_elements() == 0)
                {
                  continue;
                }
                coupled_operator.phi_left[ki][kj] = &phia[ki][kj].cores[i];
                coupled_operator.operator_cores[ki][kj] = &crA({ki, kj}).cores[i];
                coupled_operator.phi_right[ki][kj] = &phia[ki][kj].cores[i + 1];
                if constexpr (space == ::boba::execution_space::CPU)
                {
                  if (sparse_site_operator_max_fraction > 0)
                  {
                    if (coupled_operator.sparse_cores.empty())
                      coupled_operator.sparse_cores.resize(Kay, std::vector<const CoreSliceSparsity<data_t>*>(Kay, nullptr));
                    coupled_operator.sparse_cores[ki][kj] = cached_core_sparsity(crA({ki, kj}).cores[i], ki, kj, i);
                  }
                }
              }
            }
          }

          timing_.mark(1);
          // Solve the coupled site problem once, then slice the solution and
          // residual diagnostics back into the per-block containers.
          if (use_matrix_free_coupled_solve && validate_coupled_matrix_free_operator)
          {
            auto assembled_action = coupled_B * coupled_sol_prev_full;
            auto matrix_free_action = coupled_operator * coupled_sol_prev_full;
            auto action_difference = assembled_action - matrix_free_action;
            auto reference_norm = ::boba::norm_frobenius(assembled_action);
            if (::boba::is_tiny(reference_norm))
            {
              reference_norm = 1.0;
            }
            std::cout << "TensorTrainAMENBlock: matrix-free validation site " << i
                      << ", relative_action_difference: "
                      << ::boba::norm_frobenius(action_difference) / reference_norm
                      << "\n";
          }
          auto coupled_prev_lhs = use_matrix_free_coupled_solve
                                    ? coupled_operator * coupled_sol_prev_full
                                    : coupled_B * coupled_sol_prev_full;
          auto coupled_prev_residual = coupled_prev_lhs - coupled_rhs_full;
          auto coupled_rhs_norm = ::boba::norm_frobenius(coupled_rhs_full);
          if (is_tiny(coupled_rhs_norm))
          {
            coupled_rhs_norm = 1.0;
          }
          auto coupled_prev_res_norm = ::boba::norm_frobenius(coupled_prev_residual) / coupled_rhs_norm;
          coupled_prev_res_for_convergence = coupled_prev_res_norm;
          auto coupled_sol_full = coupled_sol_prev_full;

          data_t coupled_max_prev_res = 0.0;
          for (size_t ki = 0; ki < Kay; ki++)
          {
            auto local_size = coupled_offsets[ki + 1] - coupled_offsets[ki];
            auto prev_residual = extract_vector_block(coupled_prev_residual, coupled_offsets[ki], local_size);
            auto prev_denominator = coupled_norm_rhs[ki];

            if (!is_tiny(prev_denominator))
            {
              coupled_res_prev[ki] = ::boba::norm_frobenius(prev_residual) / prev_denominator;
              coupled_max_prev_res = boba::max(coupled_max_prev_res, coupled_res_prev[ki]);
            }
          }

          bool coupled_skipped_solve = true;
          if (coupled_prev_res_norm > real_tol)
          {
            coupled_skipped_solve = false;
            if (use_matrix_free_coupled_solve)
            {
              if (scale_coupled_site_system)
              {
                auto coupled_operator_scaled = coupled_operator;
                auto coupled_rhs_scaled = coupled_rhs_full;
                coupled_operator_scaled.scalar_row_factors.resize(Kay);
                for (size_t ki = 0; ki < Kay; ki++)
                {
                  auto local_size = coupled_offsets[ki + 1] - coupled_offsets[ki];
                  coupled_operator_scaled.scalar_row_factors[ki].resize({local_size});
                  auto factors_view = coupled_operator_scaled.scalar_row_factors[ki].view();
                  auto norms_view = coupled_scalar_row_norm_squared[ki].const_view();
                  auto rhs_scaled_view = coupled_rhs_scaled.view();
                  auto offset = coupled_offsets[ki];
                  ::boba::loop<space, 1>(local_size,
                                         [=] __boba_host_device__(index_t idx)
                  {
                    auto norm = ::boba::sqrt(::boba::abs(norms_view({idx})));
                    auto factor = ::boba::is_tiny(norm) ? data_t{1} : data_t{1} / norm;
                    factors_view({idx}) = factor;
                    rhs_scaled_view({offset + idx}) *= factor;
                  });
                }
                coupled_sol_full = solve_coupled_site_system(
                                     coupled_operator_scaled, coupled_rhs_scaled,
                                     coupled_sol_prev_full, i);
              }
              else
              {
                coupled_sol_full = solve_coupled_site_system(
                                     coupled_operator, coupled_rhs_full,
                                     coupled_sol_prev_full, i);
              }
            }
            else if (scale_coupled_site_system)
            {
              // This scaling is a local equation preconditioner, not a TT
              // environment normalization. Scaling complete equation rows in
              // both B and rhs preserves the solution set of B*x=rhs while
              // reducing the dynamic range seen by the dense local solver.
              // We deliberately do not scale columns/unknowns here: the
              // projected AMEn systems can have gauge/nullspace freedom, and
              // column scaling changes the least-squares metric and may select
              // a different local representative.
              auto coupled_B_scaled = coupled_B;
              auto coupled_rhs_scaled = coupled_rhs_full;
              std::vector<data_t> row_factors(Kay, 1.0);
              for (size_t ki = 0; ki < Kay; ki++)
              {
                row_factors[ki] = make_local_scale_factor(coupled_row_norm_squared[ki]);
                scale_vector_block(coupled_rhs_scaled, coupled_offsets[ki], coupled_offsets[ki + 1] - coupled_offsets[ki], row_factors[ki]);
              }

              for (size_t ki = 0; ki < Kay; ki++)
              {
                for (size_t kj = 0; kj < Kay; kj++)
                {
                  scale_matrix_block(
                    coupled_B_scaled,
                    coupled_offsets[ki],
                    coupled_offsets[ki + 1],
                    coupled_offsets[kj],
                    coupled_offsets[kj + 1],
                    row_factors[ki]);
                }
              }

              if (const char* dump_dir = std::getenv("BOBA_AMEN_DUMP_DIR"))
              {
                // Debug: write the scaled dense site system and the Kronecker factors of block (0,0) for offline
                // preconditioner studies.  Sites larger than BOBA_AMEN_DUMP_MIN_SIZE, at most BOBA_AMEN_DUMP_MAX files.
                static int dumps_written = 0;
                const char* min_env = std::getenv("BOBA_AMEN_DUMP_MIN_SIZE");
                const char* max_env = std::getenv("BOBA_AMEN_DUMP_MAX");
                const index_t min_size = min_env ? std::atol(min_env) : 0;
                const int max_dumps = max_env ? std::atoi(max_env) : 4;
                if (coupled_B_scaled.rows() >= min_size && dumps_written < max_dumps)
                {
                  const std::string base = std::string(dump_dir) + "/site" + std::to_string(i) + "_sweep" +
                                           std::to_string(swp) + "_n" + std::to_string(coupled_B_scaled.rows());
                  auto write_raw = [](const std::string& name, const data_t* data, size_t count) {
                    std::ofstream out(name, std::ios::binary);
                    out.write(reinterpret_cast<const char*>(data), std::streamsize(count * sizeof(data_t)));
                  };
                  write_raw(base + "_A.bin", coupled_B_scaled.data(), coupled_B_scaled.size());
                  write_raw(base + "_b.bin", coupled_rhs_scaled.data(), coupled_rhs_scaled.size());
                  const auto& phiL = phia[0][0].cores[i];
                  const auto& Acore = crA({0, 0}).cores[i];
                  const auto& phiR = phia[0][0].cores[i + 1];
                  write_raw(base + "_phiL.bin", phiL.data(), phiL.size());
                  write_raw(base + "_Acore.bin", Acore.data(), Acore.size());
                  write_raw(base + "_phiR.bin", phiR.data(), phiR.size());
                  std::ofstream meta(base + "_meta.txt");
                  meta << "rows " << coupled_B_scaled.rows() << "\noffsets";
                  for (size_t kj = 0; kj <= Kay; kj++) { meta << ' ' << coupled_offsets[kj]; }
                  meta << "\nrow_factors";
                  for (size_t kj = 0; kj < Kay; kj++) { meta << ' ' << row_factors[kj]; }
                  meta << "\nphiL " << phiL.sizes(0) << ' ' << phiL.sizes(1) << ' ' << phiL.sizes(2)
                       << "\nAcore " << Acore.sizes(0) << ' ' << Acore.sizes(1) << ' ' << Acore.sizes(2) << ' ' << Acore.sizes(3)
                       << "\nphiR " << phiR.sizes(0) << ' ' << phiR.sizes(1) << ' ' << phiR.sizes(2) << '\n';
                  ++dumps_written;
                }
              }
              coupled_sol_full = solve_coupled_site_system(
                                   coupled_B_scaled, coupled_rhs_scaled, coupled_sol_prev_full, i);
            }
            else
            {
              coupled_sol_full = solve_coupled_site_system(
                                   coupled_B, coupled_rhs_full, coupled_sol_prev_full, i);
            }
          }
          skip_solution_enrichment = coupled_skipped_solve && !normalize_intermediate_quantities;
          auto coupled_new_lhs = use_matrix_free_coupled_solve
                                   ? coupled_operator * coupled_sol_full
                                   : coupled_B * coupled_sol_full;
          auto coupled_new_residual = coupled_new_lhs - coupled_rhs_full;
          auto coupled_new_res_norm = ::boba::norm_frobenius(coupled_new_residual) / coupled_rhs_norm;
          coupled_new_res_for_convergence = coupled_new_res_norm;

          timing_.mark(2);
          // Keep per-block residuals as diagnostics only. Individual projected
          // RHS blocks can nearly cancel, making norm(residual_i)/norm(rhs_i)
          // dominated by roundoff even when the stacked coupled local system is
          // solved accurately. Coupled convergence and skipping therefore use
          // the stacked residual above, whose denominator is the full coupled
          // RHS norm and matches the algebraic system solved at this site.
          data_t coupled_max_new_res = 0.0;
          for (size_t ki = 0; ki < Kay; ki++)
          {
            auto local_size = coupled_offsets[ki + 1] - coupled_offsets[ki];
            coupled_sol[ki] = extract_vector_block(coupled_sol_full, coupled_offsets[ki], local_size);

            auto new_residual = extract_vector_block(coupled_new_residual, coupled_offsets[ki], local_size);
            auto new_denominator = coupled_norm_rhs[ki];

            if (!is_tiny(new_denominator))
            {
              coupled_res_new[ki] = ::boba::norm_frobenius(new_residual) / new_denominator;
              coupled_max_new_res = boba::max(coupled_max_new_res, coupled_res_new[ki]);
            }
          }
          if (verbose)
          {
            std::cout << "TensorTrainAMENBlock: site " << i
                      << ", coupled_prev_res: " << coupled_prev_res_norm
                      << ", coupled_new_res: " << coupled_new_res_norm
                      << ", coupled_max_prev_res: " << coupled_max_prev_res
                      << ", coupled_max_new_res: " << coupled_max_new_res
                      << ", coupled_skipped_solve: " << coupled_skipped_solve
                      << "\n";
          }
        }

        for (size_t ki = 0; ki < Kay; ki++)
        {
          auto y1 = cry(ki).cores[i];
          if (normalize_intermediate_quantities)
          {
            y1 *= nrmsc[ki];
          }

          boba::Vector<space, data_t> rhs_flat;
          data_t norm_rhs = 0.0;

          data_t res_new, res_prev;
          boba::Vector<space, data_t> sol;
          boba::Matrix<space, data_t> B;
          auto sol_prev = site_x_cores[ki];
          auto use_coupled_local_solution = use_coupled_site_solve;

          if (use_coupled_local_solution)
          {
            // Reuse the global coupled site solve assembled above.
            rhs_flat = coupled_rhs[ki];
            norm_rhs = coupled_norm_rhs[ki];
            sol = coupled_sol[ki];
            res_prev = coupled_prev_res_for_convergence;
            res_new = coupled_new_res_for_convergence;
          }
          else
          {
            // For a diagonal site solve, project the RHS into the current basis
            // and subtract any off-diagonal block contributions explicitly.
            auto rhs_tensor = project_phizy_block(
              phiy[ki].cores[i],
              y1,
              phiy[ki].cores[i + 1],
              ki,
              i,
              "lr-rhs-project");

            for (size_t kj = 0; kj < Kay; kj++)
            {
              if (kj == ki)
              {
                continue;
              }
              if (crA({ki, kj}).get_number_elements() == 0)
              {
                continue;
              }

              auto contrib = bfun3_block(
                phia[ki][kj].cores[i],
                crA({ki, kj}).cores[i],
                phia[ki][kj].cores[i + 1],
                site_x_cores[kj],
                ki,
                kj,
                i,
                "lr-rhs-offdiag");

              rhs_tensor -= contrib;
            }

            rhs_flat = flatten(rhs_tensor);
            norm_rhs = ::boba::norm_frobenius(rhs_flat);

            auto linear_solve_size_estimate = crx(ki).ranks(i) * crx(ki).sizes(i) * crx(ki).ranks(i + 1);

            if ((linear_solve_size_estimate < static_cast<size_t>(max_direct_solve_size)) || (max_direct_solve_size == -1))
            {
              // Small local systems are formed explicitly as dense matrices and
              // solved with a direct backsolve.
              auto phi1A1phi2 = project_operator_A_block(
                phia[ki][ki].cores[i],
                crA({ki, ki}).cores[i],
                phia[ki][ki].cores[i + 1],
                ki,
                i,
                "lr-project-operator");

              auto Bsol_prev = boba::tensor_contraction<3>(
                {"rx", "rx_", "n", "n_", "rxp1", "rxp1_"}, phi1A1phi2, {"rx_", "n_", "rxp1_"}, sol_prev, {"rx", "n", "rxp1"});

              auto norm_lhs_prev = ::boba::norm_frobenius(Bsol_prev);
              if (is_tiny(norm_lhs_prev))
              {
                norm_lhs_prev = norm_rhs;
              }
              res_prev = ::boba::norm_difference_frobenius(boba::flatten(Bsol_prev), rhs_flat);
              res_prev /= norm_lhs_prev;

              boba::permute(
                {"rx", "rx_", "n", "n_", "rxp1", "rxp1_"},
                phi1A1phi2,
                {"rx", "n", "rxp1", "rx_", "n_", "rxp1_"});

              auto B_rows = phi1A1phi2.sizes(0) * phi1A1phi2.sizes(1) * phi1A1phi2.sizes(2);
              B = reshape_to_matrix(phi1A1phi2, {B_rows, B_rows});

              if (res_prev <= real_tol)
              {
                sol = boba::flatten(sol_prev);
              }
              else
              {
                sol = backsolve(B, rhs_flat);
              }

              auto Bsol = B * sol;
              auto norm_lhs_new = ::boba::norm_frobenius(Bsol);
              if (is_tiny(norm_lhs_new))
              {
                norm_lhs_new = norm_rhs;
              }
              res_new = ::boba::norm_difference_frobenius(Bsol, rhs_flat);
              res_new /= norm_lhs_new;
            }
            else
            {
              // Large local systems stay in tensor form and are handled by the
              // iterative 3D two-multilevel local solver.
              auto bfun_sol_prev = bfun3_diag_block(
                phia[ki][ki].cores[i],
                crA({ki, ki}).cores[i],
                phia[ki][ki].cores[i + 1],
                sol_prev,
                ki,
                i,
                "lr-bfun-sol-prev");

              auto norm_lhs_prev = ::boba::norm_frobenius(bfun_sol_prev);
              if (is_tiny(norm_lhs_prev))
              {
                norm_lhs_prev = norm_rhs;
              }
              res_prev = norm_difference_frobenius(bfun_sol_prev, rhs_tensor) / norm_lhs_prev;

              if (!is_tiny(norm_rhs))
              {
                auto sol_tensor = sol_prev;
                if (res_prev > real_tol)
                {
                  sol_tensor = solve3d_2ml(
                    phia[ki][ki].cores[i],
                    crA({ki, ki}).cores[i],
                    phia[ki][ki].cores[i + 1],
                    rhs_tensor,
                    real_tol * norm_rhs,
                    sol_prev,
                    solve3d_2ml_options);
                }
                sol = flatten(sol_tensor);
                auto bfun_sol = bfun3_diag_block(
                  phia[ki][ki].cores[i],
                  crA({ki, ki}).cores[i],
                  phia[ki][ki].cores[i + 1],
                  sol_tensor,
                  ki,
                  i,
                  "lr-bfun-sol");
                auto norm_lhs_new = ::boba::norm_frobenius(bfun_sol);
                if (is_tiny(norm_lhs_new))
                {
                  norm_lhs_new = norm_rhs;
                }
                res_new = norm_difference_frobenius(bfun_sol, rhs_tensor) / norm_lhs_new;
              }
              else
              {
                sol.resize({sol_prev.size()});
                sol.fill_with_zeros();
                res_new = 0.0;
              }
            }
          }

          if (((res_prev / res_new) < resid_damp) && (res_new > real_tol))
          {
            boba_warn("the residual damp was smaller than in the truncation");
          }

          timing_.mark(3);
          // Record the site update size and residual before truncation so the
          // outer sweep can decide when to stop.
          auto sol_diff = sol - boba::flatten(sol_prev);
          auto dx = ::boba::norm_frobenius(sol_diff) / ::boba::norm_frobenius(sol);
          max_dx = boba::max(max_dx, dx);
          max_res = boba::max(max_res, res_prev);

          auto sol_matrix = reshape_to_matrix(sol, {crx(ki).ranks(i) * cry(ki).sizes(i), crx(ki).ranks(i + 1)});

          boba::SVD<space, data_t> svd;
          svd.tolerance_relative = local_solution_svd_tolerance_relative;
          if (ki < local_solution_svd_tolerance_relative_per_block.size() &&
              local_solution_svd_tolerance_relative_per_block[ki] > 0)
          {
            svd.tolerance_relative = local_solution_svd_tolerance_relative_per_block[ki];
          }
          svd.tolerance_absolute = local_solution_svd_tolerance_absolute;

          boba::Matrix<space, data_t> u, s, v;
          index_t r = boba::min(crx(ki).ranks(i) * cry(ki).sizes(i), crx(ki).ranks(i + 1));

          if ((kickrank >= 0) && (i < dimension - 1))
          {
            // Split the updated site solution with an SVD, then back off the
            // rank until the truncated local residual would grow too much.
            svd(sol_matrix);

            r = boba::min(r, svd.U.cols());
            r = boba::min(r, svd.V.cols());

            u = svd.U;
            s = boba::diagonalize(svd.S);
            v = svd.V;
            if (!use_coupled_local_solution)
            {
              for (; r > 1; r--)
              {
                svd.truncate(r);
                auto cursol = svd.reform_matrix();
                data_t res;
                auto linear_system_solution_size = crx(ki).ranks(i) * cry(ki).sizes(i) * crx(ki).ranks(i + 1);
                if ((linear_system_solution_size < static_cast<size_t>(max_direct_solve_size)) || (max_direct_solve_size == -1))
                {
                  auto diff = B * boba::flatten(cursol) - rhs_flat;
                  res = ::boba::norm_frobenius(diff) / norm_rhs;
                }
                else
                {
                  auto s0 = phia[ki][ki].cores[i].sizes(0);
                  auto s1 = crA({ki, ki}).core_cols(i);
                  auto s2 = phia[ki][ki].cores[i + 1].sizes(0);
                  auto cursol_ten = boba::reshape_from_matrix<3>(cursol, {s0, s1, s2});
                  auto bfun_rhs = flatten(bfun3_diag_block(
                    phia[ki][ki].cores[i],
                    crA({ki, ki}).cores[i],
                    phia[ki][ki].cores[i + 1],
                    cursol_ten,
                    ki,
                    i,
                    "lr-bfun-trunc"));
                  bfun_rhs -= rhs_flat;
                  res = ::boba::norm_frobenius(bfun_rhs) / norm_rhs;
                }
                if (res > boba::max(real_tol * resid_damp, res_new))
                {
                  break;
                }
              }

              r += 1;
            }
            r = boba::min(r, s.size());
            r = boba::max(r, size_t{1});
            r = boba::min(r, max_allowed_ranks);
          }
          else
          {
            // At the last site there is no right neighbor to absorb a truncated
            // factor, so keep the full rank via a QR factorization.
            QR<space, data_t> _qr;
            _qr(sol_matrix);
            u = _qr.Q;
            v = _qr.R.transpose();
            r = u.cols();
            auto svec = svd.S;
            svec.resize({r});
            svec.fill_with(1.0);
            s = boba::diagonalize(svec);
          }

          u.resize({u.rows(), r});
          v.resize({v.rows(), r});
          auto svec = boba::diagonalize(s);
          svec.resize({r});
          boba::apply_as_diagonal_right_in_place(svec, v);

          if ((kickrank > 0) && (!last_sweep))
          {
            timing_.mark(4);
            // Recompute the enrichment basis from the post-solve residual so the
            // next site can be expanded in directions missing from the current TT.
            auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx(ki).ranks(i), cry(ki).sizes(i), crx(ki).ranks(i + 1)});

            boba::Tensor<3, space, data_t> crzAt({rz[ki][i], cry(ki).sizes(i), rz[ki][i + 1]});
            crzAt.fill_with_zeros();
            for (size_t kj = 0; kj < Kay; kj++)
            {
              if (crA({ki, kj}).get_number_elements() == 0)
              {
                continue;
              }

              auto x_core = (kj == ki) ? uvT : site_x_cores[kj];
              crzAt += bfun3_block(
                phiza[ki][kj].cores[i],
                crA({ki, kj}).cores[i],
                phiza[ki][kj].cores[i + 1],
                x_core,
                ki,
                kj,
                i,
                "lr-crzAt");
            }

            auto crzy = project_phizy_block(
              phizy[ki].cores[i],
              y1,
              phizy[ki].cores[i + 1],
              ki,
              i,
              "lr-crz-project");
            crzy -= crzAt;
            crznew[ki] = boba::reshape_to_matrix(crzy, {rz[ki][i] * cry(ki).sizes(i), rz[ki][i + 1]});

            ::boba::SVD<space, data_t> _svd;
            _svd.tolerance_relative = residual_enrichment_svd_tolerance_relative;
            _svd.tolerance_absolute = residual_enrichment_svd_tolerance_absolute;
            _svd.max_kept_singular_values = kickrank;
            _svd(crznew[ki]);
            crznew[ki] = _svd.U;

            QR<space, data_t> qr;
            qr(crznew[ki]);
            crznew[ki] = qr.Q;

            rznew[ki] = crznew[ki].cols();
            crznew_tensor[ki] = boba::reshape_from_matrix<3>(crznew[ki], {rz[ki][i], cry(ki).sizes(i), rznew[ki]});
            crz[ki].cores[i] = crznew_tensor[ki];
          }

          if (i < dimension - 1)
          {
            if ((kickrank > 0) && (!last_sweep) && !skip_solution_enrichment)
            {
              // Augment the left factor with residual-based kick vectors, then
              // absorb the resulting basis change into the factor passed right.
              auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx(ki).ranks(i), cry(ki).sizes(i), crx(ki).ranks(i + 1)});

              boba::Tensor<3, space, data_t> leftresid({crx(ki).ranks(i), cry(ki).sizes(i), rz[ki][i + 1]});
              leftresid.fill_with_zeros();
              for (size_t kj = 0; kj < Kay; kj++)
              {
                if (crA({ki, kj}).get_number_elements() == 0)
                {
                  continue;
                }

                auto x_core = (kj == ki) ? uvT : site_x_cores[kj];
                leftresid += bfun3_block(
                  phia[ki][kj].cores[i],
                  crA({ki, kj}).cores[i],
                  phiza[ki][kj].cores[i + 1],
                  x_core,
                  ki,
                  kj,
                  i,
                  "lr-leftresid");
              }

              auto leftresid_mat = boba::reshape_to_matrix(leftresid, {crx(ki).ranks(i) * cry(ki).sizes(i), rz[ki][i + 1]});
              auto lefty_new_temp = project_phizy_block(
                phiy[ki].cores[i],
                y1,
                phizy[ki].cores[i + 1],
                ki,
                i,
                "lr-lefty-project");
              auto lefty_rows = lefty_new_temp.sizes(0) * cry(ki).sizes(i);
              auto lefty_cols = lefty_new_temp.size() / lefty_rows;
              auto lefty_mat = boba::reshape_to_matrix(lefty_new_temp, {lefty_rows, lefty_cols});

              auto u_kick = lefty_mat - leftresid_mat;
              if (inf_sup_enrichment_block == static_cast<int>(ki) && Kay > 1)
              {
                // Projected image of the other blocks: sum_{kj != ki} A_{ki,kj} x_kj.
                boba::Tensor<3, space, data_t> coupling({crx(ki).ranks(i), cry(ki).sizes(i), rz[ki][i + 1]});
                coupling.fill_with_zeros();
                for (size_t kj = 0; kj < Kay; kj++)
                {
                  if (kj == ki || crA({ki, kj}).get_number_elements() == 0)
                  {
                    continue;
                  }
                  coupling += bfun3_block(
                    phia[ki][kj].cores[i],
                    crA({ki, kj}).cores[i],
                    phiza[ki][kj].cores[i + 1],
                    site_x_cores[kj],
                    ki,
                    kj,
                    i,
                    "lr-infsup");
                }
                auto coupling_mat = boba::reshape_to_matrix(coupling, {crx(ki).ranks(i) * cry(ki).sizes(i), rz[ki][i + 1]});
                if (::boba::norm_frobenius(coupling_mat) > 0)
                {
                  ::boba::SVD<space, data_t> csvd;
                  csvd.tolerance_relative = residual_enrichment_svd_tolerance_relative;
                  csvd.tolerance_absolute = residual_enrichment_svd_tolerance_absolute;
                  csvd.max_kept_singular_values = kickrank;
                  csvd(coupling_mat);
                  u_kick = boba::concatenate_columns(u_kick, csvd.U);
                }
              }
              auto u_kicked = boba::concatenate_columns(u, u_kick);

              QR<space, data_t> qr;
              qr(u_kicked);

              u = qr.Q;
              auto rv = qr.R;

              boba::Matrix<space, data_t> zeros({crx(ki).ranks(i + 1), u_kick.cols()});
              zeros.fill_with_zeros();

              auto v_expand = boba::concatenate_columns(v, zeros);
              Tensor<2, space, data_t> v_temp;
              try
              {
                v_temp = boba::tensor_contraction<1>(
                  {"row", "k"}, v_expand, {"col^T", "k"}, rv, {"row", "col^T"});
              }
              catch (const std::exception& error)
              {
                std::ostringstream msg;
                msg << "TensorTrainAMENBlock lr-v-expand failed at site " << i
                    << " for block " << ki
                    << " with v_expand " << v_expand.sizes()
                    << ", rv " << rv.sizes()
                    << ": " << error.what();
                throw std::runtime_error(msg.str());
              }

              v = boba::reshape_to_matrix(v_temp, v_temp.sizes());
            }

            timing_.mark(5);
            // Propagate the updated right factor into the next TT core so the
            // current site becomes left-orthogonal after this update.
            r = u.cols();
            auto cr2 = crx(ki).cores[i + 1];

            Tensor<3, space, data_t> v_new;
            try
            {
              v_new = boba::tensor_contraction<1>(
                {"rxp1", "?"}, v, {"rxp1", "np1", "rxp2"}, cr2, {"?", "np1", "rxp2"});
            }
            catch (const std::exception& error)
            {
              std::ostringstream msg;
              msg << "TensorTrainAMENBlock lr-v-next failed at site " << i
                  << " for block " << ki
                  << " with v " << v.sizes()
                  << ", cr2 " << cr2.sizes()
                  << ": " << error.what();
              throw std::runtime_error(msg.str());
            }

            v = boba::reshape_to_matrix(v_new, {v_new.sizes(0), v_new.sizes(1) * v_new.sizes(2)});

            // Renormalize the propagated factor and store the two updated TT cores.
            if (normalize_intermediate_quantities)
            {
              nrmsc[ki] = nrmsc[ki] / (nrmsy[ki][i] / (nrmsa[ki][ki][i] * nrmsx[ki][i]));
            }

            auto curnorm = ::boba::norm_frobenius(v);
            if (normalize_intermediate_quantities && curnorm > 0)
            {
              v *= (1.0 / curnorm);
            }
            else
            {
              curnorm = 1.0;
            }

            if (normalize_intermediate_quantities)
            {
              nrmsx[ki][i] *= curnorm;
            }

            auto u_tensor = boba::reshape_from_matrix<3>(u, {crx(ki).ranks(i), cry(ki).sizes(i), r});
            auto v_tensor = boba::reshape_from_matrix<3>(v, {r, cry(ki).sizes(i + 1), crx(ki).ranks(i + 2)});

            crx(ki).cores[i] = u_tensor;
            crx(ki).cores[i + 1] = v_tensor;

            if ((kickrank > 0) && (!last_sweep))
            {
              crznew_tensor[ki] = boba::reshape_from_matrix<3>(crznew[ki], {rz[ki][i], cry(ki).sizes(i), rznew[ki]});
              rz[ki][i + 1] = rznew[ki];
            }
          }
          else
          {
            // The final site stores the solved local tensor directly because no
            // further factor propagation is needed.
            crx(ki).cores[i] = boba::reshape<3>(sol, {crx(ki).ranks(i), cry(ki).sizes(i), crx(ki).ranks(i + 1)});
          }
        }

        if (i < dimension - 1)
        {
          timing_.mark(6);
          // Refresh the left-to-right environments so the next site sees the
          // updated solution and enrichment bases.
          for (size_t row = 0; row < Kay; row++)
          {
            for (size_t col = 0; col < Kay; col++)
            {
              if (crA({row, col}).get_number_elements() == 0)
              {
                continue;
              }

              auto phia_external_norm = normalize_intermediate_quantities
                                          ? ((row == col) ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0))
                                          : static_cast<data_t>(1.0);

              std::tie(phia[row][col].cores[i + 1], nrmsa[row][col][i]) =
                compute_next_phi_block<dimension>(
                  phia[row][col].cores[i],
                  crx(row).cores[i],
                  crA({row, col}).cores[i],
                  crx(col).cores[i],
                  true,
                  phia_external_norm,
                  row,
                  col,
                  i,
                  "lr-phia");

              if (row != col || !normalize_intermediate_quantities)
              {
                nrmsa[row][col][i] = 1.0;
              }
            }

            std::tie(phiy[row].cores[i + 1], nrmsy[row][i]) =
              compute_next_phi_block<dimension>(
                phiy[row].cores[i],
                crx(row).cores[i],
                cry(row).cores[i],
                true,
                normalize_intermediate_quantities ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0),
                row,
                row,
                i,
                "lr-phiy");

            if (normalize_intermediate_quantities)
            {
              nrmsc[row] = nrmsc[row] * (nrmsy[row][i] / (nrmsa[row][row][i] * nrmsx[row][i]));
            }
          }

          if ((kickrank > 0) && (!last_sweep))
          {
            for (size_t row = 0; row < Kay; row++)
            {
              for (size_t col = 0; col < Kay; col++)
              {
                if (crA({row, col}).get_number_elements() == 0)
                {
                  continue;
                }
                auto phiza_external_norm = static_cast<data_t>(1.0);
                if (normalize_intermediate_quantities && (row == col))
                {
                  phiza_external_norm = nrmsa[row][col][i];
                }

                std::tie(phiza[row][col].cores[i + 1], std::ignore) =
                  compute_next_phi_block<dimension>(
                    phiza[row][col].cores[i],
                    crz[row].cores[i],
                    crA({row, col}).cores[i],
                    crx(col).cores[i],
                    true,
                    phiza_external_norm,
                    row,
                    col,
                    i,
                    "lr-phiza");
              }

              std::tie(phizy[row].cores[i + 1], std::ignore) =
                compute_next_phi_block<dimension>(
                  phizy[row].cores[i],
                  crz[row].cores[i],
                  cry(row).cores[i],
                  true,
                  normalize_intermediate_quantities ? nrmsy[row][i] : static_cast<data_t>(1.0),
                  row,
                  row,
                  i,
                  "lr-phizy");
            }
          }
        }
      }

      timing_.end_left_to_right();
      data_t raw_global_res = 0.0;
      data_t global_res = 0.0;
      bool have_global_res = false;
      if (verbose && !normalize_intermediate_quantities && (global_res = global_residual(crx)) >= 0)
      {
        raw_global_res = global_res;
        have_global_res = true;
      }
      else if (verbose)
      {
        auto raw_global_residual = apply_nonempty_blocks(crA, crx);
        raw_global_residual -= cry;
        raw_global_res = ::boba::norm_frobenius(raw_global_residual) / norm_cry;

        if (normalize_intermediate_quantities)
        {
          auto crx_rescaled = crx;
          for (size_t ki = 0; ki < Kay; ki++)
          {
            auto scaled_nrmsx = boba::exp(boba::sum(boba::log(nrmsx[ki])) / static_cast<data_t>(dimension));
            for (index_t d = 0; d < dimension; d++)
            {
              crx_rescaled(ki).cores[d] = crx_rescaled(ki).cores[d] * scaled_nrmsx;
            }
          }
          auto rescaled_residual = apply_nonempty_blocks(crA, crx_rescaled);
          rescaled_residual -= cry;
          global_res = ::boba::norm_frobenius(rescaled_residual) / norm_cry;
        }
        else
        {
          // Coupled systems disable intermediate normalization, so nrmsx is
          // identically one and the rescaled residual equals the raw residual.
          global_res = raw_global_res;
        }
        have_global_res = true;
      }

      timing_.report(swp);
      if (verbose)
      {
        std::cout << "TensorTrainAMENBlock: sweep " << swp
                  << ", max_dx: " << max_dx
                  << ", max_res: " << max_res;
        for (size_t ki = 0; ki < Kay; ki++)
        {
          std::cout << ", ranks[" << ki << "]: " << crx(ki).ranks_string();
        }
        if (normalize_intermediate_quantities)
        {
          std::cout << ", raw_global_res: " << raw_global_res;
        }
        std::cout << ", global_res: " << global_res << std::endl;
      }

      if ((swp + 1 >= minimum_sweeps) && (max_res < convergence_tolerance || certify_every_sweep))
      {
        if (!have_global_res)
        {
          auto crx_rescaled = crx;
          for (size_t ki = 0; ki < Kay; ki++)
          {
            auto scaled_nrmsx = boba::exp(boba::sum(boba::log(nrmsx[ki])) / static_cast<data_t>(dimension));
            for (index_t d = 0; d < dimension; d++)
            {
              crx_rescaled(ki).cores[d] = crx_rescaled(ki).cores[d] * scaled_nrmsx;
            }
          }

          const auto t_certify = std::chrono::steady_clock::now();
          global_res = global_residual(crx_rescaled);
          const bool built_in = global_res < 0;
          if (built_in)
          {
            auto tt_global_residual = apply_nonempty_blocks(crA, crx_rescaled);
            tt_global_residual -= cry;
            global_res = ::boba::norm_frobenius(tt_global_residual) / norm_cry;
          }
          if (timing_.enabled)
          {
            std::cout << "TensorTrainAMENBlock timing: certification residual (" << (built_in ? "TT product" : "caller")
                      << ") " << AmenTiming::since(t_certify) << " s, global_res " << global_res << std::endl;
          }
        }

        if (verbose)
        {
          std::cout << "TensorTrainAMENBlock: global_res " << global_res << std::endl;
        }

        if (global_res < convergence_tolerance)
        {
          converged = true;
          break;
        }
        // stagnation: the residual of the last two certified sweeps fell by less than stagnation_reduction
        certified_history.push_back(global_res);
        const size_t h = certified_history.size();
        if (stagnation_reduction > 0 && h >= 3 && certified_history[h - 1] > (1 - stagnation_reduction) * certified_history[h - 3])
        {
          stagnated = true;
          std::cout << "TensorTrainAMENBlock: stopping at sweep " << swp << ", global residual stagnated at " << global_res
                    << " (" << certified_history[h - 3] << " two certified sweeps earlier; tolerance " << convergence_tolerance
                    << ")" << std::endl;
          break;
        }
      }
    }

    if (!converged && !stagnated)
    {
      boba_warn("TensorTrainAMENBlock did not certify convergence before max_sweeps; returning the last iterate.");
    }

    // Rebalance the accumulated core scaling before returning the final block vector.
    checkpoint();
    for (size_t ki = 0; ki < Kay; ki++)
    {
      auto scaled_nrmsx = boba::exp(boba::sum(boba::log(nrmsx[ki])) / static_cast<data_t>(dimension));
      for (index_t d = 0; d < dimension; d++)
      {
        crx(ki).cores[d] = crx(ki).cores[d] * scaled_nrmsx;
      }
    }

    // Coupled systems are already certified against the stacked global residual
    // above. Do not apply a post-solve block-scalar refit here: it can move a
    // converged coupled solution away from the residual-minimizing iterate.

    checkpoint();
    ::boba::BlockVector<::boba::TensorTrain<dimension, host_space, data_t>> host_crx(Kay);
    host_crx.m_name = crx.m_name;
    for (size_t ki = 0; ki < Kay; ki++)
    {
      host_crx(ki) = crx(ki);
    }
    return host_crx;
  }

  /**
   * \brief Solves a block TT system using the process-wide default generator.
   *
   * Calling `boba::random::set_seed()` resets the sequence used for residual
   * enrichment.
   *
   * \tparam dimension Tensor dimension.
   * \tparam space Execution space.
   * \param[in] crA Square block operator.
   * \param[in] cry Block right-hand side.
   * \param[in] tt_initial_guess Initial block TT iterate.
   * \return Final block TT solution, returned in host space.
   */
  template <size_t dimension, execution_space space>
  ::boba::BlockVector<::boba::TensorTrain<dimension, host_space, data_t>> solve(
    const ::boba::BlockOperator<::boba::TensorTrainMatrix<dimension, space, data_t>>& crA,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& cry,
    const ::boba::BlockVector<::boba::TensorTrain<dimension, space, data_t>>& tt_initial_guess)
  {
    auto& random_generator = ::boba::random::default_context();
    return solve(crA, cry, tt_initial_guess, random_generator);
  }

private:
  /// Optional per-sweep timing breakdown, enabled by the environment variable BOBA_AMEN_TIMING (any value).
  struct AmenTiming
  {
    bool enabled = std::getenv("BOBA_AMEN_TIMING") != nullptr;
    using clock = std::chrono::steady_clock;
    clock::time_point sweep_t{}, phase_t{}, seg_t{};
    double right_to_left = 0, left_to_right = 0, solve = 0;
    std::array<double, 7> seg{};
    size_t solves = 0, max_size = 0, total_its = 0, max_its = 0;
    static double since(clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); }
    void start_sweep()
    {
      if (!enabled) return;
      sweep_t = phase_t = seg_t = clock::now();
      right_to_left = left_to_right = solve = 0; seg.fill(0.0); solves = max_size = total_its = max_its = 0;
    }
    void end_right_to_left() { if (enabled) { right_to_left = since(phase_t); phase_t = seg_t = clock::now(); } }
    void end_left_to_right() { if (enabled) { left_to_right = since(phase_t); phase_t = clock::now(); } }
    void mark(int k)   // attribute the time since the previous mark to segment k - 1 (k = 0 starts a site)
    {
      if (!enabled) return;
      const auto now = clock::now();
      if (k > 0) seg[size_t(k - 1)] += std::chrono::duration<double>(now - seg_t).count();
      seg_t = now;
    }
    void add_solve(size_t site, size_t size, size_t its, double seconds)
    {
      if (!enabled) return;
      solve += seconds; ++solves; max_size = std::max(max_size, size); total_its += its; max_its = std::max(max_its, its);
      std::cout << "TensorTrainAMENBlock timing: site " << site << " size " << size << " GMRES its " << its << " "
                << seconds << " s" << std::endl;
    }
    void report(size_t swp) const
    {
      if (!enabled) return;
      std::cout << "TensorTrainAMENBlock timing: sweep " << swp << " total " << since(sweep_t) << " s: right-to-left "
                << right_to_left << ", left-to-right " << left_to_right << " (assemble/project " << seg[0] << ", solve "
                << seg[1] << ", residual diagnostics " << seg[2] << ", SVD truncation " << seg[3] << ", enrichment " << seg[4]
                << ", propagate " << seg[5] << "; environment refresh = rest), global residual " << since(phase_t)
                << "; GMRES solves " << solves << " (" << solve << " s), max site size " << max_size << ", its total "
                << total_its << " max " << max_its << std::endl;
    }
  };
  AmenTiming timing_;
  // nonzero-slice patterns of the operator cores, per (block row, block column, site); the cores do not change
  std::map<std::tuple<size_t, size_t, size_t>, CoreSliceSparsity<data_t>> core_sparsity_cache_;
  const CoreSliceSparsity<data_t>* cached_core_sparsity(const boba::Tensor<4, ::boba::execution_space::CPU, data_t>& A,
                                                         size_t ki, size_t kj, size_t site)
  {
    const auto key = std::make_tuple(ki, kj, site);
    auto it = core_sparsity_cache_.find(key);
    if (it == core_sparsity_cache_.end())
      it = core_sparsity_cache_.emplace(key, core_slice_sparsity(A, sparse_site_operator_max_fraction)).first;
    return &it->second;
  }
  // operator row Grams reused across sweeps (key: block row, block column, site); filled first come, first served
  // (host only: the cache is keyed on host tensors)
  std::map<std::tuple<size_t, size_t, size_t>, boba::Tensor<5, ::boba::execution_space::CPU, data_t>> row_norm_gram_cache_;
  size_t row_norm_gram_cache_used_ = 0;
  const boba::Tensor<5, ::boba::execution_space::CPU, data_t>* cached_operator_row_gram(
    const boba::Tensor<4, ::boba::execution_space::CPU, data_t>& A, size_t ki, size_t kj, size_t site)
  {
    const auto key = std::make_tuple(ki, kj, site);
    auto it = row_norm_gram_cache_.find(key);
    if (it != row_norm_gram_cache_.end()) return &it->second;
    const size_t bytes = size_t(A.sizes(0)) * A.sizes(0) * A.sizes(3) * A.sizes(3) * A.sizes(1) * sizeof(data_t);
    if (row_norm_gram_cache_used_ + bytes > row_norm_gram_cache_bytes) return nullptr;
    row_norm_gram_cache_used_ += bytes;
    return &row_norm_gram_cache_.emplace(key, ::boba::bfun3_operator_row_gram(A)).first->second;
  }

  /// Inverse diagonal of a host coupled site operator: for diagonal block b, entry (i, n, j) is
  /// sum_{ra1, ra2} Phi1(i, i, ra1) A(ra1, n, n, ra2) Phi2(j, ra2, j), times the row factor.  Zero or missing
  /// diagonals give 1.
  static ::boba::Vector<::boba::execution_space::CPU, data_t> site_operator_inverse_diagonal(
    const CoupledSolve3DGMRESOperator<::boba::execution_space::CPU, data_t>& op)
  {
    using MatX = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
    ::boba::Vector<::boba::execution_space::CPU, data_t> dinv({op.offsets.back()});
    data_t* d = dinv.data();
    for (index_t k = 0; k < op.offsets.back(); ++k) d[k] = 0;
    for (size_t b = 0; b + 1 < op.offsets.size(); ++b)
    {
      if (op.operator_cores[b][b] == nullptr) continue;
      const auto& P1 = *op.phi_left[b][b];
      const auto& A = *op.operator_cores[b][b];
      const auto& P2 = *op.phi_right[b][b];
      const index_t Y1 = P1.sizes(0), X1 = P1.sizes(1), R1 = P1.sizes(2);
      const index_t N = A.sizes(1), M = A.sizes(2), R2 = A.sizes(3);
      const index_t X2 = P2.sizes(0), Y2 = P2.sizes(2);
      if (Y1 != X1 || N != M || X2 != Y2) continue;
      MatX p1(Y1, R1), p2(R2, Y2), an(R1, R2);
      for (index_t r = 0; r < R1; ++r) for (index_t i = 0; i < Y1; ++i) p1(i, r) = P1.data()[i + Y1 * (i + X1 * r)];
      for (index_t j = 0; j < Y2; ++j) for (index_t r = 0; r < R2; ++r) p2(r, j) = P2.data()[j + X2 * (r + R2 * j)];
      data_t* db = d + op.offsets[b];
      for (index_t n = 0; n < N; ++n)
      {
        for (index_t r2 = 0; r2 < R2; ++r2) for (index_t r1 = 0; r1 < R1; ++r1) an(r1, r2) = A.data()[r1 + R1 * (n + N * (n + M * r2))];
        const MatX dn = p1 * an * p2;   // (Y1, Y2)
        for (index_t j = 0; j < Y2; ++j) for (index_t i = 0; i < Y1; ++i) db[i + Y1 * (n + N * j)] = dn(i, j);
      }
      const data_t block_factor = op.row_factors.empty() ? data_t{1} : op.row_factors[b];
      const index_t size = op.offsets[b + 1] - op.offsets[b];
      for (index_t k = 0; k < size; ++k)
        db[k] *= op.scalar_row_factors.empty() ? block_factor : op.scalar_row_factors[b].data()[k];
    }
    for (index_t k = 0; k < op.offsets.back(); ++k) d[k] = (d[k] != 0 && std::isfinite(d[k])) ? 1 / d[k] : data_t{1};
    return dinv;
  }

  template <execution_space space_local>
  ::boba::Vector<space_local, data_t> solve_coupled_site_system(
    const CoupledSolve3DGMRESOperator<space_local, data_t>& linear_operator,
    const ::boba::Vector<space_local, data_t>& rhs,
    const ::boba::Vector<space_local, data_t>& initial_guess,
    size_t site)
  {
    const auto site_region = std::string("coupled_site_boba_gmres_") +
                             std::to_string(site) + "_n" +
                             std::to_string(rhs.size());
    BOBA_CALI_BEGIN(site_region.c_str());
    using operator_t = CoupledSolve3DGMRESOperator<space_local, data_t>;
    using vector_t = ::boba::Vector<space_local, data_t>;
    ::boba::Krylov<operator_t, vector_t> solver(
      1, solve3d_2ml_options.eigen_gmres_max_iterations,
      solve3d_2ml_options.tolerance_relative);
    solver.method = ::boba::KrylovMethods::gmres;
    solver.set_matrix(linear_operator);
    solver.verbose = false;
    solver.relative_threshold = solve3d_2ml_options.tolerance_relative;
    solver.absolute_threshold = solve3d_2ml_options.tolerance_absolute;
    if constexpr (space_local == ::boba::execution_space::CPU)
    {
      if (site_jacobi_preconditioner)
      {
        operator_t jacobi;
        jacobi.diagonal = site_operator_inverse_diagonal(linear_operator);
        solver.set_right_preconditioner(jacobi);
      }
    }

    vector_t result(rhs.sizes());
    const auto t_solve = std::chrono::steady_clock::now();
    solver.solve(rhs, initial_guess, result);
    timing_.add_solve(site, rhs.size(), solver.used_inner_iterations,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_solve).count());
    if (verbose)
    {
      auto residual = rhs - linear_operator * result;
      auto rhs_norm = ::boba::norm_frobenius(rhs);
      if (::boba::is_tiny(rhs_norm))
      {
        rhs_norm = 1.0;
      }
      std::cout << "TensorTrainAMENBlock: coupled BoBa GMRES site " << site
                << ", size: " << rhs.size()
                << ", iterations: " << solver.used_inner_iterations
                << ", relative_residual: "
                << ::boba::norm_frobenius(residual) / rhs_norm << "\n";
    }
    BOBA_CALI_END(site_region.c_str());
    return result;
  }

  template <execution_space space_local>
  ::boba::Vector<space_local, data_t> solve_coupled_site_system(
    const ::boba::Matrix<space_local, data_t>& matrix,
    const ::boba::Vector<space_local, data_t>& rhs,
    const ::boba::Vector<space_local, data_t>& initial_guess,
    size_t site)
  {
    const auto site_region = std::string("coupled_site_solve_") + std::to_string(site) +
                             "_n" + std::to_string(matrix.rows());
    BOBA_CALI_BEGIN(site_region.c_str());
    if constexpr (space_local == execution_space::CPU)
    {
      if (use_eigen_gmres_for_coupled_site_solve)
      {
        size_t iterations_used = 0;
        data_t estimated_error = 0.0;
        auto result = ::boba::detail::eigen_gmres(
                        matrix,
                        rhs,
                        solve3d_2ml_options.tolerance_relative,
                        solve3d_2ml_options.eigen_gmres_max_iterations,
                        solve3d_2ml_options.eigen_gmres_sparsity,
                        &initial_guess,
                        &iterations_used,
                        &estimated_error);
        if (verbose)
        {
          std::cout << "TensorTrainAMENBlock: coupled GMRES site " << site
                    << ", size: " << matrix.rows()
                    << ", iterations: " << iterations_used
                    << ", estimated_error: " << estimated_error << "\n";
        }
        BOBA_CALI_END(site_region.c_str());
        return result;
      }
      auto rhs_matrix = reshape_to_matrix(rhs, {rhs.size(), 1});
      ::boba::Matrix<space_local, data_t> output({matrix.cols(), 1});
      if (coupled_site_direct_solver_method == 0)
      {
        ::boba::detail::ls_solve_cod_eigen(matrix, rhs_matrix, output);
      }
      else if (coupled_site_direct_solver_method == 1)
      {
        ::boba::detail::ls_solve_qr_eigen(matrix, rhs_matrix, output);
      }
      else if (coupled_site_direct_solver_method == 3 || coupled_site_direct_solver_method == 4)
      {
        const double kept = ::boba::detail::ls_solve_sparse_lu_eigen(matrix, rhs_matrix, output, 1e-15, coupled_site_dense_fill_threshold,
                                                                     coupled_site_direct_solver_method == 4);
        if (std::getenv("BOBA_AMEN_LOG_SITE_SIZES"))
        {
          std::cout << "TensorTrainAMENBlock: sparse LU site " << site << ", size: " << matrix.rows()
                    << ", fill: " << kept << std::endl;
        }
      }
      else
      {
        boba_always_assert_equal(coupled_site_direct_solver_method, size_t{2}, "Invalid coupled site direct solver method.");
        ::boba::detail::ls_solve_lu_eigen(matrix, rhs_matrix, output);
      }
      auto result = flatten(output);
      BOBA_CALI_END(site_region.c_str());
      return result;
    }
    else
    {
      auto result = backsolve(matrix, rhs);
      BOBA_CALI_END(site_region.c_str());
      return result;
    }
  }

  template <size_t dimension_local, execution_space space_local>
  ::boba::BlockVector<::boba::TensorTrain<dimension_local, space_local, data_t>>
  apply_nonempty_blocks(
    const ::boba::BlockOperator<::boba::TensorTrainMatrix<dimension_local, space_local, data_t>>& crA,
    const ::boba::BlockVector<::boba::TensorTrain<dimension_local, space_local, data_t>>& crx)
  {
    BOBA_CALI_MARK
    ::boba::BlockVector<::boba::TensorTrain<dimension_local, space_local, data_t>> output(crA.block_rows);

    for (size_t row = 0; row < crA.block_rows; row++)
    {
      bool initialized = false;
      for (size_t col = 0; col < crA.block_cols; col++)
      {
        if (crA({row, col}).get_number_elements() == 0)
        {
          continue;
        }

        if (initialized)
        {
          output(row) += crA({row, col}) * crx(col);
        }
        else
        {
          output(row) = crA({row, col}) * crx(col);
          initialized = true;
        }
      }
      boba_always_assert(initialized, "Block AMEn global residual requires at least one nonempty block per row.");
      output(row).round();
    }

    return output;
  }

  /**
   * \brief Update a projected operator environment and attach block/site context to errors.
   *
   * This overload is used for operator environments such as `phia` and
   * `phiza`. It contracts the previous environment with the local block TT
   * core `x`, the local operator core `A_core`, and the matching partner core
   * `y`.
   *
   * \param[in] Phi_prev Previous environment tensor.
   * \param[in] x Current block TT core.
   * \param[in] A_core Local operator core.
   * \param[in] y Local partner core.
   * \param[in] sweep_left_right Direction flag for the environment update.
   * \param[in] external_norm Scaling factor applied during the update.
   * \param[in] row Block row index.
   * \param[in] col Block column index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Updated environment tensor and its normalization factor.
   */
  template <size_t dimension_local, execution_space space_local>
  std::pair<::boba::Tensor<3, space_local, data_t>, data_t> compute_next_phi_block(
    const ::boba::Tensor<3, space_local, data_t>& Phi_prev,
    const ::boba::Tensor<3, space_local, data_t>& x,
    const ::boba::Tensor<4, space_local, data_t>& A_core,
    const ::boba::Tensor<3, space_local, data_t>& y,
    bool sweep_left_right,
    data_t external_norm,
    size_t row,
    size_t col,
    size_t site,
    const char* phase)
  {
    try
    {
      return compute_next_Phi<dimension_local>(Phi_prev, x, A_core, y, sweep_left_right, external_norm);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block (" << row << ", " << col << ")"
          << " with Phi_prev " << Phi_prev.sizes()
          << ", x " << x.sizes()
          << ", A " << A_core.sizes()
          << ", y " << y.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

  /**
   * \brief Update a projected RHS environment and attach block/site context to errors.
   *
   * This overload is used for RHS and enrichment environments such as `phiy`
   * and `phizy`. It contracts the previous environment with the local block TT
   * core `x` and the local RHS core `y`, but does not involve an operator core.
   *
   * \param[in] Phi_prev Previous environment tensor.
   * \param[in] x Current block TT core.
   * \param[in] y Local RHS core.
   * \param[in] sweep_left_right Direction flag for the environment update.
   * \param[in] external_norm Scaling factor applied during the update.
   * \param[in] row Block row index.
   * \param[in] col Block column index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Updated environment tensor and its normalization factor.
   */
  template <size_t dimension_local, execution_space space_local>
  std::pair<::boba::Tensor<3, space_local, data_t>, data_t> compute_next_phi_block(
    const ::boba::Tensor<3, space_local, data_t>& Phi_prev,
    const ::boba::Tensor<3, space_local, data_t>& x,
    const ::boba::Tensor<3, space_local, data_t>& y,
    bool sweep_left_right,
    data_t external_norm,
    size_t row,
    size_t col,
    size_t site,
    const char* phase)
  {
    try
    {
      auto [phi_out, phi_norm] = compute_next_Phi<dimension_local>(Phi_prev, x, y, sweep_left_right, external_norm);
      return std::make_pair(phi_out, phi_norm);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block (" << row << ", " << col << ")"
          << " with Phi_prev " << Phi_prev.sizes()
          << ", x " << x.sizes()
          << ", y " << y.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

public:
  /**
   * \brief Copy a block vector slice into a concatenated dense work vector.
   *
   * \param[in,out] destination Concatenated target vector.
   * \param[in] offset Starting index within the target vector.
   * \param[in] source Slice to copy.
   */
  template <execution_space space_local>
  void copy_vector_block(
    ::boba::Vector<space_local, data_t>& destination,
    index_t offset,
    const ::boba::Vector<space_local, data_t>& source)
  {
    // Copy one block-sized slice into the concatenated dense work vector.
    auto destination_view = destination.view();
    auto source_view = source.const_view();
    ::boba::loop<space_local, 1>(static_cast<index_t>(source.size()),
                                 [=] __boba_host_device__(index_t idx)
    {
      destination_view({offset + idx}) = source_view({idx});
    });
  }

  /**
   * \brief Extract a block-sized slice from a concatenated dense work vector.
   *
   * \param[in] source Concatenated source vector.
   * \param[in] offset Starting index of the slice.
   * \param[in] size Number of elements to extract.
   * \return A vector containing the requested slice.
   */
  template <execution_space space_local>
  ::boba::Vector<space_local, data_t> extract_vector_block(
    const ::boba::Vector<space_local, data_t>& source,
    index_t offset,
    index_t size)
  {
    // Extract one block-sized slice back out of the concatenated work vector.
    ::boba::Vector<space_local, data_t> output({size});
    auto output_view = output.view();
    auto source_view = source.const_view();
    ::boba::loop<space_local, 1>(size,
                                 [=] __boba_host_device__(index_t idx)
    {
      output_view({idx}) = source_view({offset + idx});
    });
    return output;
  }

  /**
   * \brief Convert a squared local block norm into a multiplicative scale.
   *
   * A zero or tiny norm means there is no useful scale information for that
   * block, so the neutral factor is returned.
   */
  data_t make_local_scale_factor(data_t norm_squared) const
  {
    if (norm_squared <= 0.0)
    {
      return 1.0;
    }
    auto norm = ::boba::sqrt(norm_squared);
    if (::boba::is_tiny(norm))
    {
      return 1.0;
    }
    return 1.0 / norm;
  }

  /**
   * \brief Scale one contiguous slice of a dense work vector in place.
   */
  template <execution_space space_local>
  void scale_vector_block(
    ::boba::Vector<space_local, data_t>& vector,
    index_t offset,
    index_t size,
    data_t factor)
  {
    if (::boba::abs(factor - 1.0) <= ::boba::epsilon<data_t>())
    {
      return;
    }
    auto vector_view = vector.view();
    ::boba::loop<space_local, 1>(size,
                                 [=] __boba_host_device__(index_t idx)
    {
      vector_view({offset + idx}) *= factor;
    });
  }

  /**
   * \brief Scale one dense matrix subblock in place.
   */
  template <execution_space space_local>
  void scale_matrix_block(
    ::boba::Matrix<space_local, data_t>& matrix,
    index_t row_begin,
    index_t row_end,
    index_t col_begin,
    index_t col_end,
    data_t factor)
  {
    if (::boba::abs(factor - 1.0) <= ::boba::epsilon<data_t>())
    {
      return;
    }
    auto rows = row_end - row_begin;
    auto cols = col_end - col_begin;
    auto matrix_view = matrix.view();
    ::boba::loop<space_local, 1>(rows * cols,
                                 [=] __boba_host_device__(index_t idx)
    {
      auto row = idx / cols;
      auto col = idx - row * cols;
      matrix_view({row_begin + row, col_begin + col}) *= factor;
    });
  }

private:
  /**
   * \brief Project the RHS environment with block/site context on errors.
   *
   * \param[in] phizy_i Left environment tensor.
   * \param[in] y1 Local RHS core.
   * \param[in] phizy_ip1 Right environment tensor.
   * \param[in] row Block row index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Projected local RHS tensor.
   */
  template <execution_space space_local>
  ::boba::Tensor<3, space_local, data_t> project_phizy_block(
    const ::boba::Tensor<3, space_local, data_t>& phizy_i,
    const ::boba::Tensor<3, space_local, data_t>& y1,
    const ::boba::Tensor<3, space_local, data_t>& phizy_ip1,
    size_t row,
    size_t site,
    const char* phase)
  {
    try
    {
      return project_phizy(phizy_i, y1, phizy_ip1);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block " << row
          << " with phizy_i " << phizy_i.sizes()
          << ", y1 " << y1.sizes()
          << ", phizy_ip1 " << phizy_ip1.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

  /**
   * \brief Contract the projected operator with a block TT core and add context to errors.
   *
   * Block analogue of bfun3 in TensorTrainSolver_utilities.
   *
   * \param[in] Phi1 Left environment tensor.
   * \param[in] A Local operator core.
   * \param[in] Phi2 Right environment tensor.
   * \param[in] x Local block TT core.
   * \param[in] row Block row index.
   * \param[in] col Block column index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Local operator application in tensor form.
   */
  template <execution_space space_local>
  ::boba::Tensor<3, space_local, data_t> bfun3_block(
    const ::boba::Tensor<3, space_local, data_t>& Phi1,
    const ::boba::Tensor<4, space_local, data_t>& A,
    const ::boba::Tensor<3, space_local, data_t>& Phi2,
    const ::boba::Tensor<3, space_local, data_t>& x,
    size_t row,
    size_t col,
    size_t site,
    const char* phase)
  {
    try
    {
      return bfun3(Phi1, A, Phi2, x);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block (" << row << ", " << col << ")"
          << " with Phi1 " << Phi1.sizes()
          << ", A " << A.sizes()
          << ", Phi2 " << Phi2.sizes()
          << ", x " << x.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

  /**
   * \brief Contract the diagonal projected operator with a block TT core and add context to errors.
   *
   * \param[in] Phi1 Left environment tensor.
   * \param[in] A Local operator core.
   * \param[in] Phi2 Right environment tensor.
   * \param[in] x Local block TT core.
   * \param[in] row Block row index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Local diagonal operator application in tensor form.
   */
  template <execution_space space_local>
  ::boba::Tensor<3, space_local, data_t> bfun3_diag_block(
    const ::boba::Tensor<3, space_local, data_t>& Phi1,
    const ::boba::Tensor<4, space_local, data_t>& A,
    const ::boba::Tensor<3, space_local, data_t>& Phi2,
    const ::boba::Tensor<3, space_local, data_t>& x,
    size_t row,
    size_t site,
    const char* phase)
  {
    try
    {
      return bfun3(Phi1, A, Phi2, x);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block " << row
          << " with Phi1 " << Phi1.sizes()
          << ", A " << A.sizes()
          << ", Phi2 " << Phi2.sizes()
          << ", x " << x.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

  /**
   * \brief Form the local six-index operator induced by the left and right environments.
   *
   * \param[in] phia_i Left environment tensor.
   * \param[in] A_core Local operator core.
   * \param[in] phia_ip1 Right environment tensor.
   * \return Local operator tensor before reshaping to a matrix.
   */
  template <execution_space space>
  Tensor<6, space, data_t> project_operator_A(
    const Tensor<3, space, data_t>& phia_i,
    const Tensor<4, space, data_t>& A_core,
    const Tensor<3, space, data_t>& phia_ip1)
  {
    BOBA_CALI_MARK
    checkpoint();
    // First contract the left environment into the operator core.
    auto phi1A1 = ::boba::tensor_contraction<1>(
      {"rx", "rx_", "ra"}, phia_i, {"ra", "row", "col", "rap1"}, A_core, {"rx", "rx_", "row", "col", "rap1"});

    checkpoint();
    // Then contract the right environment to finish the local operator tensor.
    auto phi1A1phi2 = ::boba::tensor_contraction<1>(
      {"rx", "rx_", "row", "col", "rap1"}, phi1A1, {"rxp1", "rap1", "rxp1_"}, phia_ip1, {"rx", "rx_", "row", "col", "rxp1", "rxp1_"});

    return phi1A1phi2;
  }

  /**
   * \brief Project the local operator with block/site context on errors.
   *
   * \param[in] phia_i Left environment tensor.
   * \param[in] A_core Local operator core.
   * \param[in] phia_ip1 Right environment tensor.
   * \param[in] row Block row index.
   * \param[in] site Current TT site.
   * \param[in] phase Short label used in exception messages.
   * \return Local operator tensor before reshaping to a matrix.
   */
  template <execution_space space_local>
  Tensor<6, space_local, data_t> project_operator_A_block(
    const Tensor<3, space_local, data_t>& phia_i,
    const Tensor<4, space_local, data_t>& A_core,
    const Tensor<3, space_local, data_t>& phia_ip1,
    size_t row,
    size_t site,
    const char* phase)
  {
    try
    {
      return project_operator_A(phia_i, A_core, phia_ip1);
    }
    catch (const std::exception& error)
    {
      std::ostringstream msg;
      msg << "TensorTrainAMENBlock " << phase
          << " failed at site " << site
          << " for block " << row
          << " with phia_i " << phia_i.sizes()
          << ", A_core " << A_core.sizes()
          << ", phia_ip1 " << phia_ip1.sizes()
          << ": " << error.what();
      throw std::runtime_error(msg.str());
    }
  }

  /**
   * \brief Project the right-hand side into the current site basis.
   *
   * \param[in] phizy_i Left environment tensor.
   * \param[in] y1 Local RHS core.
   * \param[in] phizy_ip1 Right environment tensor.
   * \return Projected RHS tensor for the current site.
   */
  template <execution_space space>
  Tensor<3, space, data_t> project_phizy(
    const Tensor<3, space, data_t>& phizy_i,
    const Tensor<3, space, data_t>& y1,
    const Tensor<3, space, data_t>& phizy_ip1)
  {
    BOBA_CALI_MARK
    checkpoint();
    // Contract the left environment with the local RHS core first.
    auto phizyy1 = boba::tensor_contraction<1>(
      {"rz", "ry", "one"}, phizy_i, {"ry", "n", "ryp1"}, y1, {"rz", "one", "n", "ryp1"});

    checkpoint();
    // Finish by contracting the right environment and collapsing singleton axes.
    auto crzy = boba::tensor_contraction<1>(
      {"rz", "one", "n", "ryp1"}, phizyy1, {"ryp1", "one_", "rzp1"}, phizy_ip1, {"rz", "one", "n", "one_", "rzp1"});

    auto crzy_final = reshape<3>(crzy, {crzy.sizes(0), crzy.sizes(2), crzy.sizes(4)});

    return crzy_final;
  }
};

} // namespace boba
