// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace boba
{

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
  /// \brief Enable per-sweep progress output from `solve()`.
  bool verbose = false;

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
    BOBA_CALI_MARK
    checkpoint();

    boba_always_assert_equal(crA.block_rows, crA.block_cols, "To use Block AMEn, the operator must be square");
    boba_always_assert_equal(cry.block_size, crA.block_rows, "RHS block size must match operator block size");
    boba_always_assert_equal(tt_initial_guess.block_size, crA.block_rows, "Initial guess block size must match operator block size");

    size_t K = crA.block_rows;
    bool has_offdiag_blocks = false;

    // Validate the block layout and detect whether any off-diagonal couplings are present.
    for (size_t ki = 0; ki < K; ki++)
    {
      for (size_t kj = 0; kj < K; kj++)
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

    checkpoint();

    // Allocate the AMEn environments and the enrichment tensors for each block row/column pair.
    std::vector<std::vector<::boba::TensorTrain<dimension + 1, space, data_t>>> phia(K);
    std::vector<::boba::TensorTrain<dimension + 1, space, data_t>> phiy(K);
    std::vector<std::vector<::boba::TensorTrain<dimension + 1, space, data_t>>> phiza(K);
    std::vector<::boba::TensorTrain<dimension + 1, space, data_t>> phizy(K);
    std::vector<::boba::TensorTrain<dimension, space, data_t>> crz(K);

    checkpoint();
    for (size_t ki = 0; ki < K; ki++)
    {
      phia[ki].resize(K);
      phiza[ki].resize(K);

      for (size_t kj = 0; kj < K; kj++)
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
        auto rank_left = (d == 0) ? 1 : kickrank;
        auto rank_right = (d == dimension - 1) ? 1 : kickrank;

        crz[ki].cores[d].resize({rank_left, cry(ki).sizes(d), rank_right});
        crz[ki].cores[d].fill_with_random();
      }
    }

    checkpoint();

    // Track per-interface normalization factors so the left/right environment
    // contractions stay numerically stable across sweeps.
    std::vector<::boba::Array<data_t, dimension - 1>> nrmsx(K);
    std::vector<std::vector<::boba::Array<data_t, dimension - 1>>> nrmsa(K);
    std::vector<::boba::Array<data_t, dimension - 1>> nrmsy(K);
    std::vector<data_t> nrmsc(K, 1.0);
    std::vector<::boba::Array<size_t, dimension + 1>> rz(K);
    std::vector<index_t> rznew(K, 0);
    std::vector<boba::Matrix<space, data_t>> crznew(K);
    std::vector<boba::Tensor<3, space, data_t>> crznew_tensor(K);

    for (size_t ki = 0; ki < K; ki++)
    {
      nrmsx[ki] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      nrmsy[ki] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      nrmsa[ki].resize(K);
      for (size_t kj = 0; kj < K; kj++)
      {
        nrmsa[ki][kj] = ::boba::filled_array<dimension - 1, data_t>(1.0);
      }
      rz[ki] = crz[ki].ranks();
    }

    checkpoint();
    size_t max_sweep = 20;

    // Main AMEn sweeps: first move right-to-left to update environments and enrichment data.
    bool last_sweep = false;
    for (size_t swp = 0; swp < max_sweep; swp++)
    {
      for (size_t ip1 = dimension; ip1 > 1; ip1--)
      {
        size_t i = ip1 - 1;
        // Snapshot the current site cores before any block is updated so
        // off-diagonal couplings at this site all see a consistent iterate.
        std::vector<boba::Tensor<3, space, data_t>> site_x_cores(K);
        for (size_t kj = 0; kj < K; kj++)
        {
          site_x_cores[kj] = crx(kj).cores[i];
        }

        for (size_t ki = 0; ki < K; ki++)
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

              for (size_t kj = 0; kj < K; kj++)
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

              auto crzAt_matrix = boba::reshape_to_matrix(crzAt, {rz[ki][i], cry(ki).sizes(i) * rz[ki][i + 1]});
              auto crzy2 = project_phizy_block(
                phizy[ki].cores[i],
                cry(ki).cores[i],
                phizy[ki].cores[i + 1],
                ki,
                i,
                "rl-project_phizy");
              crzy2 *= nrmsc[ki];
              auto crzy2_matrix = boba::reshape_to_matrix(crzy2, {rz[ki][i], cry(ki).sizes(i) * rz[ki][i + 1]});
              crznew[ki] = crzy2_matrix - crzAt_matrix;

              ::boba::SVD<space, data_t> svd;
              svd.tolerance_relative = 0.0;
              svd.tolerance_absolute = 0.0;
              svd(crznew[ki]);
              crznew[ki] = svd.V;

              auto fetch_col = boba::min(kickrank, crznew[ki].cols());
              auto crznew_temp = crznew[ki];
              crznew[ki] = crznew_temp.get_submatrix({0, crznew[ki].rows()}, {0, fetch_col});
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
            nrmsc[ki] = nrmsc[ki] / (nrmsy[ki][i - 1] / (nrmsa[ki][ki][i - 1] * nrmsx[ki][i - 1]));
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
          if (!::boba::is_tiny(curnorm))
          {
            new_cr2mat *= (1.0 / curnorm);
          }
          else
          {
            curnorm = 1;
          }

          nrmsx[ki][i - 1] *= curnorm;
          crx(ki).cores[i - 1] = new_cr2mat;
          crx(ki).cores[i] = cr;
        }

        // Update right-to-left operator and RHS environments after every block
        // has contributed its newly orthogonalized site core.
        for (size_t row = 0; row < K; row++)
        {
          for (size_t col = 0; col < K; col++)
          {
            if (crA({row, col}).get_number_elements() == 0)
            {
              continue;
            }
            auto phia_external_norm = (row == col) ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0);

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

            if (row != col)
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
              -1.0,
              row,
              row,
              i,
              "rl-phiy");

          nrmsc[row] = nrmsc[row] * (nrmsy[row][i - 1] / (nrmsa[row][row][i - 1] * nrmsx[row][i - 1]));
        }

        if ((kickrank > 0) && (!last_sweep))
        {
          // Mirror the same environment propagation for the enrichment tensor z.
          for (size_t row = 0; row < K; row++)
          {
            rz[row][i] = rznew[row];

            for (size_t col = 0; col < K; col++)
            {
              if (crA({row, col}).get_number_elements() == 0)
              {
                continue;
              }

              auto phiza_external_norm = (row == col) ? nrmsa[row][col][i - 1] : static_cast<data_t>(1.0);

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
                nrmsy[row][i - 1],
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

      // Sweep left-to-right, solve the local problems, and adapt the TT ranks.
      for (size_t i = 0; i < dimension; i++)
      {
        std::vector<boba::Tensor<3, space, data_t>> site_x_cores(K);
        for (size_t kj = 0; kj < K; kj++)
        {
          site_x_cores[kj] = crx(kj).cores[i];
        }

        // Temporary storage for the site-local linear system. For diagonal block
        // systems each block is solved independently; otherwise all blocks are
        // stacked into one coupled dense solve at this site.
        std::vector<boba::Vector<space, data_t>> coupled_rhs(K);
        std::vector<boba::Vector<space, data_t>> coupled_sol(K);
        std::vector<data_t> coupled_norm_rhs(K, 0.0);
        std::vector<data_t> coupled_res_prev(K, 0.0);
        std::vector<data_t> coupled_res_new(K, 0.0);
        std::vector<index_t> coupled_offsets(K + 1, 0);
        auto use_coupled_site_solve = has_offdiag_blocks;

        // For coupled systems, assemble one local block system spanning all blocks at this site.
        if (use_coupled_site_solve)
        {
          for (size_t ki = 0; ki < K; ki++)
          {
            // Project each block RHS into the current local basis so every block
            // contributes one contiguous segment to the coupled site system.
            auto y1 = cry(ki).cores[i];
            y1 *= nrmsc[ki];

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

          auto total_local_size = coupled_offsets[K];
          boba::Matrix<space, data_t> coupled_B({total_local_size, total_local_size});
          coupled_B.fill_with_zeros();

          boba::Vector<space, data_t> coupled_rhs_full({total_local_size});
          coupled_rhs_full.fill_with_zeros();

          boba::Vector<space, data_t> coupled_sol_prev_full({total_local_size});
          coupled_sol_prev_full.fill_with_zeros();

          for (size_t ki = 0; ki < K; ki++)
          {
            // Assemble the dense site matrix block-by-block from the projected
            // TT operator environments and the old local iterate.
            copy_vector_block(coupled_rhs_full, coupled_offsets[ki], coupled_rhs[ki]);
            copy_vector_block(coupled_sol_prev_full, coupled_offsets[ki], flatten(site_x_cores[ki]));

            for (size_t kj = 0; kj < K; kj++)
            {
              if (crA({ki, kj}).get_number_elements() == 0)
              {
                continue;
              }

              auto B_block = bfun3_matrix(
                phia[ki][kj].cores[i],
                crA({ki, kj}).cores[i],
                phia[ki][kj].cores[i + 1]);

              coupled_B.replace_submatrix(
                {coupled_offsets[ki], coupled_offsets[ki + 1]},
                {coupled_offsets[kj], coupled_offsets[kj + 1]},
                B_block);
            }
          }

          // Solve the coupled site problem once, then slice the solution and
          // residual diagnostics back into the per-block containers.
          auto coupled_prev_residual = coupled_B * coupled_sol_prev_full - coupled_rhs_full;
          auto coupled_sol_full = backsolve(coupled_B, coupled_rhs_full);
          auto coupled_new_residual = coupled_B * coupled_sol_full - coupled_rhs_full;

          for (size_t ki = 0; ki < K; ki++)
          {
            auto local_size = coupled_offsets[ki + 1] - coupled_offsets[ki];
            coupled_sol[ki] = extract_vector_block(coupled_sol_full, coupled_offsets[ki], local_size);

            auto prev_residual = extract_vector_block(coupled_prev_residual, coupled_offsets[ki], local_size);
            auto new_residual = extract_vector_block(coupled_new_residual, coupled_offsets[ki], local_size);

            if (!is_tiny(coupled_norm_rhs[ki]))
            {
              coupled_res_prev[ki] = ::boba::norm_frobenius(prev_residual) / coupled_norm_rhs[ki];
              coupled_res_new[ki] = ::boba::norm_frobenius(new_residual) / coupled_norm_rhs[ki];
            }
          }
        }

        for (size_t ki = 0; ki < K; ki++)
        {
          auto y1 = cry(ki).cores[i];
          y1 *= nrmsc[ki];

          boba::Vector<space, data_t> rhs_flat;
          data_t norm_rhs = 0.0;
          auto real_tol = (convergence_tolerance / ::boba::sqrt(static_cast<data_t>(dimension))) / resid_damp;

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
            res_prev = coupled_res_prev[ki];
            res_new = coupled_res_new[ki];
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

            for (size_t kj = 0; kj < K; kj++)
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

              res_prev = ::boba::norm_difference_frobenius(boba::flatten(Bsol_prev), rhs_flat);
              res_prev /= norm_rhs;

              boba::permute(
                {"rx", "rx_", "n", "n_", "rxp1", "rxp1_"},
                phi1A1phi2,
                {"rx", "n", "rxp1", "rx_", "n_", "rxp1_"});

              auto B_rows = phi1A1phi2.sizes(0) * phi1A1phi2.sizes(1) * phi1A1phi2.sizes(2);
              B = reshape_to_matrix(phi1A1phi2, {B_rows, B_rows});

              sol = backsolve(B, rhs_flat);

              res_new = ::boba::norm_difference_frobenius(B * sol, rhs_flat);
              res_new /= norm_rhs;
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

              res_prev = norm_difference_frobenius(bfun_sol_prev, rhs_tensor) / norm_rhs;

              if (!is_tiny(norm_rhs))
              {
                auto sol_tensor = solve3d_2ml(
                  phia[ki][ki].cores[i],
                  crA({ki, ki}).cores[i],
                  phia[ki][ki].cores[i + 1],
                  rhs_tensor,
                  real_tol * norm_rhs,
                  sol_prev,
                  solve3d_2ml_options);
                sol = flatten(sol_tensor);
                auto bfun_sol = bfun3_diag_block(
                  phia[ki][ki].cores[i],
                  crA({ki, ki}).cores[i],
                  phia[ki][ki].cores[i + 1],
                  sol_tensor,
                  ki,
                  i,
                  "lr-bfun-sol");
                res_new = norm_difference_frobenius(bfun_sol, rhs_tensor) / norm_rhs;
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

          // Record the site update size and residual before truncation so the
          // outer sweep can decide when to stop.
          auto sol_diff = sol - boba::flatten(sol_prev);
          auto dx = ::boba::norm_frobenius(sol_diff) / ::boba::norm_frobenius(sol);
          max_dx = boba::max(max_dx, dx);
          max_res = boba::max(max_res, res_prev);

          auto sol_matrix = reshape_to_matrix(sol, {crx(ki).ranks(i) * cry(ki).sizes(i), crx(ki).ranks(i + 1)});

          boba::SVD<space, data_t> svd;
          svd.tolerance_relative = 0.0;
          svd.tolerance_absolute = 0.0;

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
            r = boba::min(r, s.size() - 1);
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
            // Recompute the enrichment basis from the post-solve residual so the
            // next site can be expanded in directions missing from the current TT.
            auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx(ki).ranks(i), cry(ki).sizes(i), crx(ki).ranks(i + 1)});

            boba::Tensor<3, space, data_t> crzAt({rz[ki][i], cry(ki).sizes(i), rz[ki][i + 1]});
            crzAt.fill_with_zeros();
            for (size_t kj = 0; kj < K; kj++)
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

            auto crzAt_matrix = boba::reshape_to_matrix(crzAt, {rz[ki][i] * cry(ki).sizes(i), rz[ki][i + 1]});
            auto crzy = project_phizy_block(
              phizy[ki].cores[i],
              y1,
              phizy[ki].cores[i + 1],
              ki,
              i,
              "lr-crz-project");
            auto crzy_matrix = boba::reshape_to_matrix(crzy, {rz[ki][i] * cry(ki).sizes(i), rz[ki][i + 1]});
            crznew[ki] = crzy_matrix - crzAt_matrix;

            ::boba::SVD<space, data_t> _svd;
            _svd.tolerance_relative = 0.0;
            _svd.tolerance_absolute = 0.0;
            _svd(crznew[ki]);
            crznew[ki] = _svd.U;

            auto fetch_col = boba::min(kickrank, crznew[ki].cols());
            auto crznew_temp = crznew[ki];
            crznew[ki] = crznew_temp.get_submatrix({0, crznew[ki].rows()}, {0, fetch_col});

            QR<space, data_t> qr;
            qr(crznew[ki]);
            crznew[ki] = qr.Q;

            rznew[ki] = crznew[ki].cols();
            crznew_tensor[ki] = boba::reshape_from_matrix<3>(crznew[ki], {rz[ki][i], cry(ki).sizes(i), rznew[ki]});
            crz[ki].cores[i] = crznew_tensor[ki];
          }

          if (i < dimension - 1)
          {
            if ((kickrank > 0) && (!last_sweep))
            {
              // Augment the left factor with residual-based kick vectors, then
              // absorb the resulting basis change into the factor passed right.
              auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx(ki).ranks(i), cry(ki).sizes(i), crx(ki).ranks(i + 1)});

              boba::Tensor<3, space, data_t> leftresid({crx(ki).ranks(i), cry(ki).sizes(i), rz[ki][i + 1]});
              leftresid.fill_with_zeros();
              for (size_t kj = 0; kj < K; kj++)
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
            nrmsc[ki] = nrmsc[ki] / (nrmsy[ki][i] / (nrmsa[ki][ki][i] * nrmsx[ki][i]));

            auto curnorm = ::boba::norm_frobenius(v);
            if (curnorm > 0)
            {
              v *= (1.0 / curnorm);
            }
            else
            {
              curnorm = 1.0;
            }

            nrmsx[ki][i] *= curnorm;

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
          // Refresh the left-to-right environments so the next site sees the
          // updated solution and enrichment bases.
          for (size_t row = 0; row < K; row++)
          {
            for (size_t col = 0; col < K; col++)
            {
              if (crA({row, col}).get_number_elements() == 0)
              {
                continue;
              }

              auto phia_external_norm = (row == col) ? static_cast<data_t>(-1.0) : static_cast<data_t>(1.0);

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

              if (row != col)
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
                -1.0,
                row,
                row,
                i,
                "lr-phiy");

            nrmsc[row] = nrmsc[row] * (nrmsy[row][i] / (nrmsa[row][row][i] * nrmsx[row][i]));
          }

          if ((kickrank > 0) && (!last_sweep))
          {
            for (size_t row = 0; row < K; row++)
            {
              for (size_t col = 0; col < K; col++)
              {
                if (crA({row, col}).get_number_elements() == 0)
                {
                  continue;
                }
                auto phiza_external_norm = (row == col) ? nrmsa[row][col][i] : static_cast<data_t>(1.0);

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
                  nrmsy[row][i],
                  row,
                  row,
                  i,
                  "lr-phizy");
            }
          }
        }
      }

      if (verbose)
      {
        std::cout << "TensorTrainAMENBlock: sweep " << swp
                  << ", max_dx: " << max_dx
                  << ", max_res: " << max_res;
        for (size_t ki = 0; ki < K; ki++)
        {
          std::cout << ", ranks[" << ki << "]: " << crx(ki).ranks_string();
        }
        std::cout << std::endl;
      }

      // Convergence is driven by the largest pre-truncation local residual seen
      // during the sweep.
      if (max_res < convergence_tolerance)
      {
        break;
      }
    }

    // Rebalance the accumulated core scaling before the final post-processing step.
    checkpoint();
    for (size_t ki = 0; ki < K; ki++)
    {
      auto scaled_nrmsx = boba::exp(boba::sum(boba::log(nrmsx[ki])) / static_cast<data_t>(dimension));
      for (index_t d = 0; d < dimension; d++)
      {
        crx(ki).cores[d] = crx(ki).cores[d] * scaled_nrmsx;
      }
    }

    // For coupled systems, finish with a small dense block-scalar correction.
    if (has_offdiag_blocks)
    {
      // Build A*x block-by-block and solve a small dense least-squares-like
      // correction so the block components are scaled consistently.
      std::vector<std::vector<::boba::TensorTrain<dimension, space, data_t>>> block_images(K);
      for (size_t row = 0; row < K; row++)
      {
        block_images[row].resize(K);
        for (size_t col = 0; col < K; col++)
        {
          if (crA({row, col}).get_number_elements() == 0)
          {
            continue;
          }
          block_images[row][col] = crA({row, col}) * crx(col);
        }
      }

      boba::Matrix<host_space, data_t> scalar_system({static_cast<index_t>(K), static_cast<index_t>(K)});
      scalar_system.fill_with_zeros();
      boba::Vector<host_space, data_t> scalar_rhs({static_cast<index_t>(K)});
      scalar_rhs.fill_with_zeros();

      for (size_t col_p = 0; col_p < K; col_p++)
      {
        data_t rhs_entry = 0.0;
        for (size_t row = 0; row < K; row++)
        {
          if (block_images[row][col_p].get_number_elements() == 0)
          {
            continue;
          }
          rhs_entry += ::boba::inner_product(block_images[row][col_p], cry(row));
        }
        scalar_rhs({static_cast<index_t>(col_p)}) = rhs_entry;

        for (size_t col_q = 0; col_q < K; col_q++)
        {
          data_t system_entry = 0.0;
          for (size_t row = 0; row < K; row++)
          {
            if ((block_images[row][col_p].get_number_elements() == 0) || (block_images[row][col_q].get_number_elements() == 0))
            {
              continue;
            }
            system_entry += ::boba::inner_product(block_images[row][col_p], block_images[row][col_q]);
          }
          scalar_system({static_cast<index_t>(col_p), static_cast<index_t>(col_q)}) = system_entry;
        }
      }

      auto block_scalars = backsolve(scalar_system, scalar_rhs);
      auto block_scalars_view = block_scalars.const_view();
      for (size_t col = 0; col < K; col++)
      {
        // Applying the scalar to the first core scales the full TT block vector.
        crx(col).cores[0] *= block_scalars_view(static_cast<index_t>(col));
      }
    }

    checkpoint();
    ::boba::BlockVector<::boba::TensorTrain<dimension, host_space, data_t>> host_crx(K);
    host_crx.m_name = crx.m_name;
    for (size_t ki = 0; ki < K; ki++)
    {
      host_crx(ki) = crx(ki);
    }
    return host_crx;
  }

private:
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
