// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace boba
{

template <typename data_t>
struct TensorTrainAMEN
{
  /*
    This class is a reimplements Oseledets' amen_solve2 using the BoBa framework
    See https://github.com/oseledets/TT-Toolbox/blob/master/solve/amen_solve2.m
  */

  data_t convergence_tolerance = 1.0e-10;
  index_t kickrank = 4;
  data_t resid_damp = 2.0;
  int max_direct_solve_size = 1500;
  index_t max_allowed_ranks = 1000;

  Solve3D2MLOptions<data_t> solve3d_2ml_options;

  template <typename unsupported_operator, typename unsupported_vector>
  unsupported_vector solve(
    const unsupported_operator& crA,
    const unsupported_vector& cry,
    const unsupported_vector& tt_initial_guess)
  {
    detail::ignore(crA);
    detail::ignore(cry);
    detail::ignore(tt_initial_guess);
    boba_error("AMEN not yet supported for this operator/vector combination.");
    return cry;
  }

  template <size_t dimension, execution_space space>
  ::boba::TensorTrain<dimension, host_space, data_t> solve(
    const ::boba::TensorTrainMatrix<dimension, space, data_t>& crA,
    const ::boba::TensorTrain<dimension, space, data_t>& cry,
    const ::boba::TensorTrain<dimension, space, data_t>& tt_initial_guess)
  {
    BOBA_CALI_MARK
    checkpoint();

    boba_always_assert_equal(crA.core_rows(), crA.core_cols(), "To use AMEN with rectangular systems, first form the normal equations  A^TAx = A^Tb and solve that system instead.");

    auto crx = tt_initial_guess;

    checkpoint();
    ::boba::TensorTrain<dimension + 1, space, data_t> phia(::boba::filled_array<dimension + 1>(1_z));
    ::boba::TensorTrain<dimension + 1, space, data_t> phiy(::boba::filled_array<dimension + 1>(1_z));

    ::boba::TensorTrain<dimension + 1, space, data_t> phiza(::boba::filled_array<dimension + 1>(1_z));
    ::boba::TensorTrain<dimension + 1, space, data_t> phizy(::boba::filled_array<dimension + 1>(1_z));

    checkpoint();
    ::boba::TensorTrain<dimension, space, data_t> crz(cry.sizes());

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      auto rank_left = (d == 0) ? 1 : kickrank;
      auto rank_right = (d == dimension - 1) ? 1 : kickrank;
      checkpoint();

      crz.cores[d].resize({rank_left, cry.sizes(d), rank_right});
      checkpoint();
      crz.cores[d].fill_with_random();
    }

    checkpoint();

    auto rz = crz.ranks();

    checkpoint();
    phiza.fill_with(1);
    phizy.fill_with(1);
    phia.fill_with(1);
    phiy.fill_with(1);

    auto nrmsx = ::boba::filled_array<dimension - 1, data_t>(1.0);
    auto nrmsa = ::boba::filled_array<dimension - 1, data_t>(1.0);
    auto nrmsy = ::boba::filled_array<dimension - 1, data_t>(1.0);

    checkpoint();
    size_t max_sweep = 20;

    data_t nrmsc = 1.0;
    size_t rznew = 0;

    boba::Tensor<3, space, data_t> crznew_tensor;

    // AMEn sweeps
    bool last_sweep = false;
    for (size_t swp = 0; swp < max_sweep; swp++)
    {
      boba::Matrix<space, data_t> crznew;

      for (size_t i = dimension - 1; i > 0; i--)
      {
        // Update the Z in the ALS version
        checkpoint();
        if ((kickrank > 0) and (not(last_sweep)))
        {
          bool propagate_z_factor = false;
          auto current_z_right_rank = rz[i + 1];
          if (swp > 0)
          {
            // Update crz (we dont want just random-svd)
            checkpoint();
            auto crxi = crx.cores[i];
            auto crzAt = bfun3(phiza.cores[i], crA.cores[i], phiza.cores[i + 1], crxi);
            auto crzAt_matrix = boba::reshape_to_matrix(crzAt, {rz[i], cry.sizes(i) * rz[i + 1]});
            checkpoint();
            auto crzy2 = project_phizy(phizy.cores[i], cry.cores[i], phizy.cores[i + 1]);
            crzy2 *= nrmsc;
            auto crzy2_matrix = boba::reshape_to_matrix(crzy2, {rz[i], cry.sizes(i) * rz[i + 1]});
            crznew = crzy2_matrix - crzAt_matrix;

            ::boba::SVD<space, data_t> svd;
            svd.tolerance_relative = 0.0;
            svd.tolerance_absolute = 0.0;
            svd(crznew);
            crznew = svd.V;

            checkpoint();
            {
              auto fetch_col = boba::min(kickrank, crznew.cols());
              auto crznew_temp = crznew;
              crznew = crznew_temp.get_submatrix({0, crznew.rows()}, {0, fetch_col});
            }
          }
          else
          {
            checkpoint();
            propagate_z_factor = true;
            auto crznew_permute = crz.cores[i];
            current_z_right_rank = crznew_permute.sizes(2);
            crznew = boba::reshape_to_matrix(
                       crznew_permute,
                       {crznew_permute.sizes(0), crznew_permute.sizes(1) * crznew_permute.sizes(2)})
                       .transpose();
          }

          checkpoint();
          QR<space, data_t> qr;

          qr(crznew);
          crznew = qr.Q;
          auto rv = qr.R;

          checkpoint();
          rznew = crznew.cols();
          if ((swp == 0) && propagate_z_factor)
          {
            auto crz_prev = crz.cores[i - 1];
            crz.cores[i - 1] = ::boba::tensor_contraction<1>(
              {"rzm1", "n", "rz"}, crz_prev, {"k", "rz"}, rv, {"rzm1", "n", "k"});
          }
          crz.cores[i] = boba::reshape_from_matrix<3>(crznew.transpose(), {rznew, cry.sizes(i), current_z_right_rank});
          checkpoint();
        }
        if (swp > 0)
        {
          // Remove old norm correction
          nrmsc = nrmsc / (nrmsy[i - 1] / (nrmsa[i - 1] * nrmsx[i - 1]));
        }

        checkpoint();
        auto cr = crx.cores[i];
        auto unfold = ::boba::compute_unfold_right(cr).transpose();

        checkpoint();
        QR<space, data_t> qr;
        qr(unfold);
        cr = boba::reshape_from_matrix<3>(qr.Q.transpose(), {qr.Q.cols(), crx.sizes(i), crx.ranks(i + 1)});
        auto rv = qr.R;

        checkpoint();
        auto cr2 = crx.cores[i - 1];

        auto new_cr2mat = ::boba::tensor_contraction<1>(
          {"rxm1", "n", "rx"}, cr2, {"k", "rx"}, rv, {"rxm1", "n", "k"});

        checkpoint();
        auto curnorm = ::boba::norm_frobenius(new_cr2mat);

        if (not(::boba::is_tiny(curnorm)))
        {
          new_cr2mat *= (1.0 / curnorm);
        }
        else
        {
          curnorm = 1;
        }

        checkpoint();

        nrmsx[i - 1] *= curnorm;
        crx.cores[i - 1] = new_cr2mat;
        crx.cores[i] = cr;

        checkpoint();
        std::tie(phia.cores[i], nrmsa[i - 1]) = compute_next_Phi<dimension>(phia.cores[i + 1], cr, crA.cores[i], cr, false, -1.0);
        std::tie(phiy.cores[i], nrmsy[i - 1]) = compute_next_Phi<dimension>(phiy.cores[i + 1], cr, cry.cores[i], false, -1.0);

        // Add new scales
        nrmsc = nrmsc * (nrmsy[i - 1] / (nrmsa[i - 1] * nrmsx[i - 1]));

        if ((kickrank > 0) and (not(last_sweep)))
        {
          rz[i] = rznew;
          std::tie(phiza.cores[i], std::ignore) = compute_next_Phi<dimension>(phiza.cores[i + 1], crz.cores[i], crA.cores[i], crx.cores[i], false, nrmsa[i - 1]);
          std::tie(phizy.cores[i], std::ignore) = compute_next_Phi<dimension>(phizy.cores[i + 1], crz.cores[i], cry.cores[i], false, nrmsy[i - 1]);
        }
      }
      checkpoint();

      data_t max_res = 0.0;
      data_t max_dx = 0.0;

      checkpoint();

      for (size_t i = 0; i < dimension; i++)
      {
        // Extract partial projections (and scales)
        auto y1 = cry.cores[i];

        // Rescale the RHS
        y1 *= nrmsc;

        auto rhs_tensor = project_phizy(phiy.cores[i], y1, phiy.cores[i + 1]);
        auto rhs_flat = flatten(rhs_tensor);

        auto norm_rhs = ::boba::norm_frobenius(rhs_flat);

        // We need slightly better accuracy for the solution, since otherwise
        // the truncation will catch the noise and report the full rank
        auto real_tol = (convergence_tolerance / ::boba::sqrt(dimension)) / resid_damp;
        checkpoint();

        data_t res_new, res_prev;
        boba::Vector<space, data_t> sol;
        boba::Matrix<space, data_t> B;
        auto sol_prev = crx.cores[i];

        auto linear_solve_size_estimate = crx.ranks(i) * crx.sizes(i) * crx.ranks(i + 1);
        if ((linear_solve_size_estimate < static_cast<size_t>(max_direct_solve_size)) or (max_direct_solve_size == -1)) // Full solution
        {
          checkpoint();
          auto phi1A1phi2 = project_operator_A(phia.cores[i], crA.cores[i], phia.cores[i + 1]);

          checkpoint();
          auto Bsol_prev = boba::tensor_contraction<3>(
            {"rx", "rx_", "n", "n_", "rxp1", "rxp1_"}, phi1A1phi2, {"rx_", "n_", "rxp1_"}, sol_prev, {"rx", "n", "rxp1"});

          res_prev = ::boba::norm_difference_frobenius(boba::flatten(Bsol_prev), rhs_flat);
          res_prev /= norm_rhs;

          boba::permute(
            {"rx", "rx_", "n", "n_", "rxp1", "rxp1_"},
            phi1A1phi2,
            {"rx", "n", "rxp1", "rx_", "n_", "rxp1_"});

          checkpoint();
          auto B_rows = phi1A1phi2.sizes(0) * phi1A1phi2.sizes(1) * phi1A1phi2.sizes(2);
          B = reshape_to_matrix(phi1A1phi2, {B_rows, B_rows});

          checkpoint();
          sol = backsolve(B, rhs_flat);

          res_new = ::boba::norm_difference_frobenius(B * sol, rhs_flat);
          res_new /= norm_rhs;
        }
        else // Structured Solution
        {
          checkpoint();
          auto bfun_sol_prev = bfun3(phia.cores[i], crA.cores[i], phia.cores[i + 1], sol_prev);

          checkpoint();
          res_prev = norm_difference_frobenius(bfun_sol_prev, rhs_tensor) / norm_rhs;

          if (not(is_tiny(norm_rhs)))
          {
            checkpoint();
            auto sol_tensor = solve3d_2ml(phia.cores[i], crA.cores[i], phia.cores[i + 1], rhs_tensor, real_tol * norm_rhs, sol_prev, solve3d_2ml_options);
            sol = flatten(sol_tensor);
            auto bfun_sol = bfun3(phia.cores[i], crA.cores[i], phia.cores[i + 1], sol_tensor);
            res_new = norm_difference_frobenius(bfun_sol, rhs_tensor) / norm_rhs;
          }
          else
          {
            checkpoint();
            boba_assert_equal(sol.size(), sol_prev.size(), "Unexpected size");
            sol.fill_with_zeros();
            res_new = 0.0;
          }
        }

        if (((res_prev / res_new) < resid_damp) and (res_new > real_tol))
        {
          boba_warn("the residual damp was smaller than in the truncation ");
          //  Bad things may happen. We are to introduce an error definetly
          // larger than the improvement by the local solution. Usually it
          // means that a preconditioner is needed.
        }

        checkpoint();
        auto sol_diff = sol - boba::flatten(sol_prev);
        auto dx = ::boba::norm_frobenius(sol_diff) / ::boba::norm_frobenius(sol);
        max_dx = boba::max(max_dx, dx);
        max_res = boba::max(max_res, res_prev);

        // Truncation
        checkpoint();
        auto sol_matrix = reshape_to_matrix(sol, {crx.ranks(i) * cry.sizes(i), crx.ranks(i + 1)});

        boba::SVD<space, data_t> svd;
        svd.tolerance_relative = 0.0;
        svd.tolerance_absolute = 0.0;

        boba::Matrix<space, data_t> u, s, v;
        index_t r = boba::min(crx.ranks(i) * cry.sizes(i), crx.ranks(i + 1));

        checkpoint();
        if ((kickrank >= 0) and (i < dimension - 1))
        {
          svd(sol_matrix);

          r = min(r, svd.U.cols());
          r = min(r, svd.V.cols());

          u = svd.U;
          s = boba::diagonalize(svd.S);
          v = svd.V;
          for (; r > 1; r--)
          {
            svd.truncate(r);
            auto cursol = svd.reform_matrix();
            data_t res;
            auto linear_system_solution_size = crx.ranks(i) * cry.sizes(i) * crx.ranks(i + 1);
            if ((linear_system_solution_size < static_cast<size_t>(max_direct_solve_size)) or (max_direct_solve_size == -1))
            {
              auto diff = B * boba::flatten(cursol) - rhs_flat;
              res = ::boba::norm_frobenius(diff) / norm_rhs;
            }
            else
            {
              auto s0 = phia.cores[i].sizes(0);
              auto s1 = crA.core_cols(i);
              auto s2 = phia.cores[i + 1].sizes(0);
              checkpoint();
              auto cursol_ten = boba::reshape_from_matrix<3>(cursol, {s0, s1, s2});
              auto bfun_rhs = flatten(bfun3(phia.cores[i], crA.cores[i], phia.cores[i + 1], cursol_ten));
              bfun_rhs -= rhs_flat;
              res = ::boba::norm_frobenius(bfun_rhs) / norm_rhs;
            }
            if (res > boba::max(real_tol * resid_damp, res_new))
            {
              break;
            }
          }
          checkpoint();

          r += 1;
          r = boba::min(r, s.size() - 1);
          r = boba::min(r, max_allowed_ranks);
          r = boba::max(r, size_t(1));
          checkpoint();
        }
        else
        {
          QR<space, data_t> _qr;
          _qr(sol_matrix);
          u = _qr.Q;
          v = _qr.R.transpose();
          r = u.cols();
          auto svec = svd.S;
          svec.resize({r});
          svec.fill_with(1.0);
          s = boba::diagonalize(svec);
          checkpoint();
        }

        checkpoint();
        u.resize({u.rows(), r});
        v.resize({v.rows(), r});
        auto svec = boba::diagonalize(s);
        svec.resize({r});

        boba::apply_as_diagonal_right_in_place(svec, v);

        if ((kickrank > 0) and (not(last_sweep)))
        {
          // Update crz (we dont want just random-svd)
          checkpoint();
          auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx.ranks(i), cry.sizes(i), crx.ranks(i + 1)});
          checkpoint();
          auto crzAt = bfun3(phiza.cores[i], crA.cores[i], phiza.cores[i + 1], uvT);
          checkpoint();
          auto crzAt_matrix = boba::reshape_to_matrix(crzAt, {rz[i] * cry.sizes(i), rz[i + 1]});

          auto crzy = project_phizy(phizy.cores[i], y1, phizy.cores[i + 1]);

          checkpoint();
          auto crzy_matrix = boba::reshape_to_matrix(crzy, {rz[i] * cry.sizes(i), rz[i + 1]});

          crznew = crzy_matrix - crzAt_matrix;

          ::boba::SVD<space, data_t> _svd;
          _svd.tolerance_relative = 0.0;
          _svd.tolerance_absolute = 0.0;
          _svd(crznew);
          crznew = _svd.U;

          {
            auto fetch_col = boba::min(kickrank, crznew.cols());
            auto crznew_temp = crznew;
            crznew = crznew_temp.get_submatrix({0, crznew.rows()}, {0, fetch_col});
          }

          QR<space, data_t> qr;
          qr(crznew);
          crznew = qr.Q;

          rznew = crznew.cols();
          crznew_tensor = boba::reshape_from_matrix<3>(crznew, {rz[i], cry.sizes(i), rznew});
          crz.cores[i] = crznew_tensor;
        }

        if (i < dimension - 1) // enrichment, etc
        {
          if ((kickrank > 0) and (not(last_sweep)))
          {
            // Enrichment in X
            checkpoint();
            auto uvT = boba::reshape_from_matrix<3>(u * v.transpose(), {crx.ranks(i), cry.sizes(i), crx.ranks(i + 1)});
            auto leftresid = bfun3(phia.cores[i], crA.cores[i], phiza.cores[i + 1], uvT);

            // leftresid = leftresid(rx, n, rzp1)
            checkpoint();
            auto leftresid_mat = boba::reshape_to_matrix(leftresid, {crx.ranks(i) * cry.sizes(i), rz[i + 1]});

            auto lefty_new_temp = project_phizy(phiy.cores[i], y1, phizy.cores[i + 1]);

            checkpoint();
            // lefty_new(rx, n, ryp1) = lefty_new_temp(1, rz, n, 1, ry3)
            auto lefty_rows = lefty_new_temp.sizes(0) * cry.sizes(i);
            auto lefty_cols = lefty_new_temp.size() / lefty_rows;
            auto lefty_mat = boba::reshape_to_matrix(lefty_new_temp, {lefty_rows, lefty_cols});

            checkpoint();
            auto u_kick = lefty_mat - leftresid_mat;
            auto u_kicked = boba::concatenate_columns(u, u_kick);

            QR<space, data_t> qr;
            qr(u_kicked);

            u = qr.Q;
            auto rv = qr.R;

            if (u_kick.cols() > 0)
            {
              boba::Matrix<space, data_t> zeros({crx.ranks(i + 1), u_kick.cols()});
              zeros.fill_with_zeros();
              auto v_expand = boba::concatenate_columns(v, zeros);

              auto v_temp = boba::tensor_contraction<1>(
                {"row", "k"}, v_expand, {"col^T", "k"}, rv, {"row", "col^T"});
              checkpoint();
              v = boba::reshape_to_matrix(v_temp, v_temp.sizes());
            }
            else
            {
              auto v_temp = boba::tensor_contraction<1>(
                {"row", "k"}, v, {"col^T", "k"}, rv, {"row", "col^T"});
              v = boba::reshape_to_matrix(v_temp, v_temp.sizes());
            }
            checkpoint();
          }

          r = u.cols();
          auto cr2 = crx.cores[i + 1];

          checkpoint();
          auto v_new = boba::tensor_contraction<1>(
            {"rxp1", "?"}, v, {"rxp1", "np1", "rxp2"}, cr2, {"?", "np1", "rxp2"});

          v = boba::reshape_to_matrix(v_new, {v_new.sizes(0), v_new.sizes(1) * v_new.sizes(2)});

          // Remove old scale component from nrmsc
          nrmsc = nrmsc / (nrmsy[i] / (nrmsa[i] * nrmsx[i]));

          auto curnorm = ::boba::norm_frobenius(v);
          if (curnorm > 0)
          {
            v *= (1.0 / curnorm);
          }
          else
          {
            curnorm = 1.0;
          }

          nrmsx[i] *= curnorm;

          checkpoint();
          auto u_tensor = boba::reshape_from_matrix<3>(u, {crx.ranks(i), cry.sizes(i), r});
          checkpoint();
          auto v_tensor = boba::reshape_from_matrix<3>(v, {r, cry.sizes(i + 1), crx.ranks(i + 2)});

          // Recompute phi.
          std::tie(phia.cores[i + 1], nrmsa[i]) = compute_next_Phi<dimension>(phia.cores[i], u_tensor, crA.cores[i], u_tensor, true, -1.0);
          std::tie(phiy.cores[i + 1], nrmsy[i]) = compute_next_Phi<dimension>(phiy.cores[i], u_tensor, cry.cores[i], true, -1.0);

          // Add new scales
          nrmsc *= (nrmsy[i] / (nrmsa[i] * nrmsx[i]));

          crx.cores[i] = u_tensor;
          crx.cores[i + 1] = v_tensor;

          // Update z and its projections
          if ((kickrank > 0) and (not(last_sweep)))
          {
            checkpoint();
            crznew_tensor = boba::reshape_from_matrix<3>(crznew, {rz[i], cry.sizes(i), rznew});
            rz[i + 1] = rznew;
            std::tie(phiza.cores[i + 1], std::ignore) = compute_next_Phi<dimension>(phiza.cores[i], crznew_tensor, crA.cores[i], crx.cores[i], true, nrmsa[i]);
            std::tie(phizy.cores[i + 1], std::ignore) = compute_next_Phi<dimension>(phizy.cores[i], crznew_tensor, cry.cores[i], true, nrmsy[i]);
          }
        }
        else // i==d
        {
          checkpoint();
          crx.cores[i] = boba::reshape<3>(sol, {crx.ranks(i), cry.sizes(i), crx.ranks(i + 1)});
        }
      }

      std::cout << "TensorTrainAMEN: sweep " << swp << ", max_dx: " << max_dx << ", max_res: " << max_res << ", ranks: " << crx.ranks_string() << ", CR: " << crx.compression_rate() << std::endl;

      if (last_sweep)
      {
        break;
      }

      if (max_res < convergence_tolerance)
      {
        last_sweep = true;
      }
    }

    checkpoint();
    // Redistribute norms equally
    auto scaled_nrmsx = boba::exp(boba::sum(boba::log(nrmsx)) / static_cast<data_t>(dimension));
    for (index_t k = 0; k < dimension; k++)
    {
      crx.cores[k] = crx.cores[k] * scaled_nrmsx;
    }

    checkpoint();
    return crx;
  }

private:
  template <execution_space space>
  Tensor<6, space, data_t> project_operator_A(
    const Tensor<3, space, data_t>& phia_i,
    const Tensor<4, space, data_t>& A_core,
    const Tensor<3, space, data_t>& phia_ip1)
  {
    BOBA_CALI_MARK
    checkpoint();
    auto phi1A1 = ::boba::tensor_contraction<1>(
      {"rx", "rx_", "ra"}, phia_i, {"ra", "row", "col", "rap1"}, A_core, {"rx", "rx_", "row", "col", "rap1"});

    checkpoint();
    auto phi1A1phi2 = ::boba::tensor_contraction<1>(
      {"rx", "rx_", "row", "col", "rap1"}, phi1A1, {"rxp1", "rap1", "rxp1_"}, phia_ip1, {"rx", "rx_", "row", "col", "rxp1", "rxp1_"});

    return phi1A1phi2;
  }

  template <execution_space space>
  Tensor<3, space, data_t> project_phizy(
    const Tensor<3, space, data_t>& phizy_i,
    const Tensor<3, space, data_t>& y1,
    const Tensor<3, space, data_t>& phizy_ip1)
  {
    BOBA_CALI_MARK
    checkpoint();
    auto phizyy1 = boba::tensor_contraction<1>(
      {"rz", "one", "ry"}, phizy_i, {"ry", "n", "ryp1"}, y1, {"rz", "one", "n", "ryp1"});

    checkpoint();
    auto crzy = boba::tensor_contraction<1>(
      {"rz", "one", "n", "ryp1"}, phizyy1, {"ryp1", "one_", "rzp1"}, phizy_ip1, {"rz", "one", "n", "one_", "rzp1"});

    // crzy(rz, n, rzp1)
    auto crzy_final = reshape<3>(crzy, {crzy.sizes(0), crzy.sizes(2), crzy.sizes(4)});

    return crzy_final;
  }
};

} // namespace boba
