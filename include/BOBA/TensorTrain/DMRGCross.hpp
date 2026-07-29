// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace boba
{

/**
 * \brief Reimplements Oseledets' DMRG Cross using the BoBa framework.
 *
 * See https://github.com/oseledets/TT-Toolbox/blob/master/cross/dmrg_cross.m.
 * Given a function `f(i, j, k, ...)`, this class builds a tensor train that
 * approximates `f` with a DMRG cross sweep.
 *
 * @tparam data_t Scalar value type.
 * @tparam index_t Index type.
 */
template <typename data_t>
struct DMRGCross
{
  bool verbose = false;
  index_t radd = 0;
  index_t kickrank = 2;
  size_t nswp = 10;
  data_t tolerance = 1.0e-7;

  /**
   * @brief Selects the row-submatrix extraction strategy during the sweep.
   */
  enum class SubmatrixSelectionType
  {
    MAXVOL,
    DEIM ///< Uses the DEIM selector from `include/BOBA/linear_algebra/CUR.hpp`.
  };

  SubmatrixSelectionType submatrix_selection_type = SubmatrixSelectionType::MAXVOL;

  /**
   * @brief Selects interpolation indices for a core unfolding.
   * @tparam space Matrix execution space.
   * @param a_matrix Matrix whose rows are to be sampled.
   * @return Selected row indices on the host.
   */
  template <::boba::execution_space space>
  [[nodiscard]]
  ::boba::Vector<host_space, index_t> select_indices(
    ::boba::Matrix<space, data_t>& a_matrix) const
  {
    BOBA_CALI_OBJECT_MARK

    if (submatrix_selection_type == SubmatrixSelectionType::DEIM)
    {
      return DEIM(a_matrix);
    }
    else // MAXVOL
    {
      return maxvol2(a_matrix);
    }
  }

  /**
   * @brief Applies DMRG cross to approximate a function as a tensor train.
   * @tparam dimension Tensor dimension.
   * @tparam space Initial-guess execution space.
   * @tparam lambda_t Callable type used to evaluate tensor entries.
   * @param tt_initial_guess Initial tensor-train guess.
   * @param function_to_approximate Callable that evaluates the target tensor entrywise.
   * @return A host-space tensor train approximation.
   */
  template <size_t dimension, execution_space space, typename lambda_t>
  [[nodiscard]]
  ::boba::TensorTrain<dimension, host_space, data_t> apply(
    const ::boba::TensorTrain<dimension, space, data_t>& tt_initial_guess,
    lambda_t&& function_to_approximate)
  {
    BOBA_CALI_BEGIN("DMRGCross_setup");
    checkpoint();

    ::boba::TensorTrain<dimension, host_space, data_t> approximated_tt = tt_initial_guess;

    checkpoint();
    auto mode_sizes = approximated_tt.sizes();
    auto tt_ranks = approximated_tt.ranks();
    size_t sweep_count = 1;

    checkpoint();
    ::boba::Array<::boba::Matrix<space, data_t>, dimension + 1> transfer_matrices;
    transfer_matrices[0].resize({1, 1});
    transfer_matrices[0].fill_with(1);
    transfer_matrices[dimension].resize({1, 1});
    transfer_matrices[dimension].fill_with(1);

    checkpoint();
    ::boba::Array<::boba::Matrix<space, index_t>, dimension + 1> interpolation_indices;

    checkpoint();
    interpolation_indices[dimension].resize({0, tt_ranks[dimension]});
    interpolation_indices[0].resize({tt_ranks[0], 0});

    ::boba::Matrix<space, data_t> right_interface({1, 1});
    right_interface.fill_with(1);

    checkpoint();
    BOBA_CALI_SWITCH("DMRGCross_setup", "DMRGCross_core_setup");
    for (size_t core_id = dimension - 1; core_id > 0; core_id--)
    {
      checkpoint();
      const auto mode_extent = approximated_tt.sizes(core_id);
      checkpoint();
      ::boba::Matrix<host_space, data_t> core_matrix = reshape_to_matrix(approximated_tt.cores[core_id], {tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

      auto core_times_right_interface = core_matrix * right_interface;
      checkpoint();
      core_matrix.resize({tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
      core_matrix.reshape(core_times_right_interface);
      checkpoint();
      core_matrix.transpose_in_place();

      checkpoint();
      ::boba::QR<host_space, data_t> qr;
      qr(core_matrix);

      core_matrix = qr.Q;
      auto triangular_factor = qr.R;

      auto selected_indices = select_indices(qr.Q);

      checkpoint();
      ::boba::Matrix<space, index_t> trailing_indices = interpolation_indices[core_id + 1];

      auto next_rank = boba::min(mode_extent * tt_ranks[core_id + 1], tt_ranks[core_id]);

      checkpoint();
      ::boba::Matrix<space, index_t> leading_indices({dimension - core_id, next_rank});
      leading_indices.fill_with_zeros();

      checkpoint();
      for (size_t selected_column = 0; selected_column < next_rank; selected_column++)
      {
        index_t flattened_index = selected_indices({selected_column});
        ::boba::Multiindexer<2> unfolded_core_indices({tt_ranks[core_id + 1], mode_extent});
        auto [right_rank_index, mode_index] = unfolded_core_indices.multiindex(flattened_index);

        checkpoint();
        auto trailing_indices_view = trailing_indices.view();
        auto leading_indices_view = leading_indices.view();

        ::boba::loop<host_space, 1>(leading_indices.rows(),
                                    [=] __boba_host_device__(index_t i)
        {
          index_t value = 0;
          if (i == 0)
          {
            value = mode_index;
          }
          else
          {
            value = trailing_indices_view({i - 1, right_rank_index});
          }
          leading_indices_view({i, selected_column}) = value;
        });
      }

      interpolation_indices[core_id] = leading_indices;

      right_interface = core_matrix.extract_rows(selected_indices);

      {
        auto temp = right_backsolve(right_interface, core_matrix);
        core_matrix = temp;
      }
      {
        auto temp = right_interface * triangular_factor;
        right_interface = temp;
      }
      right_interface.transpose_in_place();

      checkpoint();
      core_matrix.transpose_in_place();
      checkpoint();

      approximated_tt.cores[core_id].resize({tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
      checkpoint();
      approximated_tt.cores[core_id].reshape(core_matrix);

      checkpoint();
      core_matrix.reshape({tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});
      {
        auto temp = core_matrix * transfer_matrices[core_id + 1];
        core_matrix = temp;
      }

      checkpoint();
      core_matrix.reshape({tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
      core_matrix.transpose_in_place();

      checkpoint();
      ::boba::QR<host_space, data_t> QR2;
      QR2(core_matrix);
      triangular_factor = QR2.R;
      transfer_matrices[core_id] = triangular_factor;
    }

    checkpoint();

    ::boba::Matrix<host_space, data_t> first_core_matrix({tt_ranks[0] * mode_sizes[0], tt_ranks[1]});
    {
      auto temp = approximated_tt.cores[0];
      first_core_matrix.reshape(temp);
    }

    checkpoint();
    {
      ::boba::Tensor<3, host_space, data_t> temp({tt_ranks[0], mode_sizes[0], tt_ranks[1]});
      temp.reshape(first_core_matrix * right_interface);
      approximated_tt.cores[0] = temp;
    }

    bool not_converged = true;
    bool sweep_right = true;

    size_t core_id = 0;
    auto max_relative_error = 0.0;

    BOBA_CALI_SWITCH("DMRGCross_core_setup", "DMRGCross_sweep");
    while ((sweep_count < nswp) && not_converged)
    {
      checkpoint();
      auto left_core = approximated_tt.cores[core_id];
      auto right_core = approximated_tt.cores[core_id + 1];
      auto left_indices = interpolation_indices[core_id];
      auto right_indices = interpolation_indices[core_id + 2];

      checkpoint();
      ::boba::Tensor<4, host_space, data_t> score_tensor({tt_ranks[core_id], mode_sizes[core_id], mode_sizes[core_id + 1], tt_ranks[core_id + 2]});

      create_score_tensor_from_index_and_function<dimension>(left_indices, right_indices, score_tensor, std::forward<lambda_t>(function_to_approximate));

      checkpoint();
      ::boba::Matrix<host_space, data_t> score({tt_ranks[core_id], mode_sizes[core_id] * mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
      score.reshape(score_tensor);

      score = transfer_matrices[core_id] * score;
      tt_ranks[core_id] = score.sizes(0);

      checkpoint();
      score.reshape({tt_ranks[core_id] * mode_sizes[core_id] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
      {
        auto temp = score * transfer_matrices[core_id + 2];
        score = temp;
      }
      tt_ranks[core_id + 2] = score.sizes(1);

      score.reshape({tt_ranks[core_id] * mode_sizes[core_id], mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});

      ::boba::SVD<host_space, data_t> svd;
      svd.tolerance_relative = 1.0e-7;
      svd(score);
      auto u = svd.U;
      auto s = svd.S;
      auto v = svd.V;
      auto r = svd.significant_singular_values;

      // for compatibility with tt-toolbox
      u *= -1.0;
      v *= -1.0;

      // Kick rank
      if (sweep_right)
      {
        checkpoint();
        // v = (s*v')' = v * s'
        ::boba::apply_as_diagonal_right_in_place(s, v);

        ::boba::Matrix<host_space, data_t> ur({u.rows(), kickrank});
        ur.fill_with_zeros();

        ::boba::Matrix<host_space, data_t> ur_preort({u.rows(), kickrank});
        ur_preort.fill_with_random();
        ur.reshape(ur_preort);

        auto u_reort = reort(u, ur);
        u = u_reort;
        radd = u.cols() - r;
        if (radd > 0)
        {
          ::boba::Matrix<host_space, data_t> vr({v.rows(), radd});
          vr.fill_with_zeros();
          auto v_temp = concatenate_columns(v, vr);
          v = v_temp;
        }
        r = r + radd;
      }
      else
      {
        checkpoint();
        ::boba::apply_as_diagonal_right_in_place(s, u);

        ::boba::Matrix<host_space, data_t> vr({v.rows(), kickrank});
        vr.fill_with_zeros();

        ::boba::Matrix<host_space, data_t> ur_preort({v.rows(), kickrank});
        ur_preort.fill_with_random();

        vr.reshape(ur_preort);
        v = reort(v, vr);

        radd = v.cols() - r;
        if (radd > 0)
        {
          ::boba::Matrix<host_space, data_t> ur({u.rows(), radd});
          ur.fill_with_zeros();
          auto u_temp = concatenate_columns(u, ur);
          u = u_temp;
        }
        r = r + radd;
      }

      v.transpose_in_place();

      boba::Matrix<space, data_t> left_core_matrix({left_core.size() / tt_ranks[core_id + 1], tt_ranks[core_id + 1]});
      left_core_matrix.reshape(left_core);

      boba::Matrix<space, data_t> right_core_matrix({tt_ranks[core_id + 1], right_core.size() / tt_ranks[core_id + 1]});
      right_core_matrix.reshape(right_core);

      auto approximation = left_core_matrix * right_core_matrix;
      approximation.reshape({tt_ranks[core_id], mode_sizes[core_id] * mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
      approximation = transfer_matrices[core_id] * approximation;
      approximation.reshape({tt_ranks[core_id] * mode_sizes[core_id] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
      approximation = approximation * transfer_matrices[core_id + 2];

      auto norm_diff = ::boba::norm_difference_frobenius(::boba::flatten(score), ::boba::flatten(approximation));
      auto norm_score = ::boba::norm_frobenius(score);
      auto local_relative_error = norm_diff / norm_score;
      max_relative_error = ::boba::max(max_relative_error, local_relative_error);

      if (verbose)
      {
        boba_print(sweep_count);
        boba_print(core_id);
        boba_print(r);
        boba_print(local_relative_error);
      }

      checkpoint();
      tt_ranks[core_id + 1] = r;
      u.reshape({tt_ranks[core_id], mode_sizes[core_id] * r});
      {
        auto temp = backsolve(transfer_matrices[core_id], u);
        u = temp;
      }

      checkpoint();
      v.reshape({r * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
      u.reshape({tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

      {
        auto temp = right_backsolve(transfer_matrices[core_id + 2], v);
        v = temp;
      }

      checkpoint();
      v.reshape({r, mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});

      if (sweep_right)
      {
        checkpoint();
        ::boba::QR<host_space, data_t> qr;
        qr(u);
        u = qr.Q;
        auto triangular_factor = qr.R;
        auto selected_indices = select_indices(u);

        checkpoint();
        right_interface = u.extract_rows(selected_indices);

        {
          auto temp = right_backsolve(right_interface, u);
          u = temp;
        }

        checkpoint();
        approximated_tt.cores[core_id].resize({tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
        approximated_tt.cores[core_id].reshape(u);

        {
          auto temp = right_interface * triangular_factor;
          right_interface = temp;
        }
        v = right_interface * v;

        checkpoint();
        approximated_tt.cores[core_id + 1].resize({tt_ranks[core_id + 1], mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
        approximated_tt.cores[core_id + 1].reshape(v);

        ::boba::Matrix<host_space, data_t> updated_left_core({tt_ranks[core_id], mode_sizes[core_id] * tt_ranks[core_id + 1]});
        updated_left_core.reshape(u);
        auto updated_left_core_with_transfer = transfer_matrices[core_id] * updated_left_core;

        checkpoint();
        updated_left_core_with_transfer.reshape({tt_ranks[core_id] * mode_sizes[core_id], tt_ranks[core_id + 1]});

        qr(updated_left_core_with_transfer);
        triangular_factor = qr.R;
        transfer_matrices[core_id + 1] = triangular_factor;

        auto previous_indices = interpolation_indices[core_id];
        ::boba::Matrix<host_space, index_t> updated_indices({tt_ranks[core_id + 1], core_id + 1});
        updated_indices.fill_with_zeros();
        for (size_t selected_row = 0; selected_row < tt_ranks[core_id + 1]; selected_row++)
        {
          auto flattened_index = selected_indices({selected_row});
          ::boba::Multiindexer<2> unfolded_core_indices({tt_ranks[core_id], mode_sizes[core_id]});
          auto [left_rank_index, mode_index] = unfolded_core_indices.multiindex(flattened_index);

          checkpoint();
          auto previous_indices_view = previous_indices.view();
          auto updated_indices_view = updated_indices.view();

          ::boba::loop<host_space, 1>(updated_indices.cols(),
                                      [=] __boba_host_device__(index_t k)
          {
            index_t value = 0;

            if (core_id == previous_indices_view.sizes(1))
            {
              value = mode_index;
            }
            else
            {
              value = previous_indices_view({left_rank_index, k});
            }
            updated_indices_view({selected_row, k}) = value;
          });
        }

        interpolation_indices[core_id + 1] = updated_indices;
        if (core_id == dimension - 2)
        {
          sweep_right = not(sweep_right);
        }
        else
        {
          core_id += 1;
        }
      }
      else
      {
        // Reverse direction
        checkpoint();

        v.transpose_in_place();
        ::boba::QR<host_space, data_t> qr;
        qr(v);
        v = qr.Q;
        auto triangular_factor = qr.R;
        auto selected_indices = select_indices(v);

        checkpoint();
        right_interface = v.extract_rows(selected_indices);

        {
          auto temp = right_backsolve(right_interface, v);
          v = temp;
        }

        checkpoint();
        ::boba::Tensor<3, space, data_t> reshaped_right_core({mode_sizes[core_id + 1], tt_ranks[core_id + 2], tt_ranks[core_id + 1]});
        reshaped_right_core.reshape(v);

        approximated_tt.cores[core_id + 1] = reshaped_right_core;
        ::boba::permute({"index", "rank_right", "rank_left"}, approximated_tt.cores[core_id + 1], {"rank_left", "index", "rank_right"});

        {
          auto temp = right_interface * triangular_factor;
          right_interface = temp;
          right_interface.transpose_in_place();
          temp = u * right_interface;
          u = temp;
        }
        approximated_tt.cores[core_id].resize({tt_ranks[core_id], mode_sizes[core_id], tt_ranks[core_id + 1]});
        approximated_tt.cores[core_id].reshape(u);

        v.transpose_in_place();
        v.reshape({tt_ranks[core_id + 1] * mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
        {
          auto temp = v * transfer_matrices[core_id + 2];
          v = temp;
        }

        checkpoint();
        v.reshape({tt_ranks[core_id + 1], mode_sizes[core_id + 1] * tt_ranks[core_id + 2]});
        v.transpose_in_place();

        qr(v);
        triangular_factor = qr.R;

        transfer_matrices[core_id + 1] = triangular_factor;
        auto previous_indices = interpolation_indices[core_id + 2];

        ::boba::Matrix<host_space, index_t> updated_indices({dimension - core_id - 1, tt_ranks[core_id + 1]});
        updated_indices.fill_with_zeros();

        for (size_t selected_row = 0; selected_row < tt_ranks[core_id + 1]; selected_row++)
        {
          auto flattened_index = selected_indices({selected_row});
          ::boba::Multiindexer<2> unfolded_core_indices({mode_sizes[core_id + 1], tt_ranks[core_id + 2]});
          auto [mode_index, right_rank_index] = unfolded_core_indices.multiindex(flattened_index);

          checkpoint();
          auto previous_indices_view = previous_indices.view();
          auto updated_indices_view = updated_indices.view();

          ::boba::loop<host_space, 1>(updated_indices.rows(),
                                      [=] __boba_host_device__(index_t i)
          {
            index_t value = 0;
            if (i == 0)
            {
              value = mode_index;
            }
            else
            {
              value = previous_indices_view({i - 1, right_rank_index});
            }
            updated_indices_view({i, selected_row}) = value;
          });
        }

        interpolation_indices[core_id + 1] = updated_indices;

        if (core_id == 0)
        {
          sweep_right = not(sweep_right);
          sweep_count = sweep_count + 1;
          if (max_relative_error < tolerance)
          {
            not_converged = false;
          }
          else
          {
            max_relative_error = 0.0;
          }
        }
        else
        {
          core_id--;
        }
        checkpoint();
      }
      checkpoint();
    }
    checkpoint();

    BOBA_CALI_END("DMRGCross_sweep");
    return approximated_tt;
  }

  /**
   * @brief Reorthogonalizes additional columns against an existing basis.
   *
   * See https://github.com/oseledets/TT-Toolbox/blob/master/core/reort.m.
   *
   * @tparam space Matrix execution space.
   * @param u_in Existing basis vectors.
   * @param uadd_in Candidate columns to orthogonalize and append.
   * @return Concatenated orthogonalized basis on the host.
   */
  template <::boba::execution_space space>
  [[nodiscard]]
  ::boba::Matrix<host_space, data_t> reort(const ::boba::Matrix<space, data_t>& u_in, ::boba::Matrix<space, data_t> const& uadd_in)
    const
  {
    BOBA_CALI_OBJECT_MARK

    if (uadd_in.cols() == 0)
    {
      return u_in;
    }
    if (u_in.rows() == u_in.cols())
    {
      return u_in;
    }

    ::boba::Matrix<host_space, data_t> uadd;

    if (u_in.cols() + uadd_in.cols() >= u_in.rows())
    {
      auto uadd_truncate = uadd_in.get_submatrix({0, uadd_in.rows()}, {0, u_in.rows() - u_in.cols()});
      uadd = uadd_truncate;
    }
    else
    {
      uadd = uadd_in;
    }

    auto u = u_in;
    auto projection_coefficients = u.transpose() * uadd;

    auto unew = uadd - u * projection_coefficients;
    bool reort_flag = true;
    size_t reorthogonalization_iteration = 1;

    while (reort_flag && (reorthogonalization_iteration <= 20))
    {
      ::boba::Vector<host_space, data_t> norm_unew({unew.cols()});
      ::boba::Vector<host_space, data_t> norm_uadd({uadd.cols()});
      norm_unew.fill_with_zeros();
      norm_uadd.fill_with_zeros();

      auto norm_unew_view = norm_unew.atomic_view();
      auto norm_uadd_view = norm_uadd.atomic_view();
      auto unew_view = unew.const_view();
      auto uadd_view = uadd.const_view();

      ::boba::loop<space, 2>(unew.sizes(),
                             [=] __boba_host_device__(::boba::Array<index_t, 2> ij)
      {
        norm_unew_view(ij[1]) += ::boba::pow(unew_view(ij), 2.0);
        norm_uadd_view(ij[1]) += ::boba::pow(uadd_view(ij), 2.0);
      });

      size_t reort_condition = 0;

      ::boba::sum_reduce<space>(reort_condition, 0_z, norm_unew.size(), [=] __boba_host_device__(index_t i, boba::sum_reducer_operator<size_t> & local_value)
      {
        if (norm_unew_view(i) <= 0.25 * norm_uadd_view(i))
        {
          local_value += 1;
        }
      });

      reort_flag = (reort_condition > 0);

      ::boba::QR<host_space, data_t> qr;
      qr(unew);
      // [unew,~]=qr(unew,0);
      unew = qr.Q;

      if (reort_flag)
      {
        uadd = unew;
        auto reorthogonalization_coefficients = u.transpose() * unew;
        unew = unew - u * reorthogonalization_coefficients;
        reorthogonalization_iteration = reorthogonalization_iteration + 1;
      }
    }

    auto u_temp = ::boba::concatenate_columns(u, unew);
    u = u_temp;

    if (reort_flag)
    {
      boba_print("Reort failed to reort!");

      // [y,~]=qr([u,unew],0);
      ::boba::QR<host_space, data_t> qr;
      qr(u_temp);
      return qr.Q;
    }
    else
    {
      return u_temp;
    }
  }

  /**
   * @brief Selects a near-maximum-volume row submatrix.
   *
   * See https://github.com/oseledets/TT-Toolbox/blob/master/core/maxvol2.m.
   *
   * @tparam space Matrix execution space.
   * @param a_in Input matrix.
   * @return Selected row indices on the host.
   */
  template <::boba::execution_space space>
  [[nodiscard]]
  ::boba::Vector<host_space, index_t> maxvol2(::boba::Matrix<space, data_t>& a_in)
    const
  {
    BOBA_CALI_OBJECT_MARK

    auto row_count = a_in.rows();
    auto column_count = a_in.cols();

    boba::Vector<host_space, index_t> selected_indices;

    if (row_count <= column_count)
    {
      selected_indices.resize({row_count});
      auto selected_indices_view = selected_indices.view();

      ::boba::loop<host_space, 1>(selected_indices.size(),
                                  [=] __boba_host_device__(index_t i)
      {
        selected_indices_view(i) = i;
      });
      return selected_indices;
    }

    size_t max_iterations = 100;
    data_t maxvol_tolerance = 5e-2;

    ::boba::LU<space, data_t> lu;
    lu.lu_type = ::boba::LU<space, data_t>::lu_types::full_pivot;
    lu(a_in);
    auto pivot_indices = lu.P.transpose();

    {
      selected_indices.resize({column_count});
      auto selected_indices_view = selected_indices.view();
      auto pivot_indices_view = pivot_indices.view();
      ::boba::loop<space, 1>(selected_indices.size(),
                             [=] __boba_host_device__(index_t i)
      {
        selected_indices_view(i) = pivot_indices_view(i);
      });
    }

    auto selected_submatrix = a_in.extract_rows(selected_indices);

    auto interpolation_matrix = right_backsolve(selected_submatrix, a_in);

    size_t iteration_count = 0;
    while (iteration_count <= max_iterations)
    {
      auto [max_entry, max_entry_index] = interpolation_matrix.max_abs_loc_reduce();

      auto [pivot_row, pivot_column] = a_in.multiindex(max_entry_index);

      if (max_entry <= 1.0 + maxvol_tolerance)
      {
        ::boba::sort(selected_indices);
        return selected_indices;
      }

      checkpoint();
      auto replaced_row = selected_indices({pivot_column});
      auto selected_column = interpolation_matrix.extract_columns(pivot_column);
      auto replaced_row_values = interpolation_matrix.extract_rows(replaced_row);
      auto pivot_row_values = interpolation_matrix.extract_rows(pivot_row);
      auto interpolation_update = replaced_row_values - pivot_row_values;
      // `selected_column` already has the right column, so grab the right row and transfer to the cpu.
      interpolation_update /= selected_column.extract_rows(pivot_row).sum_reduce();

      interpolation_matrix += selected_column * interpolation_update;
      selected_indices({pivot_column}) = pivot_row;
      iteration_count++;
    }

    checkpoint();
    return selected_indices;
  }

  /**
   * @brief Populates the local score tensor used by the two-site DMRG cross update.
   * @tparam dimension Tensor dimension.
   * @tparam space Index-matrix execution space.
   * @tparam lambda_t Callable type used to evaluate tensor entries.
   * @param left_indices Left interpolation indices.
   * @param right_indices Right interpolation indices.
   * @param score_tensor Output score tensor to populate.
   * @param function Entry-evaluation callable.
   */
  template <size_t dimension, ::boba::execution_space space, typename lambda_t>
  void create_score_tensor_from_index_and_function(
    ::boba::Matrix<space, index_t> const& left_indices,
    ::boba::Matrix<space, index_t> const& right_indices,
    ::boba::Tensor<4, host_space, data_t>& score_tensor,
    lambda_t function) const
  {
    BOBA_CALI_MARK

    auto left_indices_view = left_indices.const_view();
    auto right_indices_view = right_indices.const_view();

    auto left_index_count = left_indices_view.sizes(1);
    auto first_right_index_offset = left_index_count + 2;
    auto full_index_count = first_right_index_offset + right_indices_view.sizes(0);

    auto score_tensor_view = score_tensor.view();

    ::boba::loop<host_space, 1>(score_tensor_view.size(),
                                [=] __boba_host_device__(index_t k)
    {
      auto [s1, i1, i2, s2] = score_tensor_view.multiindex(k);

      ::boba::Array<size_t, dimension> tensor_index;
      for (size_t left_dim = 0; left_dim < left_index_count; left_dim++)
      {
        tensor_index[left_dim] = left_indices_view({s1, left_dim});
      }
      tensor_index[left_index_count] = i1;
      tensor_index[left_index_count + 1] = i2;
      for (size_t right_dim = first_right_index_offset; right_dim < full_index_count; right_dim++)
      {
        tensor_index[right_dim] = right_indices_view({right_dim - first_right_index_offset, s2});
      }

      score_tensor_view(k) = function(tensor_index);
    });
  }
};

} // namespace boba
