// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <functional>
#include <vector>

namespace boba
{

/**
 * \brief DMRG cross for tensor trains whose number of cores is chosen at run time.
 *
 * The same algorithm, options and helpers as DMRGCross (it derives from it and reuses maxvol2,
 * reorthogonalize and select_indices unchanged); only apply() differs, with the per-dimension arrays
 * held in std::vector and the result returned as a RuntimeTensorTrain. The function to approximate
 * is called with the full multi-index as a `std::vector<size_t> const&` of length get_dimension().
 *
 * @tparam data_t Scalar value type.
 */
template <typename data_t>
struct RuntimeDMRGCross : public DMRGCross<data_t>
{
  using base_t = DMRGCross<data_t>;
  using real_data_t = typename base_t::real_data_t;

  /**
   * @brief Applies DMRG cross to approximate a function as a tensor train.
   * @tparam space Initial-guess execution space.
   * @tparam lambda_t Callable `data_t(std::vector<size_t> const&)` evaluating tensor entries.
   * @tparam random_generator_t Type satisfying the C++ UniformRandomBitGenerator requirements.
   * @param tt_initial_guess Initial tensor-train guess (at least two cores).
   * @param function_to_approximate Callable that evaluates the target tensor entrywise.
   * @param[in,out] random_generator Generator used for rank-kick fills.
   * @return A host-space tensor train approximation.
   */
  template <execution_space space, typename lambda_t, typename random_generator_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply(
    const ::boba::RuntimeTensorTrain<space, data_t>& tt_initial_guess,
    lambda_t&& function_to_approximate,
    random_generator_t& random_generator)
  {
    // nvcc rejects a function-local lambda type as a template argument of a function containing extended
    // __host__ __device__ lambdas, so the callable is type-erased here and the algorithm is not templated on it.
    entry_function_t function(std::forward<lambda_t>(function_to_approximate));
    return apply_erased(tt_initial_guess, function, random_generator);
  }

  /**
   * @brief Applies runtime DMRG cross using the process-wide default generator.
   */
  template <execution_space space, typename lambda_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply(
    const ::boba::RuntimeTensorTrain<space, data_t>& tt_initial_guess,
    lambda_t&& function_to_approximate)
  {
    auto& random_generator = ::boba::random::default_context();
    return apply(tt_initial_guess, std::forward<lambda_t>(function_to_approximate), random_generator);
  }

  /// Type-erased entry function: multi-index (length get_dimension()) to tensor entry.
  using entry_function_t = std::function<data_t(std::vector<size_t> const&)>;

  /**
   * @brief The DMRG cross algorithm (same as DMRGCross::apply) on a type-erased entry function.
   */
  template <execution_space space, typename random_generator_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply_erased(
    const ::boba::RuntimeTensorTrain<space, data_t>& tt_initial_guess,
    entry_function_t& function_to_approximate,
    random_generator_t& random_generator)
  {
    BOBA_CALI_BEGIN("RuntimeDMRGCross_setup");
    checkpoint();

    const size_t dimension = tt_initial_guess.get_dimension();
    boba_always_assert(dimension >= 2, "RuntimeDMRGCross: need at least two cores");

    ::boba::RuntimeTensorTrain<host_space, data_t> approximated_tt;
    approximated_tt.m_name = tt_initial_guess.m_name;
    for (size_t d = 0; d < dimension; d++)
    {
      approximated_tt.cores.push_back(::boba::Tensor<3, host_space, data_t>(tt_initial_guess.cores[d]));
    }

    checkpoint();
    auto mode_sizes = approximated_tt.sizes();
    auto tt_ranks = approximated_tt.ranks();
    size_t sweep_count = 1;
    checkpoint();
    std::vector<::boba::Matrix<space, data_t>> transfer_matrices(dimension + 1);
    transfer_matrices[0].resize({1, 1});
    transfer_matrices[0].fill_with(1);
    transfer_matrices[dimension].resize({1, 1});
    transfer_matrices[dimension].fill_with(1);

    checkpoint();
    std::vector<::boba::Matrix<space, index_t>> interpolation_indices(dimension + 1);

    checkpoint();
    interpolation_indices[dimension].resize({0, tt_ranks[dimension]});
    interpolation_indices[0].resize({tt_ranks[0], 0});

    ::boba::Matrix<space, data_t> right_interface({1, 1});
    right_interface.fill_with(1);

    checkpoint();
    BOBA_CALI_SWITCH("RuntimeDMRGCross_setup", "RuntimeDMRGCross_core_setup");
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

      auto selected_indices = this->select_indices(qr.Q);

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

    // When the sweep turns around, the next two-site step is at the same core with the same left and right index
    // sets, so its entries are the ones just evaluated: keep them instead of calling the function again.
    ::boba::Tensor<4, host_space, data_t> previous_score_tensor;
    bool reuse_previous_score = false;

    BOBA_CALI_SWITCH("RuntimeDMRGCross_core_setup", "RuntimeDMRGCross_sweep");
    while ((sweep_count < this->max_sweeps) && not_converged)
    {
      checkpoint();
      auto left_core = approximated_tt.cores[core_id];
      auto right_core = approximated_tt.cores[core_id + 1];
      auto left_indices = interpolation_indices[core_id];
      auto right_indices = interpolation_indices[core_id + 2];

      checkpoint();
      ::boba::Tensor<4, host_space, data_t> score_tensor({tt_ranks[core_id], mode_sizes[core_id], mode_sizes[core_id + 1], tt_ranks[core_id + 2]});

      if (reuse_previous_score and previous_score_tensor.sizes() == score_tensor.sizes())
      {
        score_tensor = previous_score_tensor;
      }
      else
      {
        create_score_tensor_from_index_and_function(dimension, left_indices, right_indices, score_tensor, function_to_approximate);
      }
      previous_score_tensor = score_tensor;
      reuse_previous_score = false;

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
      svd.tolerance_relative = this->two_site_svd_tolerance_relative;
      svd.tolerance_absolute = this->two_site_svd_tolerance_absolute;
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

        ::boba::Matrix<host_space, data_t> ur({u.rows(), this->kickrank});
        ur.fill_with_zeros();

        ::boba::Matrix<host_space, data_t> random_enrichment({u.rows(), this->kickrank});
        random_enrichment.fill_with_random(random_generator);
        ur.reshape(random_enrichment);

        u = this->reorthogonalize(u, ur);
        const auto rank_added = u.cols() - r;
        if (rank_added > 0)
        {
          ::boba::Matrix<host_space, data_t> vr({v.rows(), rank_added});
          vr.fill_with_zeros();
          auto v_temp = concatenate_columns(v, vr);
          v = v_temp;
        }
        r = r + rank_added;
      }
      else
      {
        checkpoint();
        ::boba::apply_as_diagonal_right_in_place(s, u);

        ::boba::Matrix<host_space, data_t> vr({v.rows(), this->kickrank});
        vr.fill_with_zeros();

        ::boba::Matrix<host_space, data_t> random_enrichment({v.rows(), this->kickrank});
        random_enrichment.fill_with_random(random_generator);

        vr.reshape(random_enrichment);
        v = this->reorthogonalize(v, vr);

        const auto rank_added = v.cols() - r;
        if (rank_added > 0)
        {
          ::boba::Matrix<host_space, data_t> ur({u.rows(), rank_added});
          ur.fill_with_zeros();
          auto u_temp = concatenate_columns(u, ur);
          u = u_temp;
        }
        r = r + rank_added;
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

      if (this->verbose)
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
        auto selected_indices = this->select_indices(u);

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

          // New left index = the selected row's previous left index, followed by its mode index in the last column.
          ::boba::loop<host_space, 1>(updated_indices.cols(),
                                      [=] __boba_host_device__(index_t k)
          {
            index_t value = 0;

            if (k == previous_indices_view.sizes(1))
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
          reuse_previous_score = true;
          if (max_relative_error < this->convergence_tolerance)
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
        auto selected_indices = this->select_indices(v);

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
          reuse_previous_score = true;
          sweep_count = sweep_count + 1;
          if (max_relative_error < this->convergence_tolerance)
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

    BOBA_CALI_END("RuntimeDMRGCross_sweep");
    return approximated_tt;
  }

  /**
   * @brief Populates the local score tensor of the two-site update (host loop over its entries).
   * @param dimension Number of tensor-train cores.
   * @param left_indices Left interpolation indices.
   * @param right_indices Right interpolation indices.
   * @param score_tensor Output score tensor to populate.
   * @param function Entry-evaluation function.
   */
  template <::boba::execution_space space>
  void create_score_tensor_from_index_and_function(
    size_t dimension,
    ::boba::Matrix<space, index_t> const& left_indices,
    ::boba::Matrix<space, index_t> const& right_indices,
    ::boba::Tensor<4, host_space, data_t>& score_tensor,
    entry_function_t& function) const
  {
    BOBA_CALI_MARK

    const ::boba::Matrix<host_space, index_t> left_host(left_indices);
    const ::boba::Matrix<host_space, index_t> right_host(right_indices);
    auto left_indices_view = left_host.const_view();
    auto right_indices_view = right_host.const_view();

    const size_t left_index_count = left_indices_view.sizes(1);
    const size_t first_right_index_offset = left_index_count + 2;
    const size_t full_index_count = first_right_index_offset + right_indices_view.sizes(0);
    boba_always_assert_equal(full_index_count, dimension, "RuntimeDMRGCross: index bookkeeping mismatch");

    auto score_tensor_view = score_tensor.view();
    std::vector<size_t> tensor_index(dimension);

    for (index_t k = 0; k < score_tensor_view.size(); k++)
    {
      auto [s1, i1, i2, s2] = score_tensor_view.multiindex(k);
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
    }
  }
};

}  // namespace boba
