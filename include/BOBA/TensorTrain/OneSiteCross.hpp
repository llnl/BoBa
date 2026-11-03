// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cmath>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <vector>

namespace boba
{

/**
 * \brief One-site TT cross (alternating maxvol cross).
 *
 * The TT-cross of [1], with the rank growth ("kickrank") and validation-set stopping rule of tntorch's cross() [2]
 * (which follows [3] for the rank adaptation):
 *
 *  - ranks start at 1; the right index sets are seeded with random fibers;
 *  - each iteration sweeps left to right, then right to left. At each core the fibers through the current left and
 *    right index sets are sampled, QR-factored, and r pivot rows are chosen by maxvol (or DEIM); the core is the
 *    interpolant Q Q(I,:)^{-1}, and the pivots extend the nested index set for the next core. The first core is left
 *    as the sampled fibers;
 *  - convergence: relative L2 error on a fixed random set of distinct grid points (an estimate of the error, not a
 *    bound) below `tolerance`;
 *  - otherwise every interior rank grows by `kickrank` (capped by `max_rank` and by feasibility,
 *    r_k <= min(r_{k-1} n_{k-1}, r_{k+1} n_k)) by appending random fibers to the right index sets.
 *
 * Compared with DMRGCross (two-site), each step samples r n r entries instead of r n n r, and ranks are set by
 * kickrank and validation rather than by a two-site SVD, so for a given accuracy it usually needs fewer function
 * evaluations but can overshoot the rank (round the result afterwards).
 *
 * Pivot selection reuses DMRGCross's selectors (maxvol2 or DEIM, chosen by `submatrix_selection_type`).
 * apply() takes the number of cores at compile time, like DMRGCross; RuntimeOneSiteCross takes it at run time.
 * Sampling and the small dense factorizations run on the host.
 *
 * References:
 *  [1] I. Oseledets, E. Tyrtyshnikov, "TT-cross approximation for multidimensional arrays",
 *      Linear Algebra Appl. 432(1):70-88, 2010. doi:10.1016/j.laa.2009.07.024
 *  [2] M. Usvyatsov, R. Ballester-Ripoll, K. Schindler, "tntorch: Tensor Network Learning with PyTorch",
 *      J. Mach. Learn. Res. 23(208):1-6, 2022. https://jmlr.org/papers/v23/21-1197.html
 *      (cross() in https://github.com/rballester/tntorch)
 *  [3] S. Dolgov, R. Scheichl, "A hybrid alternating least squares--TT-cross algorithm for parametric PDEs",
 *      SIAM/ASA J. Uncertain. Quantif. 7(1):260-291, 2019. doi:10.1137/17M1138881
 *
 * @tparam data_t Scalar value type.
 */
template <typename data_t>
struct OneSiteCross : public DMRGCross<data_t>
{
  using base_t = DMRGCross<data_t>;
  using real_data_t = typename base_t::real_data_t;
  using entry_function_t = std::function<data_t(std::vector<size_t> const&)>;
  using multi_index_t = std::vector<size_t>;

  /// Stop when the relative L2 error on the validation set is below this.
  real_data_t tolerance = 1.0e-10;
  /// Interior rank growth per iteration (>= 1).
  size_t kickrank = 3;
  /// Cap on every rank (>= 1).
  size_t max_rank = 100;
  /// Maximum number of iterations (one left-to-right plus one right-to-left sweep each).
  size_t max_iterations = 25;
  /// Number of distinct validation points (the whole grid if it has fewer points).
  size_t validation_size = 1000;
  /// Remember every evaluated entry and never call the function twice at the same multi-index.
  bool cache_evaluations = false;

  /// Filled by apply().
  struct Info
  {
    size_t evaluations = 0;
    size_t iterations = 0;
    bool converged = false;
    real_data_t validation_error = 0;
  };
  Info info;

  OneSiteCross()
  {
    // Maxvol stops when no entry of Q Q(I,:)^{-1} exceeds 1 + maxvol_tolerance in magnitude.
    this->maxvol_tolerance = static_cast<real_data_t>(1.0e-2);
  }

  /**
   * @brief Approximates the tensor `function(i_0, ..., i_{d-1})`, `0 <= i_k < sizes[k]`, by one-site cross.
   * @tparam dimension Number of cores.
   * @param sizes Mode sizes.
   * @param function Entry-evaluation callable `data_t(::boba::Array<size_t, dimension> const&)`.
   * @param random_generator UniformRandomBitGenerator used for the random fibers and the validation set.
   * @return A host-space tensor train approximation.
   */
  template <size_t dimension, typename lambda_t, typename random_generator_t>
  [[nodiscard]]
  ::boba::TensorTrain<dimension, host_space, data_t> apply(
    ::boba::Array<index_t, dimension> const& sizes,
    lambda_t&& function,
    random_generator_t& random_generator)
  {
    std::vector<size_t> runtime_sizes(sizes.begin(), sizes.end());
    entry_function_t entry_function = [&function](std::vector<size_t> const& index)
    {
      ::boba::Array<size_t, dimension> array_index;
      for (size_t k = 0; k < dimension; k++)
      {
        array_index[k] = index[k];
      }
      return function(array_index);
    };
    return apply_erased(runtime_sizes, entry_function, random_generator).template to_static<dimension>();
  }

  /**
   * @brief Applies one-site cross using the process-wide default generator.
   */
  template <size_t dimension, typename lambda_t>
  [[nodiscard]]
  ::boba::TensorTrain<dimension, host_space, data_t> apply(
    ::boba::Array<index_t, dimension> const& sizes,
    lambda_t&& function)
  {
    auto& random_generator = ::boba::random::default_context();
    return apply(sizes, std::forward<lambda_t>(function), random_generator);
  }

  /**
   * @brief The algorithm, for any number of cores, on a type-erased entry function.
   */
  template <typename random_generator_t>
  [[nodiscard]]
  ::boba::RuntimeTensorTrain<host_space, data_t> apply_erased(
    std::vector<size_t> const& sizes,
    entry_function_t& function,
    random_generator_t& random_generator)
  {
    BOBA_CALI_MARK
    const size_t dimension = sizes.size();
    boba_always_assert(dimension >= 1, "OneSiteCross: empty shape");
    boba_always_assert(kickrank >= 1, "OneSiteCross: kickrank must be >= 1");
    boba_always_assert(max_rank >= 1, "OneSiteCross: max_rank must be >= 1");
    boba_always_assert(max_iterations >= 1, "OneSiteCross: max_iterations must be >= 1");
    boba_always_assert(validation_size >= 1, "OneSiteCross: validation_size must be >= 1");
    for (size_t k = 0; k < dimension; k++)
    {
      boba_always_assert(sizes[k] >= 1, "OneSiteCross: zero mode size");
    }

    info = Info{};
    std::map<multi_index_t, data_t> cache;
    auto evaluate = [&](multi_index_t const& index) -> data_t
    {
      if (cache_evaluations)
      {
        auto found = cache.find(index);
        if (found != cache.end())
        {
          return found->second;
        }
      }
      const data_t value = function(index);
      info.evaluations++;
      if (cache_evaluations)
      {
        cache.emplace(index, value);
      }
      return value;
    };
    auto random_mode_index = [&](size_t k)
    {
      return static_cast<size_t>(random_generator() % sizes[k]);
    };

    // ranks[k] is the rank between cores k-1 and k; boundary ranks are 1.
    std::vector<size_t> ranks(dimension + 1, 1);
    cap_ranks(ranks, sizes);

    // right_indices[k]: ranks[k+1] multi-indices over modes k+1..d-1 (empty for the last core).
    // left_indices[k]: ranks[k] multi-indices over modes 0..k-1 (empty for the first core).
    std::vector<std::vector<multi_index_t>> right_indices(dimension), left_indices(dimension);
    right_indices[dimension - 1] = {multi_index_t{}};
    for (size_t k = 0; k + 1 < dimension; k++)
    {
      for (size_t c = 0; c < ranks[k + 1]; c++)
      {
        right_indices[k].push_back(random_fiber(k, dimension, random_mode_index));
      }
    }

    // Validation set: distinct random grid points (the whole grid if it is smaller).
    std::vector<multi_index_t> validation_points = sample_distinct(sizes, random_mode_index);
    std::vector<data_t> validation_values;
    real_data_t validation_norm = 0;
    for (auto const& index : validation_points)
    {
      validation_values.push_back(evaluate(index));
      validation_norm += std::norm(validation_values.back());
    }
    validation_norm = std::sqrt(validation_norm);
    if (validation_norm == real_data_t(0))
    {
      validation_norm = std::numeric_limits<real_data_t>::min();
    }

    // Fibers of core k through the current index sets: a (ranks[k], sizes[k], ranks[k+1]) tensor.
    auto sample_core = [&](size_t k)
    {
      ::boba::Tensor<3, host_space, data_t> fibers({ranks[k], sizes[k], ranks[k + 1]});
      auto view = fibers.view();
      multi_index_t index(dimension);
      for (size_t a = 0; a < ranks[k]; a++)
      {
        for (size_t m = 0; m < k; m++)
        {
          index[m] = left_indices[k][a][m];
        }
        for (size_t i = 0; i < sizes[k]; i++)
        {
          index[k] = i;
          for (size_t c = 0; c < ranks[k + 1]; c++)
          {
            for (size_t m = 0; m + k + 1 < dimension; m++)
            {
              index[k + 1 + m] = right_indices[k][c][m];
            }
            view({a, i, c}) = evaluate(index);
          }
        }
      }
      return fibers;
    };

    ::boba::RuntimeTensorTrain<host_space, data_t> train;
    train.cores.resize(dimension);
    if (dimension == 1)
    {
      left_indices[0] = {multi_index_t{}};
      train.cores[0] = sample_core(0);
      info.converged = true;
      info.iterations = 1;
      return train;
    }

    for (size_t iteration = 0; iteration < max_iterations; iteration++)
    {
      // Left to right: interpolation cores 0..d-2 and the left index sets.
      left_indices[0] = {multi_index_t{}};
      for (size_t k = 0; k + 1 < dimension; k++)
      {
        auto fibers = sample_core(k);
        ::boba::Matrix<host_space, data_t> unfolding({ranks[k] * sizes[k], ranks[k + 1]});
        unfolding.reshape(fibers);
        auto [core, pivots] = interpolation_core(unfolding);
        train.cores[k].resize({ranks[k], sizes[k], ranks[k + 1]});
        train.cores[k].reshape(core);
        // Tensors are stored first index fastest, so unfolding row p is (a, i) = (p % ranks[k], p / ranks[k]).
        left_indices[k + 1].clear();
        for (size_t t = 0; t < pivots.size(); t++)
        {
          multi_index_t extended = left_indices[k][pivots[t] % ranks[k]];
          extended.push_back(pivots[t] / ranks[k]);
          left_indices[k + 1].push_back(extended);
        }
      }

      // Right to left: interpolation cores d-1..1 and the right index sets.
      for (size_t k = dimension - 1; k >= 1; k--)
      {
        auto fibers = sample_core(k);
        // Transposed right unfolding: rows (i, c), columns a.
        ::boba::Matrix<host_space, data_t> unfolding({ranks[k], sizes[k] * ranks[k + 1]});
        unfolding.reshape(fibers);
        auto transposed = unfolding.transpose();
        auto [core_transposed, pivots] = interpolation_core(transposed);
        auto core = core_transposed.transpose();
        train.cores[k].resize({ranks[k], sizes[k], ranks[k + 1]});
        train.cores[k].reshape(core);
        // Row p of the transposed unfolding is (i, c) = (p % sizes[k], p / sizes[k]).
        right_indices[k - 1].clear();
        for (size_t t = 0; t < pivots.size(); t++)
        {
          multi_index_t extended{pivots[t] % sizes[k]};
          auto const& previous = right_indices[k][pivots[t] / sizes[k]];
          extended.insert(extended.end(), previous.begin(), previous.end());
          right_indices[k - 1].push_back(extended);
        }
      }

      // The first core holds the sampled fibers through the new right index set.
      train.cores[0] = sample_core(0);

      // Validation error.
      real_data_t difference = 0;
      for (size_t s = 0; s < validation_points.size(); s++)
      {
        difference += std::norm(validation_values[s] - entry(train, validation_points[s]));
      }
      info.validation_error = std::sqrt(difference) / validation_norm;
      info.iterations = iteration + 1;
      if (info.validation_error < tolerance)
      {
        info.converged = true;
        break;
      }

      // Grow the ranks by appending random fibers to the right index sets.
      if (iteration + 1 < max_iterations)
      {
        std::vector<size_t> new_ranks = ranks;
        for (size_t k = 1; k < dimension; k++)
        {
          new_ranks[k] = std::min(max_rank, new_ranks[k] + kickrank);
        }
        cap_ranks(new_ranks, sizes);
        for (size_t k = 0; k + 1 < dimension; k++)
        {
          while (right_indices[k].size() < new_ranks[k + 1])
          {
            right_indices[k].push_back(random_fiber(k, dimension, random_mode_index));
          }
          right_indices[k].resize(new_ranks[k + 1]);
        }
        ranks = new_ranks;
      }
    }
    return train;
  }

protected:
  // Interpolation core for an unfolding M (rows x r, rows >= r): Q Q(I,:)^{-1} with Q = qr(M).Q and I the r pivot
  // rows of Q; returns the core (rows x r) and I.
  std::pair<::boba::Matrix<host_space, data_t>, std::vector<size_t>> interpolation_core(
    ::boba::Matrix<host_space, data_t>& unfolding) const
  {
    ::boba::QR<host_space, data_t> qr;
    qr(unfolding);
    ::boba::Matrix<host_space, data_t> q = qr.Q;
    auto selected = this->select_indices(q);
    auto pivot_rows = q.extract_rows(selected);
    auto core = right_backsolve(pivot_rows, q);
    std::vector<size_t> pivots(selected.size());
    auto selected_view = selected.const_view();
    for (size_t t = 0; t < pivots.size(); t++)
    {
      pivots[t] = static_cast<size_t>(selected_view(t));
    }
    return {core, pivots};
  }

  // r_k <= min(r_{k-1} n_{k-1}, r_{k+1} n_k), so every unfolding has at least as many rows as columns.
  static void cap_ranks(std::vector<size_t>& ranks, std::vector<size_t> const& sizes)
  {
    const size_t dimension = sizes.size();
    for (size_t k = 1; k < dimension; k++)
    {
      ranks[k] = std::min(ranks[k], ranks[k - 1] * sizes[k - 1]);
    }
    for (size_t k = dimension - 1; k >= 1; k--)
    {
      ranks[k] = std::min(ranks[k], ranks[k + 1] * sizes[k]);
    }
  }

  template <typename random_mode_t>
  static multi_index_t random_fiber(size_t k, size_t dimension, random_mode_t& random_mode_index)
  {
    multi_index_t fiber(dimension - 1 - k);
    for (size_t m = 0; m < fiber.size(); m++)
    {
      fiber[m] = random_mode_index(k + 1 + m);
    }
    return fiber;
  }

  template <typename random_mode_t>
  std::vector<multi_index_t> sample_distinct(std::vector<size_t> const& sizes, random_mode_t& random_mode_index) const
  {
    double grid_size = 1.0;
    for (auto n : sizes)
    {
      grid_size *= static_cast<double>(n);
    }
    std::vector<multi_index_t> points;
    if (static_cast<double>(validation_size) >= grid_size)
    {
      // Whole grid, first mode fastest.
      multi_index_t index(sizes.size(), 0);
      for (size_t count = 0; count < static_cast<size_t>(grid_size); count++)
      {
        points.push_back(index);
        for (size_t k = 0; k < sizes.size() and ++index[k] == sizes[k]; k++)
        {
          index[k] = 0;
        }
      }
      return points;
    }
    std::set<multi_index_t> seen;
    while (points.size() < validation_size)
    {
      multi_index_t index(sizes.size());
      for (size_t k = 0; k < sizes.size(); k++)
      {
        index[k] = random_mode_index(k);
      }
      if (seen.insert(index).second)
      {
        points.push_back(index);
      }
    }
    return points;
  }

  // One entry of a host train: product of the core slices along the multi-index.
  static data_t entry(::boba::RuntimeTensorTrain<host_space, data_t> const& train, multi_index_t const& index)
  {
    std::vector<data_t> row{data_t(1)};
    for (size_t k = 0; k < train.cores.size(); k++)
    {
      auto view = train.cores[k].const_view();
      const size_t left = train.cores[k].sizes(0), right = train.cores[k].sizes(2);
      std::vector<data_t> next(right, data_t(0));
      for (size_t a = 0; a < left; a++)
      {
        for (size_t c = 0; c < right; c++)
        {
          next[c] += row[a] * view({a, index[k], c});
        }
      }
      row.swap(next);
    }
    return row[0];
  }
};

}  // namespace boba
