// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <type_traits>
#include <utility>

namespace boba
{

/**
 * \param[in] tensor Tensor to compress.
 * \param[in] rank Rank to which the CPD will approximate `tensor`.
 * \param[in] ALS_tolerance_relative Optional relative tolerance override.
 * \param[in] ALS_tolerance_absolute Optional absolute tolerance override.
 * \return The CPD approximating `tensor` to the given rank.
 */

template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> compress_to_cpd(
  const Tensor<dimension, space, data_t>& tensor,
  size_t rank,
  data_t ALS_tolerance_relative = -1.0,
  data_t ALS_tolerance_absolute = -1.0)
{
  BOBA_CALI_MARK
  CanonicalPolyadicDecomposition<dimension, space, data_t> new_cpd(tensor.sizes());
  if (ALS_tolerance_relative > 0.0)
  {
    new_cpd.ALS_tolerance_relative = ALS_tolerance_relative;
  }
  if (ALS_tolerance_absolute > 0.0)
  {
    new_cpd.ALS_tolerance_absolute = ALS_tolerance_absolute;
  }
  new_cpd.compress(tensor, rank);
  return new_cpd;
}

/**
 * \param[in] vectors Basis vectors used to create the CPD.
 * \return A rank-1 CPD using the vectors as basis functions.
 */

template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> make_cpd_from_vectors(Array<Vector<space, data_t>, dimension> const& vectors)
{
  BOBA_CALI_MARK
  Array<size_t, dimension> sizes;
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] = vectors[d].size();
  }

  CanonicalPolyadicDecomposition<dimension, space, data_t> new_cpd(sizes);
  new_cpd.fill_with(1.0);

  for (size_t d = 0; d < dimension; d++)
  {
    new_cpd.cores[d].reshape(vectors[d]);
  }
  return new_cpd;
}

/**
 * \brief Iterates through ranks and prints metrics resulting from CPD compression.
 * Useful for exploring the tradeoff between compression rate and error.
 * \param[in,out] tensor Tensor to compress.
 * \param[in] max_rank Maximum rank to test.
 */

template <size_t dimension, execution_space space, typename data_t>
void scan_cpd_compression_versus_rank(
  Tensor<dimension, space, data_t>& tensor,
  size_t max_rank = 20)
{
  BOBA_CALI_MARK
  std::cout
    << "Rank CR Inf_Error Frob_error Rel_Inf_Error Rel_Frob_Error" << std::endl;
  for (size_t rank = 1; rank < max_rank; rank++)
  {
    auto compressed_tensor = compress_to_cpd(tensor, rank);
    auto cr = compressed_tensor.compression_rate();
    auto norm_inf = ::boba::norm_inf(tensor);
    auto norm_frob = ::boba::norm_frobenius(tensor);
    auto error_inf = ::boba::norm_difference_inf(compressed_tensor.decompress(), tensor);
    auto error_frob = ::boba::norm_difference_frobenius(compressed_tensor.decompress(), tensor);
    auto rel_error_frob = error_frob / norm_frob;
    auto rel_error_inf = error_inf / norm_inf;
    std::cout
      << rank << " " << cr << " "
      << error_inf << " " << error_frob << " "
      << rel_error_inf << " " << rel_error_frob << std::endl;
  }
}

// -------------------------------------------------------------------------------------
// Section: Alternating Poisson regression
// -------------------------------------------------------------------------------------

/**
 * \brief Controls canonical polyadic alternating Poisson regression (CP-APR).
 *
 * CP-APR fits a nonnegative CP decomposition to count data by minimizing the
 * generalized Kullback-Leibler divergence. Parameter names follow Algorithm 3
 * in Chi and Kolda, "On Tensors, Sparsity, and Nonnegative Factorizations",
 * SIAM Journal on Matrix Analysis and Applications 33(4), 2012.
 * \see https://doi.org/10.1137/110859063
 */
template <typename data_t>
struct CPAPRParameters
{
  /// Maximum number of outer sweeps over all tensor modes.
  size_t max_outer_iterations = 200;
  /// Maximum multiplicative updates for one mode in each outer sweep.
  size_t max_inner_iterations = 10;
  /// KKT convergence tolerance.
  data_t tolerance = static_cast<data_t>(1.0e-4);
  /// Offset applied to variables that violate complementary slackness.
  data_t complementary_slackness_offset = static_cast<data_t>(1.0e-2);
  /// Factor-entry threshold used to identify active constraints.
  data_t active_set_tolerance = static_cast<data_t>(1.0e-10);
  /// Positive lower bound for model predictions used as divisors.
  data_t prediction_floor = static_cast<data_t>(1.0e-10);
};

namespace detail
{

/// Internal implementation details for canonical polyadic alternating Poisson regression.
namespace cpd_apr
{

/**
 * \brief Normalizes each factor column to sum to one.
 *
 * A column with zero sum is replaced by a uniform column so that every output
 * column is a valid nonnegative distribution.
 *
 * \param[in,out] factor Factor matrix to normalize columnwise.
 */
template <execution_space space, typename data_t>
void column_sum_normalize(Matrix<space, data_t>& factor)
{
  Vector<space, data_t> column_sums({factor.cols()});
  column_sums.fill_with_zeros();

  auto factor_view = factor.view();
  auto sums_view = column_sums.atomic_view();
  ::boba::loop<space, 2>(factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
  {
    sums_view(ij[1]) += factor_view(ij);
  });

  auto sums_const_view = column_sums.const_view();
  const data_t uniform = static_cast<data_t>(1) / static_cast<data_t>(factor.rows());
  ::boba::loop<space, 2>(factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
  {
    const data_t sum = sums_const_view(ij[1]);
    factor_view(ij) = sum > data_t{}
                        ? factor_view(ij) * (static_cast<data_t>(1) / sum)
                        : uniform;
  });
}

/**
 * \brief Computes the CP-APR multiplicative-update matrix for sparse data.
 *
 * Only explicitly stored entries contribute. Contributions are accumulated
 * atomically because multiple entries may share the same active-mode index.
 *
 * \param[in] tensor Sparse nonnegative count tensor.
 * \param[in] mode Active tensor mode.
 * \param[in] model CP model supplying factors outside the active mode.
 * \param[in] absorbed_factor Active factor with component weights absorbed.
 * \param[in] prediction_floor Positive lower bound for model predictions.
 * \param[out] phi Multiplicative-update matrix for the active mode.
 */
template <size_t dimension, execution_space space, typename data_t>
void compute_phi(
  const SparseTensor<dimension, space, data_t>& tensor,
  index_t mode,
  const CanonicalPolyadicDecomposition<dimension, space, data_t>& model,
  const Matrix<space, data_t>& absorbed_factor,
  data_t prediction_floor,
  Matrix<space, data_t>& phi)
{
  phi.fill_with_zeros();

  const auto tensor_view = tensor.const_view();
  const auto cores = model.get_core_const_views();
  const auto factor_view = absorbed_factor.const_view();
  auto phi_view = phi.atomic_view();
  const index_t rank = model.rank();

  ::boba::loop<space, 1>(0_z, tensor.number_nonzeros(), [=] __boba_host_device__(index_t entry)
  {
    const auto indices = tensor_view.entry_multiindex(entry);
    data_t prediction{};
    for (index_t component = 0; component < rank; ++component)
    {
      data_t product = static_cast<data_t>(1);
      for (index_t d = 0; d < dimension; ++d)
      {
        if (d != mode)
        {
          product *= cores[d]({indices[d], component});
        }
      }
      prediction += product * factor_view({indices[mode], component});
    }

    prediction = ::boba::max(prediction, prediction_floor);
    const data_t scaled_value = tensor_view.values()[entry] / prediction;
    for (index_t component = 0; component < rank; ++component)
    {
      data_t product = static_cast<data_t>(1);
      for (index_t d = 0; d < dimension; ++d)
      {
        if (d != mode)
        {
          product *= cores[d]({indices[d], component});
        }
      }
      phi_view({indices[mode], component}) += scaled_value * product;
    }
  });
}

/**
 * \brief Computes the CP-APR multiplicative-update matrix for dense data.
 *
 * The dense storage is traversed completely, but nonpositive entries are
 * skipped because they do not contribute to the update numerator.
 *
 * \param[in] tensor Dense nonnegative count tensor.
 * \param[in] mode Active tensor mode.
 * \param[in] model CP model supplying factors outside the active mode.
 * \param[in] absorbed_factor Active factor with component weights absorbed.
 * \param[in] prediction_floor Positive lower bound for model predictions.
 * \param[out] phi Multiplicative-update matrix for the active mode.
 */
template <size_t dimension, execution_space space, typename data_t>
void compute_phi(
  const Tensor<dimension, space, data_t>& tensor,
  index_t mode,
  const CanonicalPolyadicDecomposition<dimension, space, data_t>& model,
  const Matrix<space, data_t>& absorbed_factor,
  data_t prediction_floor,
  Matrix<space, data_t>& phi)
{
  phi.fill_with_zeros();

  const auto tensor_view = tensor.const_view();
  const auto cores = model.get_core_const_views();
  const auto factor_view = absorbed_factor.const_view();
  auto phi_view = phi.atomic_view();
  const index_t rank = model.rank();

  ::boba::loop<space, 1>(0_z, tensor.size(), [=] __boba_host_device__(index_t entry)
  {
    const data_t value = tensor_view(entry);
    if (value <= data_t{})
    {
      return;
    }

    const auto indices = tensor_view.multiindex(entry);
    data_t prediction{};
    for (index_t component = 0; component < rank; ++component)
    {
      data_t product = static_cast<data_t>(1);
      for (index_t d = 0; d < dimension; ++d)
      {
        if (d != mode)
        {
          product *= cores[d]({indices[d], component});
        }
      }
      prediction += product * factor_view({indices[mode], component});
    }

    prediction = ::boba::max(prediction, prediction_floor);
    const data_t scaled_value = value / prediction;
    for (index_t component = 0; component < rank; ++component)
    {
      data_t product = static_cast<data_t>(1);
      for (index_t d = 0; d < dimension; ++d)
      {
        if (d != mode)
        {
          product *= cores[d]({indices[d], component});
        }
      }
      phi_view({indices[mode], component}) += scaled_value * product;
    }
  });
}

/**
 * \brief Fits a CP model by alternating Poisson-regression mode updates.
 *
 * The supplied work arrays are resized as needed and retained by the caller
 * for reuse across fits.
 *
 * \param[in] tensor Dense or sparse nonnegative count tensor.
 * \param[in] model Initial nonnegative CP model.
 * \param[in] parameters Iteration limits and numerical tolerances.
 * \param[in,out] absorbed_factors Work matrices for weight-absorbed factors.
 * \param[in,out] phi_factors Work matrices for multiplicative updates.
 * \param[in,out] slack_factors Work matrices for active-set corrections.
 * \return The fitted CP model with normalized factor columns.
 */
template <typename tensor_t, size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> fit_impl(
  const tensor_t& tensor,
  CanonicalPolyadicDecomposition<dimension, space, data_t> model,
  const CPAPRParameters<data_t>& parameters,
  Array<Matrix<space, data_t>, dimension>& absorbed_factors,
  Array<Matrix<space, data_t>, dimension>& phi_factors,
  Array<Matrix<space, data_t>, dimension>& slack_factors)
{
  BOBA_CALI_MARK
  static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "CP-APR requires a real data type");

  boba_always_assert_positive(model.rank(), "CP-APR rank must be positive");
  boba_always_assert_positive(parameters.max_outer_iterations, "CP-APR max_outer_iterations must be positive");
  boba_always_assert_positive(parameters.max_inner_iterations, "CP-APR max_inner_iterations must be positive");
  boba_always_assert_positive(parameters.tolerance, "CP-APR tolerance must be positive");
  boba_always_assert_nonnegative(parameters.complementary_slackness_offset, "CP-APR complementary_slackness_offset must be nonnegative");
  boba_always_assert_nonnegative(parameters.active_set_tolerance, "CP-APR active_set_tolerance must be nonnegative");
  boba_always_assert_positive(parameters.prediction_floor, "CP-APR prediction_floor must be positive");

  for (index_t mode = 0; mode < dimension; ++mode)
  {
    boba_always_assert_equal(model.sizes(mode), tensor.sizes(mode), "CP-APR model and tensor sizes must match");
  }

  for (index_t mode = 0; mode < dimension; ++mode)
  {
    const auto shape = Array<index_t, 2>{model.sizes(mode), model.rank()};
    absorbed_factors[mode].resize(shape);
    phi_factors[mode].resize(shape);
    slack_factors[mode].resize(shape);
  }

  for (size_t outer = 0; outer < parameters.max_outer_iterations; ++outer)
  {
    bool converged = true;

    for (index_t mode = 0; mode < dimension; ++mode)
    {
      auto& absorbed_factor = absorbed_factors[mode];
      auto& phi = phi_factors[mode];
      auto& slack = slack_factors[mode];
      slack.fill_with_zeros();

      ::boba::apply_as_diagonal_right(model.m_weights, model.m_cores[mode], absorbed_factor);

      auto absorbed_view = absorbed_factor.view();
      const auto factor_view = model.m_cores[mode].const_view();
      const auto weights_view = model.m_weights.const_view();

      if (outer > 0)
      {
        compute_phi(tensor, mode, model, absorbed_factor, parameters.prediction_floor, phi);
        auto slack_view = slack.view();
        const auto phi_view = phi.const_view();
        ::boba::loop<space, 2>(absorbed_factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
        {
          if (factor_view(ij) < parameters.active_set_tolerance && phi_view(ij) > static_cast<data_t>(1))
          {
            slack_view(ij) = parameters.complementary_slackness_offset;
          }
        });
      }

      const auto slack_view = slack.const_view();
      ::boba::loop<space, 2>(absorbed_factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
      {
        absorbed_view(ij) = (factor_view(ij) + slack_view(ij)) * weights_view(ij[1]);
      });

      for (size_t inner = 0; inner < parameters.max_inner_iterations; ++inner)
      {
        compute_phi(tensor, mode, model, absorbed_factor, parameters.prediction_floor, phi);

        data_t kkt_violation{};
        const auto phi_view = phi.const_view();
        const auto absorbed_const_view = absorbed_factor.const_view();
        ::boba::max_reduce<space>(kkt_violation, 0_z, absorbed_factor.size(), [=] __boba_host_device__(index_t flat, max_reducer_operator<data_t> & local_max)
        {
          const auto ij = absorbed_const_view.multiindex(flat);
          local_max.max(::boba::abs(::boba::min(absorbed_const_view(ij), static_cast<data_t>(1) - phi_view(ij))));
        });

        if (kkt_violation < parameters.tolerance)
        {
          break;
        }

        converged = false;
        ::boba::loop<space, 2>(absorbed_factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
        {
          absorbed_view(ij) *= phi_view(ij);
        });
      }

      model.m_weights.fill_with_zeros();
      auto new_weights_view = model.m_weights.atomic_view();
      const auto absorbed_const_view = absorbed_factor.const_view();
      ::boba::loop<space, 2>(absorbed_factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
      {
        new_weights_view(ij[1]) += absorbed_const_view(ij);
      });

      auto updated_factor_view = model.m_cores[mode].view();
      const auto new_weights_const_view = model.m_weights.const_view();
      const data_t uniform = static_cast<data_t>(1) / static_cast<data_t>(absorbed_factor.rows());
      ::boba::loop<space, 2>(absorbed_factor.sizes(), [=] __boba_host_device__(Array<index_t, 2> ij)
      {
        const data_t weight = new_weights_const_view(ij[1]);
        updated_factor_view(ij) = weight > data_t{}
                                    ? absorbed_const_view(ij) * (static_cast<data_t>(1) / weight)
                                    : uniform;
      });
      column_sum_normalize(model.m_cores[mode]);
    }

    if (converged)
    {
      break;
    }
  }

  return model;
}

/**
 * \brief Creates a randomly initialized positive CP model.
 *
 * Factor entries are generated on the host, shifted away from zero, normalized
 * columnwise, and transferred to the model's execution space. Component
 * weights are initialized to one.
 *
 * \param[in] tensor Tensor whose extents and execution space define the model.
 * \param[in] rank Number of CP components.
 * \param[in,out] random_generator Generator used for factor entries.
 * \return A positive rank-\p rank CP model with normalized factor columns.
 */
template <typename tensor_t, typename random_generator_t>
auto make_initial_guess(
  const tensor_t& tensor,
  index_t rank,
  random_generator_t& random_generator)
{
  using data_t = typename tensor_t::data_t;
  constexpr auto space = tensor_t::get_space();
  constexpr size_t dimension = tensor_t::get_dimension();

  static_assert(std::is_floating_point_v<data_t>, "CP-APR requires a floating-point data type");
  boba_always_assert_positive(rank, "CP-APR rank must be positive");
  CanonicalPolyadicDecomposition<dimension, space, data_t> model(tensor.sizes());
  model.rename("cp_apr");
  model.m_weights.resize({rank});
  model.m_weights.fill_with(static_cast<data_t>(1));

  const data_t initialization_offset = static_cast<data_t>(1.0e-6);
  for (index_t mode = 0; mode < dimension; ++mode)
  {
    const auto shape = Array<index_t, 2>{
      tensor.sizes(mode), rank};
    model.m_cores[mode].resize(shape);
    model.m_cores[mode].fill_with_random(
      initialization_offset,
      static_cast<data_t>(1) + initialization_offset,
      random_generator);
    column_sum_normalize(model.m_cores[mode]);
  }
  return model;
}

} // namespace cpd_apr
} // namespace detail

/**
 * \brief Stateful canonical polyadic alternating Poisson regression solver.
 *
 * The solver owns its parameters and work matrices. Repeated fits with the
 * same extents and rank reuse the absorbed-factor, Phi, and slack allocations;
 * changed extents or rank resize them. Input tensors and fitted models are
 * owned by the caller.
 *
 * \note Calls on the same solver instance must not overlap.
 *
 * \tparam dimension Tensor order.
 * \tparam space Execution space used by tensors, models, and work matrices.
 * \tparam data_t Real floating-point value type.
 */
template <size_t dimension, execution_space space, typename data_t>
class CPAPRSolver
{
public:
  using model_type = CanonicalPolyadicDecomposition<dimension, space, data_t>;
  using dense_tensor_type = Tensor<dimension, space, data_t>;
  using sparse_tensor_type = SparseTensor<dimension, space, data_t>;

  /**
   * \brief Constructs a solver with the supplied iteration parameters.
   * \param[in] parameters Iteration limits and numerical tolerances.
   */
  explicit CPAPRSolver(CPAPRParameters<data_t> parameters = {})
    : parameters_(std::move(parameters))
  {
  }

  /** \return Mutable solver parameters used by subsequent fits. */
  CPAPRParameters<data_t>& parameters() noexcept
  {
    return parameters_;
  }

  /** \return Solver parameters used by subsequent fits. */
  const CPAPRParameters<data_t>& parameters() const noexcept
  {
    return parameters_;
  }

  /**
   * \brief Fits an initial CP model to a dense count tensor.
   * \param[in] tensor Dense nonnegative count tensor.
   * \param[in] initial_model Initial nonnegative model and requested rank.
   * \return The fitted CP model.
   */
  [[nodiscard]] model_type fit(
    const dense_tensor_type& tensor,
    model_type initial_model)
  {
    return fit_impl(tensor, std::move(initial_model));
  }

  /**
   * \brief Fits an initial CP model to a sparse count tensor.
   * \param[in] tensor Sparse nonnegative count tensor.
   * \param[in] initial_model Initial nonnegative model and requested rank.
   * \return The fitted CP model.
   */
  [[nodiscard]] model_type fit(
    const sparse_tensor_type& tensor,
    model_type initial_model)
  {
    return fit_impl(tensor, std::move(initial_model));
  }

  /**
   * \brief Fits a randomly initialized rank-\p rank model to count data.
   * \param[in] tensor Dense or sparse nonnegative count tensor.
   * \param[in] rank Number of CP components.
   * \param[in,out] random_generator Generator used to initialize the factors.
   * \return The fitted CP model.
   */
  template <typename tensor_t, typename random_generator_t>
  [[nodiscard]] model_type fit(
    const tensor_t& tensor,
    size_t rank,
    random_generator_t& random_generator)
  {
    static_assert(tensor_t::get_dimension() == dimension, "CP-APR tensor order must match the solver");
    static_assert(tensor_t::get_space() == space, "CP-APR tensor execution space must match the solver");
    static_assert(std::is_same_v<typename tensor_t::data_t, data_t>, "CP-APR tensor data type must match the solver");
    auto initial_model = detail::cpd_apr::make_initial_guess(tensor, rank, random_generator);
    return fit(tensor, std::move(initial_model));
  }

  /**
   * \brief Fits a randomly initialized rank-\p rank model to count data.
   *
   * This overload uses and advances `boba::random::default_context()`.
   *
   * \param[in] tensor Dense or sparse nonnegative count tensor.
   * \param[in] rank Number of CP components.
   * \return The fitted CP model.
   */
  template <typename tensor_t>
  [[nodiscard]] model_type fit(const tensor_t& tensor, size_t rank)
  {
    auto& random_generator = ::boba::random::default_context();
    return fit(tensor, rank, random_generator);
  }

private:
  template <typename tensor_t>
  model_type fit_impl(const tensor_t& tensor, model_type initial_model)
  {
    return detail::cpd_apr::fit_impl(
      tensor,
      std::move(initial_model),
      parameters_,
      absorbed_factors_,
      phi_factors_,
      slack_factors_);
  }

  CPAPRParameters<data_t> parameters_;
  Array<Matrix<space, data_t>, dimension> absorbed_factors_;
  Array<Matrix<space, data_t>, dimension> phi_factors_;
  Array<Matrix<space, data_t>, dimension> slack_factors_;
};

/**
 * \brief Fits a supplied nonnegative CP model to a dense count tensor with CP-APR.
 *
 * The input tensor and initial model must be nonnegative and have identical
 * extents. Factor columns in the returned model sum to one; component masses
 * are stored in its weights.
 *
 * One full outer sweep costs
 * `O(max_inner_iterations * tensor.size() * rank * dimension^2)`.
 *
 * \param[in] tensor Dense count tensor.
 * \param[in] initial_model Initial nonnegative model and requested rank.
 * \param[in] parameters Iteration limits and numerical tolerances.
 * \return The fitted CP model.
 */
template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> cp_apr_fit(
  const Tensor<dimension, space, data_t>& tensor,
  CanonicalPolyadicDecomposition<dimension, space, data_t> initial_model,
  const CPAPRParameters<data_t>& parameters = {})
{
  CPAPRSolver<dimension, space, data_t> solver(parameters);
  return solver.fit(tensor, std::move(initial_model));
}

/**
 * \brief Fits a supplied nonnegative CP model to a sparse count tensor with CP-APR.
 *
 * Only explicitly stored entries contribute to the mode updates, so this
 * overload avoids work proportional to the full tensor size. The input tensor
 * and initial model must be nonnegative and have identical extents.
 * One full outer sweep costs
 * `O(max_inner_iterations * number_nonzeros * rank * dimension^2)`.
 *
 * \param[in] tensor Sparse count tensor.
 * \param[in] initial_model Initial nonnegative model and requested rank.
 * \param[in] parameters Iteration limits and numerical tolerances.
 * \return The fitted CP model.
 */
template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> cp_apr_fit(
  const SparseTensor<dimension, space, data_t>& tensor,
  CanonicalPolyadicDecomposition<dimension, space, data_t> initial_model,
  const CPAPRParameters<data_t>& parameters = {})
{
  CPAPRSolver<dimension, space, data_t> solver(parameters);
  return solver.fit(tensor, std::move(initial_model));
}

/**
 * \brief Fits a randomly initialized rank-\p rank CP model to count data.
 * \param[in] tensor Dense or sparse nonnegative count tensor.
 * \param[in] rank Number of CP components.
 * \param[in] parameters Iteration limits and numerical tolerances.
 * \param[in,out] random_generator Generator used to initialize the factor matrices.
 * \return The fitted CP model.
 */
template <typename tensor_t, typename random_generator_t>
auto cp_apr(
  const tensor_t& tensor,
  size_t rank,
  const CPAPRParameters<typename tensor_t::data_t>& parameters,
  random_generator_t& random_generator)
{
  constexpr size_t dimension = tensor_t::get_dimension();
  constexpr execution_space space = tensor_t::get_space();
  using data_t = typename tensor_t::data_t;
  CPAPRSolver<dimension, space, data_t> solver(parameters);
  return solver.fit(tensor, rank, random_generator);
}

/**
 * \brief Fits a randomly initialized rank-\p rank CP model to count data.
 *
 * This overload uses and advances `boba::random::default_context()`.
 *
 * \param[in] tensor Dense or sparse nonnegative count tensor.
 * \param[in] rank Number of CP components.
 * \param[in] parameters Iteration limits and numerical tolerances.
 * \return The fitted CP model.
 */
template <typename tensor_t>
auto cp_apr(
  const tensor_t& tensor,
  size_t rank,
  const CPAPRParameters<typename tensor_t::data_t>& parameters = {})
{
  auto& random_generator = ::boba::random::default_context();
  return cp_apr(tensor, rank, parameters, random_generator);
}

} // namespace boba
