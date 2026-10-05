// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <size_t dimension, execution_space space, typename data_t>
void nan_check(CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(cpd.m_cores[d]);
  }
  ::boba::nan_check(cpd.m_weights);
}

// -------------------------------------------------------------------------------------
// Section: Cumulative sum (CDF)
// -------------------------------------------------------------------------------------

/**
 * \brief Computes the multi-dimensional cumulative sum of a CPD (a discrete CDF over bins).
 *
 * For a CPD representing `pdf(i) = sum_r w_r * prod_d A_d(i_d, r)`, this returns a CPD
 * representing the cumulative sum over axis-aligned lower orthants:
 * `cdf(i) = sum_{j<=i} pdf(j)`.
 *
 * The returned CPD is computed by replacing each factor matrix `A_d(:, r)` with its
 * 1D prefix sum along the row index; weights are unchanged.
 */
template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t>
cumulative_sum(const CanonicalPolyadicDecomposition<dimension, space, data_t>& pdf)
{
  BOBA_CALI_MARK

  CanonicalPolyadicDecomposition<dimension, space, data_t> cdf(pdf);
  cdf.rename(pdf.name() + "_cumulative_sum");

  for (size_t d = 0; d < dimension; ++d)
  {
    auto core_view = cdf.m_cores[d].view();
    const index_t rows = cdf.m_cores[d].rows();
    const index_t cols = cdf.m_cores[d].cols();

    ::boba::detail::loop<space>(0_z, static_cast<size_t>(cols), [=] __boba_host_device__(size_t r)
    {
      const auto rr = static_cast<index_t>(r);
      for (index_t i = 1; i < rows; ++i)
      {
        core_view({i, rr}) += core_view({i - 1, rr});
      }
    });
  }

  return cdf;
}

// -------------------------------------------------------------------------------------
// Section: Linear prefix sum
// -------------------------------------------------------------------------------------

/**
 * \brief Computes the inclusive prefix sum in the represented tensor's linear storage order.
 *
 * For each input rank, the output contains one rank-one term per dimension. In term `d`,
 * factors before `d` are replaced by their full sums, factor `d` is replaced by its
 * inclusive prefix for `d == 0` or its exclusive prefix otherwise, and later factors
 * are unchanged. The result therefore has rank `dimension * input.rank()` and is
 * formed without decompressing the CPD.
 *
 * For example, for one rank of a three-dimensional CPD with factors `A`, `B`, and `C`,
 * let `A_i` denote the inclusive prefix of `A`, let `B_e` and `C_e` denote the
 * exclusive prefixes of `B` and `C`, and let `a` and `b` denote constant vectors
 * containing the full sums of `A` and `B`. The linear prefix sum is represented by
 * the three rank-one terms
 *
 * ```text
 * A_i B C + a B_e C + a b C_e.
 * ```
 *
 * The implementation below repeats this pattern once per dimension and per input rank.
 */
template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t>
linear_prefix_sum(const CanonicalPolyadicDecomposition<dimension, space, data_t>& input)
{
  BOBA_CALI_MARK

  // The fastest-varying factor needs an inclusive prefix because its term includes
  // the current entry. Every later factor needs an exclusive prefix because its term
  // represents only the lower-index slices that precede the current slice.
  auto prefix_cores = input.m_cores;
  for (size_t d = 0; d < dimension; ++d)
  {
    auto core_view = prefix_cores[d].view();
    const index_t rows = prefix_cores[d].rows();
    const index_t cols = prefix_cores[d].cols();

    if (d == 0)
    {
      ::boba::detail::loop<space>(0_z, cols, [=] __boba_host_device__(size_t r)
      {
        for (index_t i = 1; i < rows; ++i)
        {
          core_view({i, r}) += core_view({i - 1, r});
        }
      });
    }
    else
    {
      ::boba::detail::loop<space>(0_z, cols, [=] __boba_host_device__(size_t r)
      {
        data_t prefix{};
        for (index_t i = 0; i < rows; ++i)
        {
          const data_t value = core_view({i, r});
          core_view({i, r}) = prefix;
          prefix += value;
        }
      });
    }
  }

  // The first group of rank-one terms uses the inclusive prefix in the
  // fastest-varying dimension and the original factors in all later dimensions.
  // Build it from only those factors instead of copying input and then overwriting
  // its first core.
  CanonicalPolyadicDecomposition<dimension, space, data_t> output;
  output.m_weights = input.m_weights;
  output.m_cores[0] = prefix_cores[0];
  for (size_t k = 1; k < dimension; ++k)
  {
    output.m_cores[k] = input.m_cores[k];
  }

  // Add one group of input.rank() terms for each remaining dimension d.
  for (size_t d = 1; d < dimension; ++d)
  {
    // Build the d'th term
    CanonicalPolyadicDecomposition<dimension, space, data_t> term;
    term.m_weights = input.m_weights;

    // All earlier dimensions have been traversed completely. Replace each earlier
    // factor column by a constant column containing that factor's full sum.
    for (size_t k = 0; k < d; ++k)
    {
      term.m_cores[k].resize(input.m_cores[k].sizes());
      auto constant_core_view = term.m_cores[k].view();
      auto prefix_core_view = prefix_cores[k].const_view();
      auto input_core_view = input.m_cores[k].const_view();
      const index_t last_row = prefix_cores[k].rows() - 1;
      // The inclusive prefix in dimension zero already ends with the full sum. An
      // exclusive prefix ends just before the final value, so add that value back.
      const bool prefix_is_exclusive = k != 0;

      ::boba::loop<space, 2>(constant_core_view.sizes(),
                             [=] __boba_host_device__(Array<index_t, 2> ij)
      {
        const auto last_prefix = prefix_core_view({last_row, ij[1]});
        const auto last_value = input_core_view({last_row, ij[1]});
        constant_core_view(ij) =
          last_prefix + (prefix_is_exclusive ? last_value : data_t{});
      });
    }

    // The exclusive prefix at dimension d contains only entries before the current
    // entry. Factors after d remain unchanged from the input CPD.
    term.m_cores[d] = prefix_cores[d];

    for (size_t k = d + 1; k < dimension; ++k)
    {
      term.m_cores[k] = input.m_cores[k];
    }

    // CPD addition concatenates weights and factor columns, increasing the rank by
    // input.rank() for each dimension-specific term.
    output += term;
  }

  output.rename(input.name() + "_linear_prefix_sum");
  return output;
}

// -------------------------------------------------------------------------------------
// Section: Norms
// -------------------------------------------------------------------------------------

/**
 * \return The Frobenius norm of a CPD.
 */

template <size_t dimension, execution_space space, typename data_t>
auto norm_frobenius(CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd)
{
  BOBA_CALI_MARK
  const auto product = cpd.inner_product(cpd);
  const auto abs_product = boba::abs(product);
  return boba::sqrt(abs_product);
}

// -------------------------------------------------------------------------
// sum and round
// -------------------------------------------------------------------------

/**
 * @brief Computes the sum of a sequence of CP tensors,
 * rounding after each addition.
 *
 * Each tensor in @p sequence is added to an accumulator, followed by a rounding
 * operation to maintain a low-rank representation.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space (e.g., host or device).
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::span of CanonicalPolyadicDecomposition objects to be summed and rounded.
 * @return A new CanonicalPolyadicDecomposition representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t>
sum_and_round(std::span<const CanonicalPolyadicDecomposition<dimension, space, data_t>> sequence)
{
  using cp_t = CanonicalPolyadicDecomposition<dimension, space, data_t>;

  boba_always_assert(!sequence.empty(), "Cannot perform sum_and_round on an empty sequence.");

  // Use the first tensor to infer shape
  const auto& first = sequence.front();
  cp_t output(first.sizes());

  for (const auto& item : sequence)
  {
    output += item;
    output.round();
  }

  return output;
}

/**
 * @brief Lightweight adapter for converting std::vector into std::span for
 * addition and rounding.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::vector of CanonicalPolyadicDecomposition objects to be summed and rounded.
 * @return A new CanonicalPolyadicDecomposition representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t>
sum_and_round(const std::vector<CanonicalPolyadicDecomposition<dimension, space, data_t>>& sequence)
{
  using cp_t = CanonicalPolyadicDecomposition<dimension, space, data_t>;
  return sum_and_round(std::span<const cp_t>{sequence});
}

/**
 * @brief Lightweight adapter for converting std::initializer_list into
 * std::span for addition and rounding.
 *
 * @note This overload is convenient, but it may copy the input tensors into the
 * initializer-list backing array. Prefer the std::span or std::vector overloads
 * for performance-sensitive code.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence Initializer list of CanonicalPolyadicDecomposition objects to be summed and rounded.
 * @return A new CanonicalPolyadicDecomposition representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t>
sum_and_round(const std::initializer_list<CanonicalPolyadicDecomposition<dimension, space, data_t>> sequence)
{
  using cp_t = CanonicalPolyadicDecomposition<dimension, space, data_t>;
  return sum_and_round(std::span<const cp_t>{sequence.begin(), sequence.size()});
}

// -------------------------------------------------------------------------------------
// Section: Multiply
// -------------------------------------------------------------------------------------

/**
 * \brief Performs the elementwise product, also known as the Hadamard product.
 * \param[in] cpd_A First CPD.
 * \param[in] cpd_B Second CPD.
 * \return A CPD representing `cpd_A * cpd_B`.
 */

template <size_t dimension, execution_space space, typename data_t>
CanonicalPolyadicDecomposition<dimension, space, data_t> elementwise_product(
  CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd_A,
  CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd_B)
{
  BOBA_CALI_MARK
  detail::ignore(cpd_A);
  detail::ignore(cpd_B);
  boba_error("Not yet implemented.");
  return CanonicalPolyadicDecomposition<dimension, space, data_t>{};
}

// -------------------------------------------------------------------------------------
// Section: Inner product
// -------------------------------------------------------------------------------------

/**
 * \return Inner product \f$\langle A, B \rangle\f$ of two CPDs.
 */

template <size_t dimension, execution_space space, typename data_t>
data_t inner_product(
  CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd_A,
  CanonicalPolyadicDecomposition<dimension, space, data_t> const& cpd_B)
{
  checkpoint();
  return cpd_A.inner_product(cpd_B);
}

/**
 * \brief Returns a scaled copy of the input CPD.
 * \param[in] scalar Scale factor.
 * \param[in] input Input CPD.
 * \return `scalar * input`.
 */
template <size_t dimension, ::boba::execution_space space, typename _data_t>
CanonicalPolyadicDecomposition<dimension, space, _data_t> operator*(_data_t scalar, CanonicalPolyadicDecomposition<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  CanonicalPolyadicDecomposition<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

// -------------------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------------------

/**
 * \brief Writes the CPD to file in a way consistent with Tensor::write_to_file.
 * \param[in] cpd CPD to write.
 * \param[in] filename Base filename. Uses the CPD name when empty.
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(const CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = cpd.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  for (size_t d = 0; d < dimension; d++)
  {
    boba::write_to_file(cpd.m_cores[d], print_filename + "_core_" + std::to_string(d));
  }
  boba::write_to_file(cpd.m_weights, print_filename + "_weights");
}

/**
 * \brief Reads a CPD from files generated by write_to_file.
 * \param[in,out] cpd CPD to populate.
 * \param[in] filename Base filename. Uses the CPD name when empty.
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = cpd.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  for (size_t d = 0; d < dimension; d++)
  {
    boba::read_from_file(cpd.m_cores[d], print_filename + "_core_" + std::to_string(d));
  }
  boba::read_from_file(cpd.m_weights, print_filename + "_weights");
}

/**
 * \brief Writes the CPD to a MATLAB mat-file as a cell array of cores.
 * \param[in] cpd CPD to write.
 * \param[in] filename Base filename. Uses the CPD name when empty.
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_mat_file(const CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = cpd.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  // TODO, fix this copy
  boba::Array<::boba::Tensor<2, space, data_t>, dimension> recast_cores;
  for (size_t d = 0; d < dimension; d++)
  {
    auto rows = cpd.m_cores[d].rows();
    auto cols = cpd.m_cores[d].cols();
    recast_cores[d] = reshape<2>(cpd.m_cores[d], {rows, cols});
  }

  detail::MatFile matlab_file(print_filename, "w");
  matlab_file.write_cell_array(print_filename, recast_cores);
  std::string print_weights = print_filename + "_weights";
  matlab_file.write_array(print_weights, cpd.m_weights);
}

/**
 * \brief Reads a CPD from a MATLAB mat-file written by write_to_mat_file.
 * \param[in,out] cpd CPD to populate.
 * \param[in] filename Base filename. Uses the CPD name when empty.
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_mat_file(CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = cpd.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  // TODO, fix this copy
  boba::Array<::boba::Tensor<2, space, data_t>, dimension> recast_cores;

  detail::MatFile matlab_file(print_filename, "r");
  matlab_file.read_cell_array(print_filename, recast_cores);
  for (size_t d = 0; d < dimension; d++)
  {
    cpd.m_cores[d] = reshape_to_matrix(recast_cores[d], recast_cores[d].sizes());
  }

  std::string print_weights = print_filename + "_weights";
  ::boba::Tensor<2, space, data_t> recast_weights;
  matlab_file.read_array(print_weights, recast_weights);
  cpd.m_weights = flatten(recast_weights);
}

/**
 * \brief
 * Write the cpd to a HDF5 file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(const CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = cpd.name();
  }

  detail::HDF5File h5_file(filename, "w");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.write_array(object_name + "_core_" + std::to_string(d), cpd.m_cores[d]);
  }
  h5_file.write_array(object_name + "_weights", cpd.m_weights);
}

/**
 * \brief
 * Read from a file generated by write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(CanonicalPolyadicDecomposition<dimension, space, data_t>& cpd, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = cpd.name();
  }

  detail::HDF5File h5_file(filename, "r");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.read_array(object_name + "_core_" + std::to_string(d), cpd.m_cores[d]);
  }
  h5_file.read_array(object_name + "_weights", cpd.m_weights);
}

// -------------------------------------------------------------------------------------
// Section: FFT
// -------------------------------------------------------------------------------------

/**
 * @returns fft along one dimension of the decomposition, consistent with fft_along_dimension
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
CanonicalPolyadicDecomposition<dimension, space, complex<data_t>> fft_along_dimension(
  const CanonicalPolyadicDecomposition<dimension, space, complex<data_t>>& input,
  index_t transform_dimension,
  fft_operation operation)
{
  auto output = input;
  auto extent_dimension = 0_z;
  output.m_cores[transform_dimension].reshape(fft_along_dimension<2, space, data_t>(input.m_cores[transform_dimension], extent_dimension, operation));
  return output;
}

} // namespace boba
