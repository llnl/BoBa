// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

namespace detail
{

/**
 * \brief Returns true when a sparse value is tiny relative to a scale.
 *
 * A positive \p tolerance is interpreted as the scale in `is_tiny(abs(value) /
 * tolerance)`. A non-positive tolerance falls back to exact structural zero
 * detection.
 *
 * \param value Sparse value to test.
 * \param tolerance Scale used for the relative tiny check.
 * \return True when \p value should be treated as structurally zero.
 */
template <typename data_t>
bool sparse_value_is_tiny(data_t value, real_type_t<data_t> tolerance)
{
  using real_data_t = real_type_t<data_t>;
  const real_data_t magnitude = ::boba::abs(value);
  if (!(tolerance > real_data_t{}))
  {
    return !(magnitude > real_data_t{});
  }
  return ::boba::is_tiny(magnitude / tolerance);
}

/**
 * \brief Checks whether two sparse entries agree on contracted dimensions.
 *
 * \param indices_A Multi-index from the left tensor.
 * \param indices_B Multi-index from the right tensor.
 * \param contraction_dimensions_A Contracted dimensions in the left tensor.
 * \param contraction_dimensions_B Contracted dimensions in the right tensor.
 * \return True when every contracted index pair has the same value.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
bool sparse_contraction_indices_match(
  Array<index_t, dimension_A> indices_A,
  Array<index_t, dimension_B> indices_B,
  Array<size_t, contractions> contraction_dimensions_A,
  Array<size_t, contractions> contraction_dimensions_B)
{
  for (std::size_t c = 0; c < contractions; ++c)
  {
    if (indices_A[contraction_dimensions_A[c]] != indices_B[contraction_dimensions_B[c]])
    {
      return false;
    }
  }
  return true;
}

/**
 * \brief Builds the output multi-index for a matching sparse contraction pair.
 *
 * The output index order matches the dense tensor_contraction convention:
 * first all uncontracted dimensions of A in their original order, then all
 * uncontracted dimensions of B in their original order.
 *
 * \param indices_A Multi-index from the left tensor.
 * \param indices_B Multi-index from the right tensor.
 * \param contraction_dimensions_A Contracted dimensions in the left tensor.
 * \param contraction_dimensions_B Contracted dimensions in the right tensor.
 * \return Multi-index in the contracted output tensor.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
Array<index_t, dimension_A + dimension_B - 2 * contractions> sparse_contraction_output_indices(
  Array<index_t, dimension_A> indices_A,
  Array<index_t, dimension_B> indices_B,
  Array<size_t, contractions> contraction_dimensions_A,
  Array<size_t, contractions> contraction_dimensions_B)
{
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;
  Array<index_t, output_dimension> output_indices;
  std::size_t output_dimension_id = 0;
  for (std::size_t d = 0; d < dimension_A; ++d)
  {
    bool is_contracted = false;
    for (std::size_t c = 0; c < contractions; ++c)
    {
      is_contracted = is_contracted || (contraction_dimensions_A[c] == d);
    }
    if (!is_contracted)
    {
      output_indices[output_dimension_id] = indices_A[d];
      ++output_dimension_id;
    }
  }
  for (std::size_t d = 0; d < dimension_B; ++d)
  {
    bool is_contracted = false;
    for (std::size_t c = 0; c < contractions; ++c)
    {
      is_contracted = is_contracted || (contraction_dimensions_B[c] == d);
    }
    if (!is_contracted)
    {
      output_indices[output_dimension_id] = indices_B[d];
      ++output_dimension_id;
    }
  }
  return output_indices;
}

/**
 * \brief Adds a contribution to a host sparse tensor entry.
 *
 * Contributions that are tiny relative to \p tolerance are ignored, and entries
 * whose accumulated value becomes tiny relative to \p tolerance are erased so
 * reductions and contractions keep nonzeros sparse.
 *
 * \param sparse Host sparse tensor to update.
 * \param indices Output multi-index receiving the contribution.
 * \param value Contribution to add.
 * \param tolerance Scale used for relative tiny checks.
 */
template <std::size_t dimension, typename sparse_tensor_t>
void sparse_accumulate(sparse_tensor_t& sparse,
                       Array<index_t, dimension> indices,
                       typename sparse_tensor_t::data_t value,
                       real_type_t<typename sparse_tensor_t::data_t> tolerance)
{
  if (::boba::detail::sparse_value_is_tiny(value, tolerance))
  {
    return;
  }

  const auto new_value = sparse(indices) + value;
  if (::boba::detail::sparse_value_is_tiny(new_value, tolerance))
  {
    sparse.erase(indices);
  }
  else
  {
    sparse.set(indices, new_value);
  }
}

} // namespace detail

/**
 * \brief Largest explicitly stored value magnitude of a sparse tensor.
 *
 * Missing entries are implicit zeros. An empty sparse value list therefore has
 * norm zero.
 *
 * \param tensor Sparse tensor to inspect.
 * \return Maximum absolute value over explicitly stored entries.
 */
template <size_t dimension, execution_space space, typename data_t>
real_type_t<data_t>
norm_inf(SparseTensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  if (tensor.number_nonzeros() == 0)
  {
    return real_type_t<data_t>{};
  }
  return reductions::max_abs_reduce<space>(tensor.values_tensor().const_view().data(), tensor.number_nonzeros());
}

/**
 * \brief Returns a sparse tensor with tiny stored values removed.
 *
 * An entry is dropped when `is_tiny(abs(value) / tolerance)` is true. Passing a
 * non-positive tolerance removes only exact structural zeros.
 *
 * \param tensor Sparse tensor to filter.
 * \param tolerance Scale used for relative tiny checks.
 * \return A sparse tensor with the same extents and only retained entries.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> filter_small_entries(
  SparseTensor<dimension, space, data_t> const& tensor,
  real_type_t<data_t> tolerance)
{
  if constexpr (space == host_space)
  {
    using sparse_tensor_t = SparseTensor<dimension, host_space, data_t>;

    auto tensor_view = tensor.const_view();
    auto values = tensor.values_tensor().const_view();
    index_t number_retained = 0;
    for (index_t entry = 0; entry < tensor.number_nonzeros(); ++entry)
    {
      if (!::boba::detail::sparse_value_is_tiny(values(entry), tolerance))
      {
        ++number_retained;
      }
    }

    typename sparse_tensor_t::index_list_array_t index_lists;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      index_lists[d].resize({number_retained});
    }
    typename sparse_tensor_t::values_list_t filtered_values({number_retained});

    auto filtered_values_view = filtered_values.view();
    index_t filtered_entry = 0;
    for (index_t entry = 0; entry < tensor.number_nonzeros(); ++entry)
    {
      if (!::boba::detail::sparse_value_is_tiny(values(entry), tolerance))
      {
        const auto indices = tensor_view.entry_multiindex(entry);
        for (std::size_t d = 0; d < dimension; ++d)
        {
          index_lists[d].view()(filtered_entry) = indices[d];
        }
        filtered_values_view(filtered_entry) = values(entry);
        ++filtered_entry;
      }
    }

    return sparse_tensor_t(tensor.sizes(), std::move(index_lists), std::move(filtered_values));
  }
  else
  {
    SparseTensor<dimension, host_space, data_t> host_tensor(tensor);
    auto host_filtered = filter_small_entries(host_tensor, tolerance);
    return SparseTensor<dimension, space, data_t>(host_filtered);
  }
}

/**
 * \brief Returns a sparse tensor with entries tiny relative to its inf norm removed.
 *
 * \param tensor Sparse tensor to filter.
 * \return A sparse tensor with values tiny relative to `norm_inf(tensor)` removed.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> filter_small_entries(
  SparseTensor<dimension, space, data_t> const& tensor)
{
  return filter_small_entries(tensor, norm_inf(tensor));
}

/**
 * \brief Permutes a sparse tensor in place.
 *
 * The permutation is applied to the sparse index lists and the output extents.
 * Missing entries remain implicit zeros.
 *
 * For non-host execution spaces, this routine copies the sparse tensor to host,
 * performs the index-list mutation there, and copies the result back.
 *
 * \param sparse Sparse tensor to permute.
 * \param permutations Output-to-input dimension map. For example, `{1, 0}`
 * transposes a rank-2 sparse tensor.
 */
template <execution_space space, size_t dimension, typename data_t, bool force_naive = false>
void permute(
  SparseTensor<dimension, space, data_t>& sparse,
  const Array<index_t, dimension>& permutations)
{
  ::boba::detail::ignore(force_naive);
  boba_always_assert(is_valid_permutation(permutations), "Invalid permutation.");

  if constexpr (space == host_space)
  {
    SparseTensor<dimension, host_space, data_t> output(::boba::permute(sparse.sizes(), permutations));
    auto sparse_view = sparse.const_view();
    auto values = sparse.values_tensor().const_view();
    for (index_t entry = 0; entry < sparse.number_nonzeros(); ++entry)
    {
      const auto indices = sparse_view.entry_multiindex(entry);
      output.set(::boba::permute(indices, permutations), values(entry));
    }
    sparse = std::move(output);
  }
  else
  {
    SparseTensor<dimension, host_space, data_t> host_sparse(sparse);
    permute<host_space, dimension, data_t, force_naive>(host_sparse, permutations);
    sparse = SparseTensor<dimension, space, data_t>(host_sparse);
  }
}

/**
 * \brief Permutes a sparse tensor in place using einsum-style labels.
 *
 * \param original_indices Labels describing the current sparse tensor modes.
 * \param sparse Sparse tensor to permute.
 * \param permuted_indices Labels describing the desired output mode order.
 */
template <execution_space space, size_t dimension, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
void permute(
  const Array<DimLabel_t, dimension>& original_indices,
  SparseTensor<dimension, space, data_t>& sparse,
  const Array<DimLabel_t, dimension>& permuted_indices)
{
  Array<index_t, dimension> permutations;
  for (size_t d = 0; d < dimension; d++)
  {
    auto this_index = permuted_indices[d];
    index_t id = 0;
    for (size_t di = 0; di < dimension; di++)
    {
      if (this_index == original_indices[di])
      {
        id = di;
      }
    }
    permutations[d] = id;
  }
  permute<space, dimension, data_t, force_naive>(sparse, permutations);
}

/**
 * \brief Reduces a sparse tensor over one or more dimensions.
 *
 * The output stores sums of explicit nonzero entries whose unremoved indices
 * coincide. Missing input entries contribute zero and are never visited.
 *
 * For non-host execution spaces, this routine copies the sparse tensor to host,
 * constructs the reduced sparse tensor there, and copies the result back.
 *
 * \param tensor_A Sparse tensor to reduce.
 * \param contraction_dimensions Dimensions of \p tensor_A to sum over.
 * \return Sparse tensor over the remaining dimensions in their original order.
 */
template <size_t reductions, execution_space space, size_t dimension_A, typename data_t, bool force_naive = false>
SparseTensor<dimension_A - reductions, space, data_t> tensor_reduction(
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  ::boba::detail::ignore(force_naive);
  constexpr size_t new_dimension = dimension_A - reductions;
  static_assert(new_dimension > 0_z, "Invalid choice of tensors and/or reduction");

  if constexpr (space == host_space)
  {
    using real_data_t = real_type_t<data_t>;
    const real_data_t accumulation_tolerance = norm_inf(tensor_A);
    auto contracted_A_mider = make_contracted_dimensions<reductions>(tensor_A.sizes(), contraction_dimensions);
    SparseTensor<new_dimension, host_space, data_t> tensor_C(contracted_A_mider.sizes());
    auto tensor_A_view = tensor_A.const_view();
    auto values = tensor_A.values_tensor().const_view();
    for (index_t entry = 0; entry < tensor_A.number_nonzeros(); ++entry)
    {
      const auto indices_A = tensor_A_view.entry_multiindex(entry);
      const auto indices_C = ::boba::delete_elements(indices_A, contraction_dimensions);
      ::boba::detail::sparse_accumulate(tensor_C, indices_C, values(entry), accumulation_tolerance);
    }
    return filter_small_entries(tensor_C);
  }
  else
  {
    SparseTensor<dimension_A, host_space, data_t> host_tensor_A(tensor_A);
    auto host_tensor_C = tensor_reduction<reductions, host_space, dimension_A, data_t, force_naive>(
      host_tensor_A,
      contraction_dimensions);
    return SparseTensor<new_dimension, space, data_t>(host_tensor_C);
  }
}

/**
 * \brief Reduces a sparse tensor using einsum-style labels.
 *
 * Dimensions present in \p labels but absent from \p final_labels are summed
 * over. The intermediate sparse reduction preserves the original remaining
 * order, then the result is permuted to match \p final_labels.
 *
 * \param labels Labels describing the input sparse tensor modes.
 * \param tensor_A Sparse tensor to reduce.
 * \param final_labels Labels describing the desired output mode order.
 * \return Sparse tensor over \p final_labels.
 */
template <size_t reductions, execution_space space, size_t dimension_A, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
SparseTensor<dimension_A - reductions, space, data_t> tensor_reduction(
  const Array<DimLabel_t, dimension_A>& labels,
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_A - reductions>& final_labels)
{
  auto missing_labels = get_missing_labels(labels, final_labels);
  auto reduced_labels = get_missing_labels(labels, missing_labels);
  auto missing_indices = get_label_indices(labels, missing_labels);
  auto output = tensor_reduction<reductions, space, dimension_A, data_t, force_naive>(
    tensor_A,
    missing_indices);
  permute(reduced_labels, output, final_labels);
  return output;
}

/**
 * \brief Contracts two sparse tensors along matching dimensions.
 *
 * Only explicitly stored entries are paired. A pair contributes when every
 * contracted index value matches, and its product is accumulated into the
 * output at the uncontracted indices. The output dimension order matches the
 * dense contraction overload: uncontracted A dimensions followed by
 * uncontracted B dimensions.
 *
 * For non-host execution spaces, this routine copies both inputs to host,
 * constructs the contracted sparse tensor there, and copies the result back.
 *
 * \param tensor_A Left sparse tensor.
 * \param tensor_B Right sparse tensor.
 * \param contraction_dimensions_A Contracted dimensions in \p tensor_A.
 * \param contraction_dimensions_B Contracted dimensions in \p tensor_B.
 * \return Sparse tensor containing the contracted result.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
  requires(((dimension_A + dimension_B) - 2 * contractions) > 0)
SparseTensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const SparseTensor<dimension_B, space, data_t>& tensor_B,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(force_naive);
  constexpr size_t new_dimension = (dimension_A + dimension_B) - 2 * contractions;

  if constexpr (space == host_space)
  {
    using real_data_t = real_type_t<data_t>;
    const real_data_t accumulation_tolerance = norm_inf(tensor_A) * norm_inf(tensor_B);
    const auto sizes_A = tensor_A.sizes();
    const auto sizes_B = tensor_B.sizes();
    for (size_t c = 0; c < contractions; c++)
    {
      boba_always_assert_equal(sizes_A[contraction_dimensions_A[c]], sizes_B[contraction_dimensions_B[c]], "Invalid contraction.");
    }

    auto contracted_A_mider = make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
    auto contracted_B_mider = make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);
    SparseTensor<new_dimension, host_space, data_t> tensor_C(concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes()));

    auto values_A = tensor_A.values_tensor().const_view();
    auto values_B = tensor_B.values_tensor().const_view();
    auto tensor_A_view = tensor_A.const_view();
    auto tensor_B_view = tensor_B.const_view();
    for (index_t entry_A = 0; entry_A < tensor_A.number_nonzeros(); ++entry_A)
    {
      const auto indices_A = tensor_A_view.entry_multiindex(entry_A);
      for (index_t entry_B = 0; entry_B < tensor_B.number_nonzeros(); ++entry_B)
      {
        const auto indices_B = tensor_B_view.entry_multiindex(entry_B);
        if (::boba::detail::sparse_contraction_indices_match(indices_A, indices_B, contraction_dimensions_A, contraction_dimensions_B))
        {
          const auto indices_C = ::boba::detail::sparse_contraction_output_indices(
            indices_A,
            indices_B,
            contraction_dimensions_A,
            contraction_dimensions_B);
          ::boba::detail::sparse_accumulate(tensor_C, indices_C, values_A(entry_A) * values_B(entry_B), accumulation_tolerance);
        }
      }
    }
    return filter_small_entries(tensor_C);
  }
  else
  {
    SparseTensor<dimension_A, host_space, data_t> host_tensor_A(tensor_A);
    SparseTensor<dimension_B, host_space, data_t> host_tensor_B(tensor_B);
    auto host_tensor_C = tensor_contraction<contractions, host_space, dimension_A, dimension_B, data_t, force_naive>(
      host_tensor_A,
      host_tensor_B,
      contraction_dimensions_A,
      contraction_dimensions_B);
    return SparseTensor<new_dimension, space, data_t>(host_tensor_C);
  }
}

/**
 * \brief Contracts two sparse tensors using einsum-style labels.
 *
 * Common labels between \p labels_A and \p labels_B identify the contracted
 * dimensions. The contracted result is then permuted to match \p labels_C.
 *
 * \param labels_A Labels describing the left sparse tensor modes.
 * \param tensor_A Left sparse tensor.
 * \param labels_B Labels describing the right sparse tensor modes.
 * \param tensor_B Right sparse tensor.
 * \param labels_C Labels describing the desired output mode order.
 * \return Sparse tensor over \p labels_C.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
  requires(((dimension_A + dimension_B) - 2 * contractions) > 0)
SparseTensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const Array<DimLabel_t, dimension_A>& labels_A,
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_B>& labels_B,
  const SparseTensor<dimension_B, space, data_t>& tensor_B,
  const Array<DimLabel_t, (dimension_A + dimension_B) - 2 * contractions>& labels_C)
{
  auto common_labels = get_common_labels<contractions>(labels_A, labels_B);
  auto contraction_indices_A = get_label_indices(labels_A, common_labels);
  auto contraction_indices_B = get_label_indices(labels_B, common_labels);
  auto uncontracted_labels = get_uncontracted_labels(labels_A, contraction_indices_A, labels_B, contraction_indices_B);
  auto output = tensor_contraction<contractions, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    contraction_indices_A,
    contraction_indices_B);
  permute(uncontracted_labels, output, labels_C);
  return output;
}

template <size_t dimension, execution_space space, typename data_t>
real_type_t<data_t>
norm_l1(SparseTensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  using real_data_t = real_type_t<data_t>;

  auto values = tensor.values_tensor().const_view().data();

  real_data_t value = 0.0;

  ::boba::sum_reduce<space>(value, index_t(0), tensor.number_nonzeros(), [=] __boba_host_device__(index_t i, sum_reducer_operator<real_data_t> & local_value)
  {
    local_value += boba::abs(values[i]);
  });

  return value;
}

} // namespace boba
