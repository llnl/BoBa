// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

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
__boba_host_device__ bool sparse_value_is_tiny(data_t value, real_type_t<data_t> tolerance)
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
 * \brief Naively contracts a host sparse tensor with a host dense tensor.
 *
 *
 */
template <bool sparse_first, size_t dimension_dense, size_t dimension_sparse, size_t dimension_out, typename accesor_dense_t, typename accesor_sparse_t, typename accesor_out_t>
void sparse_dense_contraction_core_naive(
  const TensorView<accesor_dense_t, dimension_dense>& dense_view,
  const SparseTensorView<accesor_sparse_t, dimension_sparse>& sparse_view,
  const auto& contracted_dense_mider,
  auto sparse_contracted_dimension,
  auto dense_contracted_dimension,
  const TensorView<accesor_out_t, dimension_out>& out_view)
{
  const auto sparse_values = sparse_view.values_ptr();
  // A rank-0 multiindexer reports size 0, but a fully contracted dense operand still has one free index.
  constexpr bool dense_fully_contracted = dimension_dense == std::tuple_size_v<decltype(dense_contracted_dimension)>;
  const index_t dense_free_size = dense_fully_contracted ? index_t(1) : contracted_dense_mider.size();
  ::boba::loop<host_space, 2>(
    {dense_free_size, sparse_view.nnz()},
    [=](Array<index_t, 2> indices)
  {
    const auto [cid_dense, id_sparse] = indices;
    const auto mid_sparse = sparse_view.entry_multiindex(id_sparse);
    auto contraction_ids = sparse_contracted_dimension;
    for (size_t c = 0; c < contraction_ids.size(); ++c)
    {
      contraction_ids[c] = mid_sparse[sparse_contracted_dimension[c]];
    }
    const auto cmid_sparse = delete_elements(mid_sparse, sparse_contracted_dimension);
    const auto cmid_dense = contracted_dense_mider.multiindex(cid_dense);

    //
    // Deduce dense contraction indices
    //
    const auto mid_dense = make_uncontracted_mid(cmid_dense, contraction_ids, dense_contracted_dimension);

    //
    // Deduce output contraction indices
    //
    const auto cmid_out = [=]()
    {
      if constexpr (sparse_first)
      {
        return concatenate(cmid_sparse, cmid_dense);
      }
      else
      {
        return concatenate(cmid_dense, cmid_sparse);
      }
    }();

    //
    // Contract
    //
    const auto value_dense = dense_view(mid_dense);
    const auto value_sparse = sparse_values[id_sparse];
    out_view(cmid_out) += value_dense * value_sparse;
  });
}

/**
 * \brief Naively contracts a host sparse tensor with a host dense tensor.
 *
 * The sparse operand is traversed only through its explicitly stored entries.
 * The dense operand is scanned to find entries matching each sparse entry's
 * contracted indices.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, typename data_t>
Tensor<(dimension_A + dimension_B) - 2 * contractions, host_space, data_t>
sparse_dense_contraction_naive(
  const SparseTensor<dimension_A, host_space, data_t>& tensor_A,
  const Tensor<dimension_B, host_space, data_t>& tensor_B,
  const Array<size_t, contractions> contraction_dimensions_A,
  const Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  constexpr size_t new_dimension = (dimension_A + dimension_B) - 2 * contractions;
  const auto sizes_A = tensor_A.sizes();
  const auto sizes_B = tensor_B.sizes();
  for (size_t c = 0; c < contractions; ++c)
  {
    boba_always_assert_equal(
      sizes_A[contraction_dimensions_A[c]],
      sizes_B[contraction_dimensions_B[c]],
      "Invalid contraction.");
  }

  const auto contracted_A_mider = make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
  const auto contracted_B_mider = make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);
  Tensor<new_dimension, host_space, data_t> tensor_C(
    concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes()));
  tensor_C.fill_with_zeros();

  const auto sparse_view = tensor_A.const_view();
  const auto dense_view = tensor_B.const_view();

  auto contracted_dense_mider = contracted_B_mider;
  auto dense_contracted_dimension = contraction_dimensions_B;
  auto sparse_contracted_dimension = contraction_dimensions_A;

  sparse_dense_contraction_core_naive<true, dimension_B, dimension_A, new_dimension>(
    dense_view,
    sparse_view,
    contracted_dense_mider,
    sparse_contracted_dimension,
    dense_contracted_dimension,
    tensor_C.view());

  return tensor_C;
}

/**
 * \brief Naively contracts a host dense tensor with a host sparse tensor.
 *
 * The sparse operand is traversed only through its explicitly stored entries.
 * The dense operand is scanned to find entries matching each sparse entry's
 * contracted indices.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, typename data_t>
Tensor<(dimension_A + dimension_B) - 2 * contractions, host_space, data_t>
dense_sparse_contraction_naive(
  const Tensor<dimension_A, host_space, data_t>& tensor_A,
  const SparseTensor<dimension_B, host_space, data_t>& tensor_B,
  const Array<size_t, contractions> contraction_dimensions_A,
  const Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  constexpr size_t new_dimension = (dimension_A + dimension_B) - 2 * contractions;
  const auto sizes_A = tensor_A.sizes();
  const auto sizes_B = tensor_B.sizes();
  for (size_t c = 0; c < contractions; ++c)
  {
    boba_always_assert_equal(
      sizes_A[contraction_dimensions_A[c]],
      sizes_B[contraction_dimensions_B[c]],
      "Invalid contraction.");
  }

  const auto contracted_A_mider = make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
  const auto contracted_B_mider = make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);
  Tensor<new_dimension, host_space, data_t> tensor_C(
    concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes()));
  tensor_C.fill_with_zeros();

  const auto sparse_view = tensor_B.const_view();
  const auto dense_view = tensor_A.const_view();

  auto contracted_dense_mider = contracted_A_mider;
  auto dense_contracted_dimension = contraction_dimensions_A;
  auto sparse_contracted_dimension = contraction_dimensions_B;

  sparse_dense_contraction_core_naive<false, dimension_A, dimension_B, new_dimension>(
    dense_view,
    sparse_view,
    contracted_dense_mider,
    sparse_contracted_dimension,
    dense_contracted_dimension,
    tensor_C.view());

  return tensor_C;
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

/**
 * \brief Returns the modes not listed in \p contracted_dims, in increasing order.
 */
template <size_t dimension, size_t contractions>
std::vector<size_t> free_dims(const Array<size_t, contractions>& contracted_dims)
{
  std::vector<size_t> output;
  for (size_t d = 0; d < dimension; ++d)
  {
    if (std::find(contracted_dims.begin(), contracted_dims.end(), d) == contracted_dims.end())
    {
      output.push_back(d);
    }
  }
  return output;
}

/**
 * \brief Builds the CSR form of the transpose of a column-major-sorted sparse matrix.
 *
 * \p unfolded stores strictly increasing keys `key = r + rows * c`, so sorting by
 * key sorts by column `c` first and row `r` second. Read as the transpose, this
 * is CSR order: CSR row `c`, CSR column `r`. No sort is needed.
 *
 * Example with a 2 x 3 matrix with keys {1, 2, 5} = {(1,0), (0,1), (1,2)}:
 * the CSR of the 3 x 2 transpose has row_offsets {0, 1, 2, 3} and columns {1, 0, 1}.
 *
 * \param unfolded Rank-2 sparse tensor with sorted unique linear keys.
 * \return CSR matrix of shape `(unfolded.sizes(1), unfolded.sizes(0))`.
 */
template <execution_space space, typename data_t>
CSRMatrix<space, data_t> transposed_csr_from_sorted_keys(const SparseTensor<2, space, data_t>& unfolded)
{
  BOBA_CALI_MARK
  const auto sizes = unfolded.sizes();
  const index_t csr_rows = sizes[1];
  const index_t csr_cols = sizes[0];
  const index_t nnz = unfolded.nnz();
  boba_always_assert(unfolded.size() <= static_cast<index_t>(std::numeric_limits<std::int64_t>::max()),
                     "Unfolded sparse tensor is too large for int64 CSR indices");

  CSRMatrix<space, data_t> csr(csr_rows, csr_cols);
  csr.m_values = unfolded.values();
  csr.m_column_indices = Vector<space, std::int64_t>({nnz});

  const auto& keys = unfolded.linear_indices();
  const auto keys_view = keys.const_view();
  index_t const* key_data = keys_view.data();
  auto column_view = csr.m_column_indices.view();
  auto offsets_view = csr.m_row_offsets.view();

  ::boba::loop<space, 1>(0_z, nnz, [=] __boba_host_device__(index_t entry)
  {
    column_view(entry) = static_cast<std::int64_t>(key_data[entry] % csr_cols);
  });

  // offsets(i) is the first entry whose key is at least i * csr_cols.
  ::boba::loop<space, 1>(0_z, csr_rows + 1, [=] __boba_host_device__(index_t row)
  {
    const index_t target = row * csr_cols;
    index_t low = 0;
    index_t high = nnz;
    while (low < high)
    {
      const index_t middle = low + (high - low) / 2;
      if (key_data[middle] < target)
      {
        low = middle + 1;
      }
      else
      {
        high = middle;
      }
    }
    offsets_view(row) = static_cast<std::int64_t>(low);
  });

  boba_assert(csr.is_sorted(), "CSR built from sparse tensor keys must have sorted rows");
  return csr;
}

/**
 * \brief Contracts a sparse tensor S with a dense tensor D by unfolding both and calling SpMV or SpMM.
 *
 * S is unfolded as (contracted x free) so its keys give the CSR of the
 * (free x contracted) matrix without sorting. D is unfolded as
 * (contracted x free). The result is a dense tensor over
 * `[free modes of S..., free modes of D...]`, each group in increasing order.
 * SpMV is used when D has a single free column.
 *
 * \tparam force_naive Forwarded to the CSR kernels to skip vendor libraries.
 * \note Contracted modes are paired in the caller's order, so the explicit
 *       `(row_dims, col_dims)` unfold overloads are used.
 */
template <size_t contractions, execution_space space, size_t dimension_S, size_t dimension_D, typename data_t, bool force_naive = false>
Tensor<(dimension_S + dimension_D) - 2 * contractions, space, data_t> sparse_dense_contraction_unfolded(
  const SparseTensor<dimension_S, space, data_t>& sparse,
  const Tensor<dimension_D, space, data_t>& dense,
  const Array<size_t, contractions> contraction_dimensions_S,
  const Array<size_t, contractions> contraction_dimensions_D)
{
  BOBA_CALI_MARK
  constexpr size_t new_dimension = (dimension_S + dimension_D) - 2 * contractions;
  const auto sizes_S = sparse.sizes();
  const auto sizes_D = dense.sizes();
  for (size_t c = 0; c < contractions; ++c)
  {
    boba_always_assert_equal(
      sizes_S[contraction_dimensions_S[c]],
      sizes_D[contraction_dimensions_D[c]],
      "Invalid contraction.");
  }

  const auto free_S_mider = make_contracted_dimensions<contractions>(sizes_S, contraction_dimensions_S);
  const auto free_D_mider = make_contracted_dimensions<contractions>(sizes_D, contraction_dimensions_D);
  Tensor<new_dimension, space, data_t> output(concatenate(free_S_mider.sizes(), free_D_mider.sizes()));
  if (sparse.nnz() == 0)
  {
    output.fill_with_zeros();
    return output;
  }

  const auto sparse_matrix = unfold(sparse,
                                    make_std_vector(contraction_dimensions_S),
                                    free_dims<dimension_S>(contraction_dimensions_S));
  const auto csr = transposed_csr_from_sorted_keys(sparse_matrix);

  // TODO: skip the permuted copy when the contracted modes are already leading.
  const auto dense_matrix = unfold(dense,
                                   make_std_vector(contraction_dimensions_D),
                                   free_dims<dimension_D>(contraction_dimensions_D));

  if (dense_matrix.cols() == 1)
  {
    Vector<space, data_t> dense_vector({dense_matrix.rows()});
    dense_vector.reshape(dense_matrix);
    Vector<space, data_t> product({csr.rows()});
    sparse_matrix_vector_product<space, data_t, force_naive>(csr, dense_vector, product);
    output.reshape(product);
    return output;
  }

  Matrix<space, data_t> product({csr.rows(), dense_matrix.cols()});
  sparse_matrix_matrix_product<space, data_t, force_naive>(csr, dense_matrix, product);
  output.reshape(product);
  return output;
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
  if (tensor.nnz() == 0)
  {
    return real_type_t<data_t>{};
  }
  return norm_inf(tensor.values());
}

/**
 * \brief Returns a sparse tensor with tiny stored values removed.
 *
 * An entry is dropped when `is_tiny(abs(value) / tolerance)` is true. Passing a
 * non-positive tolerance removes only exact structural zeros.
 *
 * CUDA/HIP tensors with CUB/hipCUB are filtered on device (flag, prefix scan,
 * scatter); other non-host configurations copy to host and back.
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
  return ::boba::detail::copy_entries_if(tensor, [=] __boba_host_device__(data_t value)
  {
    return !::boba::detail::sparse_value_is_tiny(value, tolerance);
  });
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
 * Each entry's linear index is remapped to the permuted extents, then entries
 * are re-sorted. Missing entries remain implicit zeros.
 *
 * Keys are remapped in the tensor's execution space and re-sorted by
 * detail::sort_and_coalesce(), which stays on device with CUB/hipCUB.
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

  const Multiindexer<dimension> permuted_indexer(::boba::permute(sparse.sizes(), permutations));
  auto sparse_view = sparse.const_view();
  Vector<space, index_t> permuted_keys({sparse.nnz()});
  auto permuted_keys_view = permuted_keys.view();
  ::boba::loop<space, 1>(0_z, sparse.nnz(), [=] __boba_host_device__(index_t entry)
  {
    permuted_keys_view(entry) = permuted_indexer.index(::boba::permute(sparse_view.entry_multiindex(entry), permutations));
  });
  sparse = SparseTensor<dimension, space, data_t>(permuted_indexer.sizes(), std::move(permuted_keys), sparse.values(), sparse.name());
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

// ---------------------------------------------------------------------------
// Unfolding and folding operations
//
// Entries are keyed by column-major linear index, so after permuting the row
// modes to the front, the N-D keys are already the keys of the unfolded matrix.
// The reshape only changes sizes; no keys are recomputed.
// ---------------------------------------------------------------------------

namespace detail
{

/**
 * \brief Concatenates \p row_dims and \p col_dims into a validated mode ordering.
 */
template <size_t dimension>
Array<index_t, dimension> unfolding_ordering(std::vector<size_t> const& row_dims, std::vector<size_t> const& col_dims)
{
  boba_always_assert(row_dims.size() + col_dims.size() == dimension,
                     "Inconsistent tensor and matrix dimensions.");
  Array<index_t, dimension> ordering;
  size_t idx = 0;
  for (size_t i : row_dims)
  {
    ordering[idx++] = i;
  }
  for (size_t i : col_dims)
  {
    ordering[idx++] = i;
  }
  boba_always_assert(is_valid_permutation(ordering), "Permutation violates one or more conditions.");
  return ordering;
}

/**
 * \brief Returns the modes not in \p row_dims, in increasing order. Sorts \p row_dims.
 */
template <size_t dimension>
std::vector<size_t> complement_dims(std::vector<size_t>& row_dims)
{
  std::vector<size_t> all_dims(dimension);
  std::iota(all_dims.begin(), all_dims.end(), 0_z);
  std::sort(row_dims.begin(), row_dims.end());
  std::vector<size_t> col_dims;
  std::set_difference(all_dims.begin(), all_dims.end(), row_dims.begin(), row_dims.end(), std::back_inserter(col_dims));
  return col_dims;
}

} // namespace detail

/**
 * \brief Unfolds a sparse tensor into a sparse matrix using explicit row and column groups.
 *
 * A({i, j, k, l, p}) -> A({row_dims}, {col_dims})
 *
 * Matches unfold() for dense tensors: within each group the first mode varies
 * fastest. Costs one sparse permute(); the reshape reuses the permuted keys.
 *
 * \param input Sparse tensor to unfold.
 * \param row_dims Tensor modes to group on the matrix rows.
 * \param col_dims Tensor modes to group on the matrix columns.
 * \return Rank-2 sparse tensor of shape `(prod(row sizes), prod(col sizes))`.
 *
 * \note The order of \p row_dims and \p col_dims is preserved. Together they
 *       must form a valid permutation of the tensor modes.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<2, space, data_t> unfold(
  const SparseTensor<dimension, space, data_t>& input,
  std::vector<size_t> row_dims,
  std::vector<size_t> col_dims)
{
  const auto ordering = ::boba::detail::unfolding_ordering<dimension>(row_dims, col_dims);
  const auto input_sizes = input.sizes();

  index_t rows = 1;
  for (size_t d : row_dims)
  {
    rows *= input_sizes[d];
  }
  index_t cols = 1;
  for (size_t d : col_dims)
  {
    cols *= input_sizes[d];
  }

  auto permuted = input;
  permute(permuted, ordering);
  return SparseTensor<2, space, data_t>({rows, cols}, permuted.linear_indices(), permuted.values(), input.name());
}

/**
 * \brief Unfolds a sparse tensor into a sparse matrix whose rows correspond to a grouping of modes.
 *
 * \note The entries of \p row_dims are sorted internally. All remaining
 *       modes are placed on the matrix columns in increasing order.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<2, space, data_t> unfold(
  const SparseTensor<dimension, space, data_t>& input,
  std::vector<size_t> row_dims)
{
  auto col_dims = ::boba::detail::complement_dims<dimension>(row_dims);
  return unfold(input, row_dims, col_dims);
}

/**
 * \brief Unfolds a sparse tensor into a sparse matrix whose rows correspond to one mode.
 *
 * A({i, j, k, l, p}) -> A({k}, {i, j, l, p})
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<2, space, data_t> unfold(
  const SparseTensor<dimension, space, data_t>& input,
  const size_t mode_k)
{
  return unfold(input, std::vector<size_t>{mode_k});
}

/**
 * \brief Folds a sparse matrix into a sparse tensor using explicit row and column groups.
 *
 * A({row_dims}, {col_dims}) -> A({i, j, k, l, p})
 *
 * Inverse of unfold(input, row_dims, col_dims): the matrix keys are
 * reinterpreted as the permuted tensor, which is then permuted back.
 *
 * \param input Rank-2 sparse tensor to fold.
 * \param sizes Sizes of the output tensor modes.
 * \param row_dims Original tensor modes grouped into the matrix rows.
 * \param col_dims Original tensor modes grouped into the matrix columns.
 * \return Sparse tensor with extents \p sizes.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> fold(
  const SparseTensor<2, space, data_t>& input,
  const Array<index_t, dimension>& sizes,
  std::vector<size_t> row_dims,
  std::vector<size_t> col_dims)
{
  boba_always_assert_equal(::boba::product(sizes), input.size(), "Inconsistent tensor and matrix sizes.");
  const auto ordering = ::boba::detail::unfolding_ordering<dimension>(row_dims, col_dims);

  Array<index_t, dimension> inverse_ordering;
  for (size_t d = 0; d < dimension; ++d)
  {
    inverse_ordering[ordering[d]] = d;
  }

  SparseTensor<dimension, space, data_t> output(::boba::permute(sizes, ordering), input.linear_indices(), input.values(), input.name());
  permute(output, inverse_ordering);
  return output;
}

/**
 * \brief Folds a sparse matrix into a sparse tensor, where the matrix rows represent a group of modes.
 *
 * \note Inverse of unfold(input, row_dims). The entries of \p row_dims are
 *       sorted internally; remaining modes are the matrix columns.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> fold(
  const SparseTensor<2, space, data_t>& input,
  const Array<index_t, dimension>& sizes,
  std::vector<size_t> row_dims)
{
  auto col_dims = ::boba::detail::complement_dims<dimension>(row_dims);
  return fold(input, sizes, row_dims, col_dims);
}

/**
 * \brief Folds a sparse matrix into a sparse tensor, where the matrix rows represent one mode.
 *
 * \note Inverse of unfold(input, mode_k).
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> fold(
  const SparseTensor<2, space, data_t>& input,
  const Array<index_t, dimension>& sizes,
  const size_t mode_k)
{
  return fold(input, sizes, std::vector<size_t>{mode_k});
}

/**
 * \brief Reduces a sparse tensor over one or more dimensions.
 *
 * The output stores sums of explicit nonzero entries whose unremoved indices
 * coincide. Missing input entries contribute zero and are never visited.
 * Sums that are tiny relative to `norm_inf(tensor_A)` are dropped.
 *
 * Keys are remapped in the tensor's execution space and re-sorted by
 * detail::sort_and_coalesce(), which stays on device with CUB/hipCUB.
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

  const auto accumulation_tolerance = norm_inf(tensor_A);
  const auto reduced_indexer = make_contracted_dimensions<reductions>(tensor_A.sizes(), contraction_dimensions);
  auto tensor_A_view = tensor_A.const_view();
  Vector<space, index_t> reduced_keys({tensor_A.nnz()});
  auto reduced_keys_view = reduced_keys.view();
  ::boba::loop<space, 1>(0_z, tensor_A.nnz(), [=] __boba_host_device__(index_t entry)
  {
    reduced_keys_view(entry) = reduced_indexer.index(::boba::delete_elements(tensor_A_view.entry_multiindex(entry), contraction_dimensions));
  });
  SparseTensor<new_dimension, space, data_t> tensor_C(reduced_indexer.sizes(), std::move(reduced_keys), tensor_A.values());
  auto cancelled_removed = filter_small_entries(tensor_C, accumulation_tolerance);
  return filter_small_entries(cancelled_removed);
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
 * \brief Contracts a sparse tensor with a dense tensor.
 *
 * The contraction follows the same dimension ordering as the dense
 * `tensor_contraction` API. The sparse operand is unfolded to a CSR matrix
 * and the product uses the backend SpMV or SpMM kernel. On host,
 * `force_naive` selects the entry traversal kernel; on other spaces it
 * selects the naive CSR kernels.
 *
 * \param tensor_A Sparse left tensor.
 * \param tensor_B Dense right tensor.
 * \param contraction_dimensions_A Contracted dimensions of \p tensor_A.
 * \param contraction_dimensions_B Contracted dimensions of \p tensor_B.
 * \return Dense tensor containing the contracted result.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const Tensor<dimension_B, space, data_t>& tensor_B,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  if constexpr (force_naive and space == host_space)
  {
    return ::boba::detail::sparse_dense_contraction_naive(
      tensor_A,
      tensor_B,
      contraction_dimensions_A,
      contraction_dimensions_B);
  }
  else
  {
    return ::boba::detail::sparse_dense_contraction_unfolded<contractions, space, dimension_A, dimension_B, data_t, force_naive>(
      tensor_A,
      tensor_B,
      contraction_dimensions_A,
      contraction_dimensions_B);
  }
}

/**
 * \brief Contracts a dense tensor with a sparse tensor.
 *
 * The contraction follows the same dimension ordering as the dense
 * `tensor_contraction` API. The sparse operand is unfolded to a CSR matrix
 * and the product uses the backend SpMV or SpMM kernel. On host,
 * `force_naive` selects the entry traversal kernel; on other spaces it
 * selects the naive CSR kernels.
 *
 * \param tensor_A Dense left tensor.
 * \param tensor_B Sparse right tensor.
 * \param contraction_dimensions_A Contracted dimensions of \p tensor_A.
 * \param contraction_dimensions_B Contracted dimensions of \p tensor_B.
 * \return Dense tensor containing the contracted result.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const SparseTensor<dimension_B, space, data_t>& tensor_B,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  if constexpr (force_naive and space == host_space)
  {
    return ::boba::detail::dense_sparse_contraction_naive(
      tensor_A,
      tensor_B,
      contraction_dimensions_A,
      contraction_dimensions_B);
  }
  else
  {
    // C^T = B^T * A^T gives [free B..., free A...]; permute back to [free A..., free B...].
    constexpr size_t free_A = dimension_A - contractions;
    constexpr size_t free_B = dimension_B - contractions;
    auto output = ::boba::detail::sparse_dense_contraction_unfolded<contractions, space, dimension_B, dimension_A, data_t, force_naive>(
      tensor_B,
      tensor_A,
      contraction_dimensions_B,
      contraction_dimensions_A);
    Array<index_t, free_A + free_B> ordering;
    for (size_t d = 0; d < free_A; ++d)
    {
      ordering[d] = static_cast<index_t>(free_B + d);
    }
    for (size_t d = 0; d < free_B; ++d)
    {
      ordering[free_A + d] = static_cast<index_t>(d);
    }
    permute(output, ordering);
    return output;
  }
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
  BOBA_CALI_MARK
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

    auto values_A = tensor_A.values().const_view();
    auto values_B = tensor_B.values().const_view();
    auto tensor_A_view = tensor_A.const_view();
    auto tensor_B_view = tensor_B.const_view();
    for (index_t entry_A = 0; entry_A < tensor_A.nnz(); ++entry_A)
    {
      const auto indices_A = tensor_A_view.entry_multiindex(entry_A);
      for (index_t entry_B = 0; entry_B < tensor_B.nnz(); ++entry_B)
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
  BOBA_CALI_MARK
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

/**
 * \brief Contracts a sparse tensor with a dense tensor using einsum labels.
 *
 * The result is dense and follows the requested \p labels_C ordering. The
 * host implementation traverses only explicitly stored sparse entries;
 * other execution spaces are not yet implemented.
 *
 * @param labels_A Labels describing the sparse left tensor modes.
 * @param tensor_A Sparse left tensor.
 * @param labels_B Labels describing the dense right tensor modes.
 * @param tensor_B Dense right tensor.
 * @param labels_C Labels describing the desired output mode order.
 * @return Dense tensor over labels_C.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const Array<DimLabel_t, dimension_A>& labels_A,
  const SparseTensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_B>& labels_B,
  const Tensor<dimension_B, space, data_t>& tensor_B,
  const Array<DimLabel_t, (dimension_A + dimension_B) - 2 * contractions>& labels_C)
{
  BOBA_CALI_MARK
  ::boba::detail::ignore(force_naive);
  auto common_labels = get_common_labels<contractions>(labels_A, labels_B);
  auto contraction_indices_A = get_label_indices(labels_A, common_labels);
  auto contraction_indices_B = get_label_indices(labels_B, common_labels);
  auto uncontracted_labels = get_uncontracted_labels(labels_A, contraction_indices_A, labels_B, contraction_indices_B);
  auto output = tensor_contraction<contractions, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    contraction_indices_A,
    contraction_indices_B);
  if constexpr (space == host_space)
  {
    permute(uncontracted_labels, output, labels_C);
    return output;
  }
  else
  {
    ::boba::detail::ignore(labels_A, tensor_A, labels_B, tensor_B, labels_C);
    boba_error("not yet implemented");
    return {};
  }
}

/**
 * \brief Contracts a dense tensor with a sparse tensor using einsum labels.
 *
 * The result is dense and follows the requested \p labels_C ordering. The
 * host implementation traverses only explicitly stored sparse entries;
 * other execution spaces are not yet implemented.
 *
 * @param labels_A Labels describing the dense left tensor modes.
 * @param tensor_A Dense left tensor.
 * @param labels_B Labels describing the sparse right tensor modes.
 * @param tensor_B Sparse right tensor.
 * @param labels_C Labels describing the desired output mode order.
 * @return Dense tensor over labels_C.
 */
template <size_t contractions, execution_space space, size_t dimension_A, size_t dimension_B, typename data_t, bool force_naive = false, typename DimLabel_t = std::string>
  requires((dimension_A > 1) or (dimension_B > 1))
Tensor<(dimension_A + dimension_B) - 2 * contractions, space, data_t> tensor_contraction(
  const Array<DimLabel_t, dimension_A>& labels_A,
  const Tensor<dimension_A, space, data_t>& tensor_A,
  const Array<DimLabel_t, dimension_B>& labels_B,
  const SparseTensor<dimension_B, space, data_t>& tensor_B,
  const Array<DimLabel_t, (dimension_A + dimension_B) - 2 * contractions>& labels_C)
{
  BOBA_CALI_MARK
  ::boba::detail::ignore(force_naive);
  auto common_labels = get_common_labels<contractions>(labels_A, labels_B);
  auto contraction_indices_A = get_label_indices(labels_A, common_labels);
  auto contraction_indices_B = get_label_indices(labels_B, common_labels);
  auto uncontracted_labels = get_uncontracted_labels(labels_A, contraction_indices_A, labels_B, contraction_indices_B);
  auto output = tensor_contraction<contractions, space, dimension_A, dimension_B, data_t, force_naive>(
    tensor_A,
    tensor_B,
    contraction_indices_A,
    contraction_indices_B);
  if constexpr (space == host_space)
  {
    permute(uncontracted_labels, output, labels_C);
    return output;
  }
  else
  {
    ::boba::detail::ignore(labels_A, tensor_A, labels_B, tensor_B, labels_C);
    boba_error("not yet implemented");
    return {};
  }
}

/**
 * @brief Computes the sum of absolute values of explicitly stored entries.
 *
 * Missing sparse entries are implicit zeros and do not contribute. For an
 * empty sparse tensor, the result is zero.
 *
 * @param tensor Sparse tensor whose stored values are summed.
 * @return The l1 norm of the sparse tensor.
 */
template <size_t dimension, execution_space space, typename data_t>
real_type_t<data_t>
norm_l1(SparseTensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  return norm_l1(tensor.values());
}

/**
 * \brief Frobenius norm of a sparse tensor.
 *
 * Only explicitly stored entries contribute; implicit zeros add nothing.
 */
template <size_t dimension, execution_space space, typename data_t>
real_type_t<data_t>
norm_frobenius(SparseTensor<dimension, space, data_t> const& tensor)
{
  checkpoint();
  return norm_frobenius(tensor.values());
}

/**
 * \brief Checks the explicitly stored values of a sparse tensor for NaN or infinity.
 */
template <size_t dimension, execution_space space, typename data_t>
void nan_check(SparseTensor<dimension, space, data_t> const& tensor)
{
  nan_check(tensor.values());
}

/**
 * \brief In-place complex conjugate of a sparse tensor.
 */
template <size_t dimension, execution_space space, typename data_t>
void complex_conjugate_in_place(SparseTensor<dimension, space, data_t>& tensor)
{
  complex_conjugate_in_place(static_cast<Tensor<1, space, data_t>&>(tensor.values()));
}

/**
 * \return Complex conjugate of a sparse tensor (out-of-place).
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> get_conj(SparseTensor<dimension, space, data_t> const& input)
{
  auto output = input;
  complex_conjugate_in_place(output);
  return output;
}

/**
 * \brief Raises each stored value of a sparse tensor to \p _power.
 *
 * \p _power must have a positive real part so that implicit zeros stay zero.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> power(
  SparseTensor<dimension, space, data_t> const& tensor,
  const data_t _power)
{
  boba_always_assert(::boba::real(_power) > real_type_t<data_t>{}, "Sparse power requires a positive exponent");
  auto output = tensor;
  output.transform_values([=] __boba_host_device__(data_t x)
  {
    return ::boba::pow(x, _power);
  });
  return output;
}

/**
 * \brief Returns a sparse tensor with f(x) = max(x, 0) applied to each entry.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> nonnegative_part(SparseTensor<dimension, space, data_t> const& input)
{
  return input.nonnegative_part();
}

/**
 * \brief Returns a sparse tensor with f(x) = max(-x, 0) applied to each entry.
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> nonpositive_part(SparseTensor<dimension, space, data_t> const& input)
{
  return input.nonpositive_part();
}

/**
 * \brief output = scalar * tensor
 */
template <size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t> operator*(data_t const scalar, SparseTensor<dimension, space, data_t> const& tensor)
{
  return tensor * scalar;
}

} // namespace boba
