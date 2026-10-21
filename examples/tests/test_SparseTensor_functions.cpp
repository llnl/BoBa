// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

constexpr boba::execution_space space = boba::default_execution_space;

/**
 * @brief Creates deterministic extents for a test tensor.
 *
 * The size of mode d is d + 2, which gives each test rank distinct
 * mode sizes while keeping the tensors small.
 *
 * @tparam dimension Tensor rank.
 * @return Mode extents for the test tensor.
 */
template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_sizes()
{
  ::boba::Array<::boba::index_t, dimension> sizes;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    sizes[d] = static_cast<::boba::index_t>(d + 2);
  }
  return sizes;
}

/**
 * @brief Checks whether two entries agree on all contracted coordinates.
 *
 * @tparam contractions Number of contracted dimensions.
 * @tparam dimension_A Rank of the left tensor.
 * @tparam dimension_B Rank of the right tensor.
 * @param indices_A Left tensor multi-index.
 * @param indices_B Right tensor multi-index.
 * @param contraction_dimensions_A Contracted dimensions in the left tensor.
 * @param contraction_dimensions_B Contracted dimensions in the right tensor.
 * @return True when every contracted coordinate matches.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
bool contraction_indices_match(
  ::boba::Array<::boba::index_t, dimension_A> indices_A,
  ::boba::Array<::boba::index_t, dimension_B> indices_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B)
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
 * @brief Builds the uncontracted output index for a pair of entries.
 *
 * The output order is all uncontracted left-tensor dimensions followed by all
 * uncontracted right-tensor dimensions, matching the contraction API.
 *
 * @tparam contractions Number of contracted dimensions.
 * @tparam dimension_A Rank of the left tensor.
 * @tparam dimension_B Rank of the right tensor.
 * @param indices_A Left tensor multi-index.
 * @param indices_B Right tensor multi-index.
 * @param contraction_dimensions_A Contracted dimensions in the left tensor.
 * @param contraction_dimensions_B Contracted dimensions in the right tensor.
 * @return Multi-index in the contracted output tensor.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
::boba::Array<::boba::index_t, dimension_A + dimension_B - 2 * contractions> contraction_output_indices(
  ::boba::Array<::boba::index_t, dimension_A> indices_A,
  ::boba::Array<::boba::index_t, dimension_B> indices_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B)
{
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;
  ::boba::Array<::boba::index_t, output_dimension> output_indices;
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
 * @brief Creates a deterministic dense tensor containing a sparse pattern.
 *
 * The seed changes which entries are retained. This gives the reference
 * calculations both explicit zeros and signed nonzero values.
 *
 * @tparam dimension Tensor rank.
 * @param sizes Tensor extents.
 * @param seed Offset used to vary the retained-entry pattern.
 * @return Dense host tensor used to construct sparse test inputs.
 */
template <std::size_t dimension>
::boba::Tensor<dimension, ::boba::host_space, double> make_sparse_test_tensor(
  ::boba::Array<::boba::index_t, dimension> sizes,
  ::boba::index_t seed)
{
  ::boba::Tensor<dimension, ::boba::host_space, double> tensor(sizes);
  auto tensor_view = tensor.view();
  for (::boba::index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
  {
    const bool is_nonzero = ((linear_index + seed) % 3 == 0) || ((linear_index + seed) % 7 == 0);
    const double sign = (linear_index % 2 == 0) ? 1.0 : -1.0;
    tensor_view(linear_index) = is_nonzero ? sign * static_cast<double>((linear_index % 11) + 1) : 0.0;
  }
  return tensor;
}

/**
 * @brief Compares a sparse tensor with a dense host reference.
 *
 * The sparse tensor is copied to host when necessary, then its implicit-zero
 * expansion is compared using the infinity norm.
 *
 * @tparam dimension Tensor rank.
 * @tparam sparse_space Execution space of the sparse tensor.
 * @param sparse Sparse tensor under test.
 * @param expected Dense reference tensor.
 * @param check Accumulated test status.
 */
template <std::size_t dimension, ::boba::execution_space sparse_space>
void compare_sparse_to_dense(
  ::boba::SparseTensor<dimension, sparse_space, double> const& sparse,
  ::boba::Tensor<dimension, ::boba::host_space, double> const& expected,
  bool& check)
{
  pass_or_fail_bool(check, sparse.sizes() == expected.sizes());
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(sparse), expected), 1.0e-14);
}

/**
 * @brief Tests sparse permutation against an explicitly permuted dense tensor.
 *
 * @tparam dimension Tensor rank.
 * @param permutation Output-to-input dimension permutation.
 * @param check Accumulated test status.
 */
template <std::size_t dimension>
void test_sparse_permutation(
  ::boba::Array<::boba::index_t, dimension> permutation,
  bool& check)
{
  std::cout << "Testing sparse permutation dimension " << dimension << std::endl;

  auto dense = make_sparse_test_tensor(make_sizes<dimension>(), 1);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);

  ::boba::Tensor<dimension, ::boba::host_space, double> expected(::boba::permute(dense.sizes(), permutation));
  expected.fill_with_zeros();
  auto dense_view = dense.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const auto indices = dense.multiindex(linear_index);
    expected_view(::boba::permute(indices, permutation)) = dense_view(linear_index);
  }

  ::boba::permute<space, dimension, double, true>(sparse, permutation);
  pass_or_fail_bool(check, sparse.sizes() == expected.sizes());
  compare_sparse_to_dense(sparse, expected, check);

  ::boba::SparseTensor<dimension, ::boba::host_space, double> sparse_host_result(sparse);
  const auto zero_index = ::boba::filled_array<dimension>(static_cast<::boba::index_t>(0));
  auto sparse_host_result_view = sparse_host_result.const_view();
  pass_or_fail(check, sparse_host_result_view(zero_index) - expected_view(zero_index), 1.0e-14);
}

/** 
 * @brief Tests sparse unfold() and fold() against the dense implementations.
 *
 * Checks all three overloads: a single row mode, a sorted row group with
 * implicit columns, and explicit unsorted row and column groups. Each
 * unfolding must match the dense unfolding, and fold() must recover the
 * original tensor.
 */
template <std::size_t dimension>
void test_sparse_unfold_and_fold(
  std::vector<std::size_t> row_dims,
  std::vector<std::size_t> col_dims,
  bool& check)
{
  std::cout << "Testing sparse unfold and fold dimension " << dimension << std::endl;

  const auto sizes = make_sizes<dimension>();
  auto dense = make_sparse_test_tensor(sizes, 2);
  auto sparse_host = ::boba::from_dense(dense);
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);

  const std::size_t mode_k = row_dims[0];
  auto mode_unfolding = ::boba::unfold(sparse, mode_k);
  compare_sparse_to_dense(mode_unfolding, ::boba::unfold(dense, mode_k), check);
  compare_sparse_to_dense(::boba::fold(mode_unfolding, sizes, mode_k), dense, check);

  auto group_unfolding = ::boba::unfold(sparse, row_dims);
  compare_sparse_to_dense(group_unfolding, ::boba::unfold(dense, row_dims), check);
  compare_sparse_to_dense(::boba::fold(group_unfolding, sizes, row_dims), dense, check);

  auto explicit_unfolding = ::boba::unfold(sparse, row_dims, col_dims);
  pass_or_fail_bool(check, explicit_unfolding.nnz() == sparse.nnz());
  compare_sparse_to_dense(explicit_unfolding, ::boba::unfold(dense, row_dims, col_dims), check);
  compare_sparse_to_dense(::boba::fold(explicit_unfolding, sizes, row_dims, col_dims), dense, check);
}

/**
 * @brief Tests sparse reduction against a dense reference calculation.
 *
 * @tparam reductions Number of dimensions summed out.
 * @tparam dimension Input tensor rank.
 * @param reduction_dimensions Dimensions to reduce.
 * @param check Accumulated test status.
 */
template <std::size_t reductions, std::size_t dimension>
void test_sparse_reduction(
  ::boba::Array<size_t, reductions> reduction_dimensions,
  bool& check)
{
  std::cout << "Testing sparse reduction dimension " << dimension << " reductions " << reductions << std::endl;
  constexpr std::size_t output_dimension = dimension - reductions;

  auto dense = make_sparse_test_tensor(make_sizes<dimension>(), 2);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);

  auto output_mider = ::boba::make_contracted_dimensions<reductions>(dense.sizes(), reduction_dimensions);
  ::boba::Tensor<output_dimension, ::boba::host_space, double> expected(output_mider.sizes());
  expected.fill_with_zeros();
  auto dense_view = dense.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < dense.size(); ++linear_index)
  {
    const auto indices = dense.multiindex(linear_index);
    expected_view(::boba::delete_elements(indices, reduction_dimensions)) += dense_view(linear_index);
  }

  auto reduced = ::boba::tensor_reduction<reductions, space, dimension, double, true>(sparse, reduction_dimensions);
  pass_or_fail_bool(check, reduced.sizes() == expected.sizes());
  compare_sparse_to_dense(reduced, expected, check);
}

/**
 * @brief Tests sparse-sparse contraction against dense enumeration.
 *
 * @tparam contractions Number of contracted dimensions.
 * @tparam dimension_A Rank of the left tensor.
 * @tparam dimension_B Rank of the right tensor.
 * @param sizes_A Left tensor extents.
 * @param sizes_B Right tensor extents.
 * @param contraction_dimensions_A Contracted dimensions in the left tensor.
 * @param contraction_dimensions_B Contracted dimensions in the right tensor.
 * @param check Accumulated test status.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
void test_sparse_contraction(
  ::boba::Array<::boba::index_t, dimension_A> sizes_A,
  ::boba::Array<::boba::index_t, dimension_B> sizes_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B,
  bool& check)
{
  std::cout << "Testing sparse contraction dimensions " << dimension_A << " and " << dimension_B << " contractions " << contractions << std::endl;
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;

  auto dense_A = make_sparse_test_tensor(sizes_A, 0);
  auto dense_B = make_sparse_test_tensor(sizes_B, 4);
  auto sparse_A_host = ::boba::convert_to_SparseTensor(dense_A, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto sparse_B_host = ::boba::convert_to_SparseTensor(dense_B, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension_A, space, double> sparse_A(sparse_A_host);
  ::boba::SparseTensor<dimension_B, space, double> sparse_B(sparse_B_host);

  auto contracted_A_mider = ::boba::make_contracted_dimensions<contractions>(sizes_A, contraction_dimensions_A);
  auto contracted_B_mider = ::boba::make_contracted_dimensions<contractions>(sizes_B, contraction_dimensions_B);
  ::boba::Tensor<output_dimension, ::boba::host_space, double> expected(::boba::concatenate(contracted_A_mider.sizes(), contracted_B_mider.sizes()));
  expected.fill_with_zeros();

  auto dense_A_view = dense_A.const_view();
  auto dense_B_view = dense_B.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index_A = 0; linear_index_A < dense_A.size(); ++linear_index_A)
  {
    const auto indices_A = dense_A.multiindex(linear_index_A);
    for (::boba::index_t linear_index_B = 0; linear_index_B < dense_B.size(); ++linear_index_B)
    {
      const auto indices_B = dense_B.multiindex(linear_index_B);
      if (contraction_indices_match(indices_A, indices_B, contraction_dimensions_A, contraction_dimensions_B))
      {
        const auto indices_C = contraction_output_indices(indices_A, indices_B, contraction_dimensions_A, contraction_dimensions_B);
        expected_view(indices_C) += dense_A_view(linear_index_A) * dense_B_view(linear_index_B);
      }
    }
  }

  auto contracted = ::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double, true>(
    sparse_A,
    sparse_B,
    contraction_dimensions_A,
    contraction_dimensions_B);
  pass_or_fail_bool(check, contracted.sizes() == expected.sizes());
  compare_sparse_to_dense(contracted, expected, check);
}

/**
 * @brief Tests sparse x dense and dense x sparse contractions against dense contraction.
 *
 * The unfolded SpMV/SpMM path is checked with vendor and naive CSR kernels,
 * along with the naive host entry traversal.
 *
 * @tparam contractions Number of contracted dimension pairs.
 * @tparam dimension_A Left tensor rank.
 * @tparam dimension_B Right tensor rank.
 * @param sizes_A Left tensor extents.
 * @param sizes_B Right tensor extents.
 * @param contraction_dimensions_A Contracted dimensions in the left tensor.
 * @param contraction_dimensions_B Contracted dimensions in the right tensor.
 * @param check Accumulated test status.
 */
template <std::size_t contractions, std::size_t dimension_A, std::size_t dimension_B>
void test_mixed_contraction(
  ::boba::Array<::boba::index_t, dimension_A> sizes_A,
  ::boba::Array<::boba::index_t, dimension_B> sizes_B,
  ::boba::Array<size_t, contractions> contraction_dimensions_A,
  ::boba::Array<size_t, contractions> contraction_dimensions_B,
  bool& check)
{
  std::cout << "Testing mixed contraction dimensions " << dimension_A << " and " << dimension_B << " contractions " << contractions << std::endl;
  constexpr std::size_t output_dimension = dimension_A + dimension_B - 2 * contractions;
  const auto is_nonzero = [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  };

  const auto dense_A_host = make_sparse_test_tensor(sizes_A, 0);
  const auto dense_B_host = make_sparse_test_tensor(sizes_B, 4);
  const auto sparse_A_host = ::boba::convert_to_SparseTensor(dense_A_host, is_nonzero);
  const auto sparse_B_host = ::boba::convert_to_SparseTensor(dense_B_host, is_nonzero);
  const ::boba::Tensor<dimension_A, space, double> dense_A(dense_A_host);
  const ::boba::Tensor<dimension_B, space, double> dense_B(dense_B_host);
  const ::boba::SparseTensor<dimension_A, space, double> sparse_A(sparse_A_host);
  const ::boba::SparseTensor<dimension_B, space, double> sparse_B(sparse_B_host);

  const auto expected = ::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, double>(
    dense_A_host, dense_B_host, contraction_dimensions_A, contraction_dimensions_B);

  const auto compare = [&](auto const& result)
  {
    const ::boba::Tensor<output_dimension, ::boba::host_space, double> actual(result);
    pass_or_fail_bool(check, actual.sizes() == expected.sizes());
    pass_or_fail(check, ::boba::norm_difference_inf(actual, expected), 1.0e-12);
  };

  compare(::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double>(
    sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B));
  compare(::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double>(
    dense_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B));
  compare(::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double, true>(
    sparse_A, dense_B, contraction_dimensions_A, contraction_dimensions_B));
  compare(::boba::tensor_contraction<contractions, space, dimension_A, dimension_B, double, true>(
    dense_A, sparse_B, contraction_dimensions_A, contraction_dimensions_B));
  compare(::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, double, true>(
    sparse_A_host, dense_B_host, contraction_dimensions_A, contraction_dimensions_B));
  compare(::boba::tensor_contraction<contractions, ::boba::host_space, dimension_A, dimension_B, double, true>(
    dense_A_host, sparse_B_host, contraction_dimensions_A, contraction_dimensions_B));
}

/**
 * @brief Tests label-based sparse reduction and contraction APIs.
 *
 * Each label-based operation is compared with the corresponding positional
 * operation after explicitly matching the output mode order.
 *
 * @param check Accumulated test status.
 */
void test_sparse_label_apis(bool& check)
{
  std::cout << "Testing sparse label APIs" << std::endl;
  auto dense = make_sparse_test_tensor<3>({2, 3, 4}, 3);
  auto sparse = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });

  auto reduced = ::boba::tensor_reduction<1, ::boba::host_space, 3, double, true>(
    ::boba::Array<std::string, 3>{"i", "j", "k"},
    sparse,
    ::boba::Array<std::string, 2>{"k", "i"});
  auto direct = ::boba::tensor_reduction<1, ::boba::host_space, 3, double, true>(
    sparse,
    ::boba::Array<size_t, 1>{1});
  ::boba::permute<::boba::host_space, 2, double, true>(direct, ::boba::Array<::boba::index_t, 2>{1, 0});

  ::boba::Tensor<2, ::boba::host_space, double> expected(direct.sizes());
  expected.fill_with_zeros();
  auto direct_view = direct.const_view();
  auto expected_view = expected.view();
  for (::boba::index_t linear_index = 0; linear_index < expected.size(); ++linear_index)
  {
    expected_view(linear_index) = direct_view(linear_index);
  }
  compare_sparse_to_dense(reduced, expected, check);

  auto dense_A = make_sparse_test_tensor<2>({2, 3}, 5);
  auto dense_B = make_sparse_test_tensor<2>({4, 2}, 6);
  auto sparse_A = ::boba::convert_to_SparseTensor(dense_A, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto sparse_B = ::boba::convert_to_SparseTensor(dense_B, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  auto contracted = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, double, true>(
    ::boba::Array<std::string, 2>{"i", "j"},
    sparse_A,
    ::boba::Array<std::string, 2>{"k", "i"},
    sparse_B,
    ::boba::Array<std::string, 2>{"k", "j"});
  auto direct_contracted = ::boba::tensor_contraction<1, ::boba::host_space, 2, 2, double, true>(
    sparse_A,
    sparse_B,
    ::boba::Array<size_t, 1>{0},
    ::boba::Array<size_t, 1>{1});
  ::boba::permute<::boba::host_space, 2, double, true>(direct_contracted, ::boba::Array<::boba::index_t, 2>{1, 0});

  ::boba::Tensor<2, ::boba::host_space, double> contraction_expected(direct_contracted.sizes());
  contraction_expected.fill_with_zeros();
  auto direct_contracted_view = direct_contracted.const_view();
  auto contraction_expected_view = contraction_expected.view();
  for (::boba::index_t linear_index = 0; linear_index < contraction_expected.size(); ++linear_index)
  {
    contraction_expected_view(linear_index) = direct_contracted_view(linear_index);
  }
  compare_sparse_to_dense(contracted, contraction_expected, check);
}

/**
 * @brief Tests removal of entries that are tiny relative to the sparse norm.
 *
 * filter_small_entries() drops an entry when `abs(value) / norm_inf` is within
 * machine epsilon. Here norm_inf is 2.0.
 * - `1e-18` is far below `2.0 * eps`, so it is dropped even though it is
 *   nonzero.
 * - The explicit zero is dropped too, so filtering removes stored zeros as
 *   well as tiny values.
 * - `-2.0` sets the norm and is kept. Its negative sign checks that the
 *   magnitude, not the signed value, is compared.
 *
 * @param check Accumulated test status.
 */
void test_sparse_filter_small_entries(bool& check)
{
  std::cout << "Testing sparse small-entry filtering" << std::endl;
  ::boba::SparseTensor<1, ::boba::host_space, double> sparse({4});
  sparse.set({0}, 1.0);
  sparse.set({1}, 1.0e-18);
  sparse.set({2}, 0.0);
  sparse.set({3}, -2.0);

  pass_or_fail(check, ::boba::norm_inf(sparse) - 2.0, 1.0e-14);
  auto filtered = ::boba::filter_small_entries(sparse);
  pass_or_fail_bool(check, filtered.nnz() == 2);
  pass_or_fail(check, filtered({0}) - 1.0, 1.0e-14);
  pass_or_fail(check, filtered({1}), 1.0e-14);
  pass_or_fail(check, filtered({2}), 1.0e-14);
  pass_or_fail(check, filtered({3}) + 2.0, 1.0e-14);
}

template <std::size_t dimension, typename data_t>
void validate_strictly_increasing(::boba::SparseTensor<dimension, ::boba::host_space, data_t> const& sparse, bool& check)
{
  auto keys = sparse.linear_indices().const_view();
  bool increasing = true;
  for (::boba::index_t entry = 1; entry < sparse.nnz(); ++entry)
  {
    increasing = increasing && keys(entry - 1) < keys(entry);
  }
  pass_or_fail_bool(check, increasing);
}

template <std::size_t dimension, typename data_t>
double dense_difference(::boba::SparseTensor<dimension, space, data_t> const& sparse,
                        ::boba::Tensor<dimension, space, data_t> const& expected)
{
  return ::boba::norm_difference_inf(::boba::to_dense(sparse), expected);
}

/**
 * @brief Tests sparse reductions, norms, and value transforms against the dense tensor.
 *
 * The input from make_sparse_test_tensor() has both signs and unstored
 * positions, so the reductions must count implicit zeros. Its stored values
 * are all nonzero, so explicit zeros do not affect this test.
 *
 * nonnegative_part() and nonpositive_part() map entries of the wrong sign to
 * zero and then drop them. The result must contain no stored zeros, and
 * because no input value is zero, the two parts must split the entries
 * exactly: `nnz(positive) + nnz(negative) == nnz(input)`.
 *
 * Scaling keeps the sparsity pattern, so nnz() must not change after `*=` and
 * `/=`.
 */
template <std::size_t dimension>
void test_sparse_value_operations(bool& check)
{
  std::cout << "Testing sparse value operations dimension " << dimension << std::endl;

  auto dense_host = make_sparse_test_tensor(make_sizes<dimension>(), 1);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense_host, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  ::boba::SparseTensor<dimension, space, double> sparse(sparse_host);
  ::boba::Tensor<dimension, space, double> dense(dense_host);
  const double scale = std::max(dense.max_abs_reduce(), 1.0);

  pass_or_fail(check, dense_difference(sparse, dense), 1.0e-14);

  pass_or_fail(check, ::boba::abs(::boba::norm_frobenius(sparse) - ::boba::norm_frobenius(dense)) / scale, 1.0e-14);
  pass_or_fail(check, ::boba::abs(sparse.sum_reduce() - dense.sum_reduce()) / scale, 1.0e-14);
  pass_or_fail(check, ::boba::abs(sparse.max_reduce() - dense.max_reduce()), 1.0e-14);
  pass_or_fail(check, ::boba::abs(sparse.min_reduce() - dense.min_reduce()), 1.0e-14);
  pass_or_fail(check, ::boba::abs(sparse.max_abs_reduce() - dense.max_abs_reduce()), 1.0e-14);
  pass_or_fail(check, ::boba::abs(sparse.min_abs_reduce() - dense.min_abs_reduce()), 1.0e-14);
  ::boba::nan_check(sparse);

  auto sparse_nonnegative = ::boba::nonnegative_part(sparse);
  pass_or_fail(check, dense_difference(sparse_nonnegative, ::boba::nonnegative_part(dense)), 1.0e-14);
  pass_or_fail_bool(check, sparse_nonnegative.min_reduce() >= 0.0);
  ::boba::SparseTensor<dimension, ::boba::host_space, double> nonnegative_host(sparse_nonnegative);
  auto nonnegative_values = nonnegative_host.values().const_view();
  bool no_stored_zeros = true;
  for (::boba::index_t entry = 0; entry < nonnegative_host.nnz(); ++entry)
  {
    no_stored_zeros = no_stored_zeros && nonnegative_values(entry) > 0.0;
  }
  pass_or_fail_bool(check, no_stored_zeros);

  auto sparse_nonpositive = ::boba::nonpositive_part(sparse);
  pass_or_fail(check, dense_difference(sparse_nonpositive, ::boba::nonpositive_part(dense)), 1.0e-14);
  pass_or_fail_bool(check, sparse_nonnegative.nnz() + sparse_nonpositive.nnz() == sparse.nnz());

  pass_or_fail(check, dense_difference(::boba::power(sparse, 2.0), ::boba::power(dense, 2.0)) / (scale * scale), 1.0e-14);
  pass_or_fail(check, dense_difference(sparse * 3.0, dense * 3.0), 1.0e-13);
  pass_or_fail(check, dense_difference(3.0 * sparse, 3.0 * dense), 1.0e-13);
  pass_or_fail(check, dense_difference(sparse / 4.0, dense / 4.0), 1.0e-14);
  pass_or_fail(check, dense_difference(-sparse, -dense), 1.0e-14);
  pass_or_fail(check, dense_difference(::boba::get_conj(sparse), dense), 1.0e-14);

  auto scaled = sparse;
  scaled *= 2.0;
  scaled /= 8.0;
  pass_or_fail(check, dense_difference(scaled, dense * 0.25), 1.0e-14);
  pass_or_fail_bool(check, scaled.nnz() == sparse.nnz());
}

/**
 * @brief Tests reductions where implicit zeros change the result.
 *
 * A sparse reduction computes over stored values and then adjusts for
 * implicit zeros when `nnz() < size()`. Each case checks one side of that rule:
 * - Empty tensor: there are no stored values, so every reduction and norm must
 *   return 0. In particular, norm_inf() must not return the max-reduction
 *   identity (the lowest double).
 * - Full tensor of negatives: every position is stored, so there are no
 *   implicit zeros. max_reduce() must return -2, not 0, and min_abs_reduce()
 *   must return 2.
 * - The same tensor with one entry erased: the missing position is an implicit
 *   0, which becomes the maximum and the smallest magnitude. min_reduce() is
 *   unaffected and stays -5.
 * - Complex values: norms use the modulus, so `3 + 4i` contributes 5. This
 *   gives Frobenius `sqrt(25 + 4)`, l1 `5 + 2 = 7`, and inf 5. Conjugation and
 *   complex scaling must act on values without changing the stored keys.
 */
void test_sparse_value_edge_cases(bool& check)
{
  std::cout << "Testing sparse value operation edge cases" << std::endl;

  ::boba::SparseTensor<2, space, double> empty({3, 4});
  pass_or_fail(check, empty.sum_reduce(), 1.0e-14);
  pass_or_fail(check, empty.max_reduce(), 1.0e-14);
  pass_or_fail(check, empty.min_reduce(), 1.0e-14);
  pass_or_fail(check, empty.max_abs_reduce(), 1.0e-14);
  pass_or_fail(check, empty.min_abs_reduce(), 1.0e-14);
  pass_or_fail(check, ::boba::norm_frobenius(empty), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(empty), 1.0e-14);
  pass_or_fail(check, ::boba::norm_inf(empty), 1.0e-14);
  pass_or_fail(check, ::boba::norm_inf(::boba::to_dense(empty)), 1.0e-14);

  // Every entry stored: implicit zeros must not enter the reductions.
  ::boba::SparseTensor<1, ::boba::host_space, double> full_host({3});
  full_host.set({0}, -3.0);
  full_host.set({1}, -2.0);
  full_host.set({2}, -5.0);
  ::boba::SparseTensor<1, space, double> full(full_host);
  pass_or_fail(check, full.max_reduce() + 2.0, 1.0e-14);
  pass_or_fail(check, full.min_reduce() + 5.0, 1.0e-14);
  pass_or_fail(check, full.min_abs_reduce() - 2.0, 1.0e-14);

  // One entry missing: implicit zero becomes the max and min magnitude.
  full_host.erase({1});
  ::boba::SparseTensor<1, space, double> partial(full_host);
  pass_or_fail(check, partial.max_reduce(), 1.0e-14);
  pass_or_fail(check, partial.min_abs_reduce(), 1.0e-14);
  pass_or_fail(check, partial.min_reduce() + 5.0, 1.0e-14);

  using complex_t = ::boba::complex<double>;
  ::boba::SparseTensor<2, ::boba::host_space, complex_t> complex_host({2, 3});
  complex_host.set({0, 1}, complex_t{3.0, 4.0});
  complex_host.set({1, 2}, complex_t{0.0, -2.0});
  ::boba::SparseTensor<2, space, complex_t> complex_sparse(complex_host);
  ::boba::Tensor<2, space, complex_t> complex_dense = ::boba::to_dense(complex_sparse);
  pass_or_fail(check, ::boba::norm_frobenius(complex_sparse) - std::sqrt(29.0), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(complex_sparse) - 7.0, 1.0e-14);
  pass_or_fail(check, ::boba::norm_inf(complex_sparse) - 5.0, 1.0e-14);
  pass_or_fail(check, dense_difference(::boba::get_conj(complex_sparse), ::boba::get_conj(complex_dense)), 1.0e-14);
  pass_or_fail(check, dense_difference(complex_sparse * complex_t{0.0, 1.0}, complex_dense * complex_t{0.0, 1.0}), 1.0e-14);
}

/**
 * @brief Tests device-side compaction on a 23 x 19 x 17 tensor with 3186 stored entries.
 *
 * filter_small_entries(), nonnegative_part(), and nonpositive_part() all
 * compact on `space`: flag kept entries, exclusive-scan the flags to get
 * output offsets, then scatter. Two `1e-20` entries are added at the first
 * and last linear indices. The last one exercises the offset computation for
 * the final entry, which reads the last flag and the last scan offset back to
 * host. The result must stay strictly increasing, so the scatter must
 * preserve order.
 */
void test_sparse_large_compaction(bool& check)
{
  std::cout << "Testing sparse compaction on a large tensor" << std::endl;
  auto dense_host = make_sparse_test_tensor(::boba::Array<::boba::index_t, 3>{23, 19, 17}, 2);
  auto sparse_host = ::boba::convert_to_SparseTensor(dense_host, [](double value, auto)
  {
    return ::boba::abs(value) > 0.0;
  });
  // Add tiny entries that filter_small_entries must remove.
  sparse_host.set({0, 0, 0}, 1.0e-20);
  sparse_host.set({22, 18, 16}, -1.0e-20);
  ::boba::SparseTensor<3, space, double> sparse(sparse_host);
  ::boba::Tensor<3, space, double> dense(dense_host);

  auto filtered = ::boba::filter_small_entries(sparse);
  pass_or_fail_bool(check, filtered.nnz() + 2 == sparse.nnz());
  pass_or_fail(check, dense_difference(filtered, dense), 1.0e-14);

  auto nonnegative = ::boba::nonnegative_part(filtered);
  auto nonpositive = ::boba::nonpositive_part(filtered);
  pass_or_fail(check, dense_difference(nonnegative, ::boba::nonnegative_part(dense)), 1.0e-14);
  pass_or_fail(check, dense_difference(nonpositive, ::boba::nonpositive_part(dense)), 1.0e-14);
  pass_or_fail_bool(check, nonnegative.nnz() + nonpositive.nnz() == filtered.nnz());

  ::boba::SparseTensor<3, ::boba::host_space, double> nonnegative_host(nonnegative);
  validate_strictly_increasing(nonnegative_host, check);
}

int main()
{
  boba::init();

  bool check = true;

  test_sparse_permutation<2>({1, 0}, check);
  test_sparse_permutation<3>({2, 0, 1}, check);
  test_sparse_permutation<4>({3, 1, 0, 2}, check);
  test_sparse_permutation<5>({4, 2, 0, 3, 1}, check);
  test_sparse_permutation<6>({5, 0, 4, 1, 3, 2}, check);
  test_sparse_unfold_and_fold<2>({1}, {0}, check);
  test_sparse_unfold_and_fold<3>({2, 0}, {1}, check);
  test_sparse_unfold_and_fold<4>({3, 1}, {2, 0}, check);
  test_sparse_unfold_and_fold<5>({4, 0, 2}, {3, 1}, check);
  test_sparse_reduction<1, 2>({0}, check);
  test_sparse_reduction<1, 3>({1}, check);
  test_sparse_reduction<2, 4>({0, 3}, check);
  test_sparse_reduction<2, 5>({1, 4}, check);
  test_sparse_reduction<3, 6>({0, 2, 5}, check);
  test_sparse_contraction<1, 2, 2>({2, 3}, {3, 4}, {1}, {0}, check);
  test_sparse_contraction<2, 3, 3>({2, 3, 4}, {4, 5, 3}, {1, 2}, {2, 0}, check);
  test_sparse_contraction<1, 4, 3>({2, 2, 2, 2}, {2, 2, 2}, {1}, {0}, check);
  test_sparse_contraction<3, 6, 4>({2, 2, 2, 2, 2, 2}, {2, 2, 2, 2}, {0, 2, 5}, {1, 2, 3}, check);
  test_mixed_contraction<1, 2, 2>({2, 3}, {3, 4}, {1}, {0}, check);
  test_mixed_contraction<1, 3, 2>({3, 4, 5}, {4, 2}, {1}, {0}, check);
  test_mixed_contraction<2, 3, 3>({2, 3, 4}, {4, 5, 3}, {1, 2}, {2, 0}, check);
  test_mixed_contraction<2, 4, 3>({2, 3, 4, 5}, {5, 3, 2}, {3, 1}, {0, 1}, check);
  test_mixed_contraction<2, 3, 2>({2, 3, 4}, {4, 3}, {2, 1}, {0, 1}, check);
  test_mixed_contraction<2, 2, 3>({4, 3}, {3, 5, 4}, {1, 0}, {0, 2}, check);
  test_sparse_label_apis(check);
  test_sparse_filter_small_entries(check);
  test_sparse_value_operations<2>(check);
  test_sparse_value_operations<3>(check);
  test_sparse_value_operations<5>(check);
  test_sparse_value_edge_cases(check);
  test_sparse_large_compaction(check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}
