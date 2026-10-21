// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <BOBA/boba.hpp>
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>

/*
  Tests for SparseTensor construction, element access, and storage invariants.

  Each check compares against a dense reference or a known entry count. Entries
  are stored by strictly increasing column-major linear index.
*/

constexpr boba::execution_space space = boba::default_execution_space;

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_sizes()
{
  ::boba::Array<::boba::index_t, dimension> sizes;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    sizes[d] = static_cast<::boba::index_t>(d + 4);
  }
  return sizes;
}

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_middle_index(
  ::boba::Array<::boba::index_t, dimension> sizes)
{
  ::boba::Array<::boba::index_t, dimension> indices;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    indices[d] = sizes[d] / 2;
  }
  return indices;
}

template <std::size_t dimension>
::boba::Array<::boba::index_t, dimension> make_last_index(
  ::boba::Array<::boba::index_t, dimension> sizes)
{
  ::boba::Array<::boba::index_t, dimension> indices;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    indices[d] = sizes[d] - 1;
  }
  return indices;
}

/**
 * \brief Checks that point lookups through a sparse view on `sparse_space` match \p expected.
 *
 * Every dense position is read through `SparseTensorView::operator()`, so
 * unstored positions must read as zero.
 */
template <std::size_t dimension, ::boba::execution_space sparse_space>
void validate_view(const ::boba::SparseTensor<dimension, sparse_space, double>& sparse,
                   const ::boba::SparseTensor<dimension, ::boba::host_space, double>& expected,
                   bool& check)
{
  ::boba::Tensor<dimension, sparse_space, double> dense_from_view(sparse.sizes());
  auto sparse_view = sparse.const_view();
  auto dense_view = dense_from_view.view();
  ::boba::loop<sparse_space, 1>(0_z, dense_from_view.size(), [=] __boba_host_device__(::boba::index_t linear_index)
  {
    dense_view(linear_index) = sparse_view(linear_index);
  });

  pass_or_fail(check, ::boba::norm_difference_inf(dense_from_view, ::boba::to_dense(expected)), 1.0e-14);
}

/**
 * \brief Checks the storage invariants of a host sparse tensor.
 *
 * There must be one linear index per entry, the view must alias the owned
 * index buffer, and linear indices must be strictly increasing.
 */
template <std::size_t dimension>
void validate_host_linear_indices(const ::boba::SparseTensor<dimension, ::boba::host_space, double>& sparse,
                                  bool& check)
{
  auto keys = sparse.linear_indices().const_view();
  auto sparse_view = sparse.const_view();
  pass_or_fail_bool(check, sparse.linear_indices().size() == sparse.nnz());
  pass_or_fail_bool(check, sparse_view.linear_indices_ptr() == keys.data());
  ::boba::index_t number_out_of_order = 0;
  if (sparse.nnz() > 1)
  {
    ::boba::sum_reduce<::boba::host_space>(number_out_of_order, 1_z, sparse.nnz(), [=](::boba::index_t entry, ::boba::sum_reducer_operator<::boba::index_t>& local_count)
    {
      if (!(keys(entry - 1) < keys(entry)))
      {
        local_count += ::boba::index_t(1);
      }
    });
  }
  pass_or_fail_bool(check, number_out_of_order == 0);
}

/**
 * \brief Tests incremental host edits with set() and erase().
 *
 * set() binary-searches the sorted keys, then either overwrites a found entry
 * or shifts later entries right to insert. erase() shifts later entries left.
 * The edge cases target those two branches and the ends of the key range:
 * - Lookups on an empty tensor must return zero without reading any storage.
 * - Inserting `last_index`, then `zero_index`, then `middle_index` appends at
 *   the end, inserts at the front, and inserts in the middle, so every shift
 *   case runs. These are also linear indices 0 and `size() - 1`, the extremes
 *   of the binary search.
 * - `missing_index` (linear index 1) falls between stored keys 0 and
 *   `middle_index`. The search lands on a neighboring entry, which must not be
 *   reported as a match.
 * - Overwriting a stored entry must take the found branch rather than insert
 *   a duplicate key, so nnz() stays at 3.
 * - Setting a value of zero stores an explicit zero: contains() is true and
 *   nnz() counts it, although operator() reads 0 exactly as for an unstored
 *   entry. Setting it again must find the stored zero instead of treating it
 *   as missing. Only erase() removes it.
 * - Erasing the entry at offset 0 shifts every remaining entry.
 *
 * Finally the tensor is copied to `space` and every dense position is read
 * through the device-side binary search in validate_view().
 */
template <std::size_t dimension>
void test_set_and_erase(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  const auto zero_index = ::boba::filled_array<dimension>(static_cast<::boba::index_t>(0));
  const auto middle_index = make_middle_index(sizes);
  const auto last_index = make_last_index(sizes);
  auto missing_index = zero_index;
  missing_index[0] = 1;

  ::boba::SparseTensor<dimension, ::boba::host_space, double> sparse(sizes);

  pass_or_fail_bool(check, sparse.sizes() == sizes);
  pass_or_fail_bool(check, sparse.size() == ::boba::product(sizes));
  pass_or_fail_bool(check, sparse.nnz() == 0);
  pass_or_fail(check, sparse(middle_index), 1.0e-14);

  // Append, then insert at the front, then insert between the two.
  sparse.set(last_index, 3.25);
  sparse.set(zero_index, 1.5);
  sparse.set(middle_index, -2.0);

  // set() may reallocate storage, so views are taken after each edit.
  auto populated_view = sparse.const_view();
  pass_or_fail_bool(check, sparse.nnz() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail_bool(check, sparse.contains(middle_index));
  pass_or_fail_bool(check, sparse.contains(last_index));
  pass_or_fail(check, sparse(zero_index) - 1.5, 1.0e-14);
  pass_or_fail(check, populated_view(middle_index) + 2.0, 1.0e-14);
  pass_or_fail(check, populated_view(last_index) - 3.25, 1.0e-14);
  // Linear index 1 lies between stored keys, so it must read as zero.
  pass_or_fail(check, populated_view(missing_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 6.75, 1.0e-14);
  validate_host_linear_indices(sparse, check);

  // Overwriting a stored entry keeps nnz unchanged.
  sparse.set(middle_index, 4.5);
  pass_or_fail_bool(check, sparse.nnz() == 3);
  pass_or_fail(check, sparse(middle_index) - 4.5, 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 9.25, 1.0e-14);

  // An explicit zero stays stored; it contributes nothing to norm_l1.
  sparse.set(zero_index, 0.0);
  pass_or_fail_bool(check, sparse.nnz() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 7.75, 1.0e-14);

  // Setting the stored zero again must overwrite it, not insert a second entry.
  sparse.set(zero_index, 0.0);
  pass_or_fail_bool(check, sparse.nnz() == 3);
  pass_or_fail_bool(check, sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);

  // Erasing offset 0 shifts both remaining entries; their order must survive.
  sparse.erase(zero_index);
  auto erased_view = sparse.const_view();
  pass_or_fail_bool(check, sparse.nnz() == 2);
  pass_or_fail_bool(check, !sparse.contains(zero_index));
  pass_or_fail(check, sparse(zero_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_l1(sparse) - 7.75, 1.0e-14);
  validate_host_linear_indices(sparse, check);
  pass_or_fail_bool(check, erased_view.entry_multiindex(0) == middle_index);
  pass_or_fail_bool(check, erased_view.entry_multiindex(1) == last_index);

  ::boba::SparseTensor<dimension, space, double> sparse_space(sparse);
  validate_view(sparse_space, sparse, check);
}

/**
 * \brief Tests convert_to_SparseTensor with value-based and index-based filters.
 *
 * The dense input holds `linear_index + 1`, so every value is nonzero and
 * distinct. Unlike from_dense(), the filter alone decides what is stored.
 * - An even-value filter keeps every other entry. make_sizes() starts with
 *   an extent of 4, so `size()` is even and exactly `size() / 2` entries are
 *   expected.
 * - The first entry is then set to zero and an index-only filter keeps the
 *   slice `indices[0] == 0`. The filter accepts the zero, so it must be stored
 *   as an explicit entry; convert_to_SparseTensor must not drop zeros itself.
 *   The slice has `size() / sizes[0]` positions.
 * - A `space` input is copied to host, filtered there, and copied back, so the
 *   filter may be a host-only lambda. The result must match the host
 *   conversion. The `value > 0.0` term excludes the zeroed entry.
 */
template <std::size_t dimension>
void test_convert_to_sparse_tensor(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  ::boba::Tensor<dimension, ::boba::host_space, double> dense(sizes);
  auto dense_view = dense.view();
  ::boba::loop<::boba::host_space, 1>(0_z, dense.size(), [=](::boba::index_t linear_index)
  {
    dense_view(linear_index) = static_cast<double>(linear_index + 1);
  });

  auto value_filtered = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return static_cast<::boba::index_t>(value) % 2 == 0;
  });

  ::boba::Tensor<dimension, ::boba::host_space, double> expected_value_filtered(dense);
  auto expected_value_filtered_view = expected_value_filtered.view();
  ::boba::loop<::boba::host_space, 1>(0_z, dense.size(), [=](::boba::index_t linear_index)
  {
    if ((linear_index + 1) % 2 != 0)
    {
      expected_value_filtered_view(linear_index) = 0.0;
    }
  });
  pass_or_fail_bool(check, value_filtered.nnz() == dense.size() / 2);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(value_filtered), expected_value_filtered), 1.0e-14);
  validate_host_linear_indices(value_filtered, check);
  validate_view(value_filtered, value_filtered, check);

  dense_view(0) = 0.0;
  auto index_filtered = ::boba::convert_to_SparseTensor(dense, [](double, auto indices)
  {
    return indices[0] == 0;
  });

  ::boba::Tensor<dimension, ::boba::host_space, double> expected_index_filtered(dense);
  auto expected_index_filtered_view = expected_index_filtered.view();
  ::boba::loop<::boba::host_space, 1>(0_z, dense.size(), [=](::boba::index_t linear_index)
  {
    const auto indices = expected_index_filtered_view.multiindex(linear_index);
    if (indices[0] != 0)
    {
      expected_index_filtered_view(linear_index) = 0.0;
    }
  });
  const auto zero_index = ::boba::filled_array<dimension>(static_cast<::boba::index_t>(0));
  pass_or_fail_bool(check, index_filtered.nnz() == dense.size() / sizes[0]);
  pass_or_fail_bool(check, index_filtered.contains(zero_index));
  pass_or_fail(check, index_filtered(zero_index), 1.0e-14);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(index_filtered), expected_index_filtered), 1.0e-14);
  validate_host_linear_indices(index_filtered, check);

  ::boba::Tensor<dimension, space, double> dense_space(dense);
  auto sparse_space = ::boba::convert_to_SparseTensor(dense_space, [](double value, auto)
  {
    return value > 0.0 && static_cast<::boba::index_t>(value) % 3 == 0;
  });
  auto expected_host = ::boba::convert_to_SparseTensor(dense, [](double value, auto)
  {
    return value > 0.0 && static_cast<::boba::index_t>(value) % 3 == 0;
  });

  pass_or_fail_bool(check, sparse_space.nnz() == expected_host.nnz());
  validate_view(sparse_space, expected_host, check);
}

/**
 * \brief Tests from_dense with default and positive absolute tolerances.
 *
 * from_dense() keeps an entry only when `abs(value) > tolerance`. The dense
 * input holds `(-1)^i * i / 4` for linear index `i`:
 *
 *   i      0     1      2     3      4   ...
 *   value  0.0  -0.25   0.5  -0.75   1.0 ...
 *
 * - With the default tolerance of 0, only the exact zero at `i = 0` is
 *   dropped. Every negative value is kept, which catches comparing the signed
 *   value instead of its magnitude.
 * - With tolerance 0.5, `i = 0, 1, 2` are dropped. `i = 2` equals the
 *   tolerance, so it checks that the comparison is strict. `i = 3` is
 *   negative but has magnitude 0.75, so it must be kept.
 * - A `space` input takes the host round trip and must match the host import.
 * - For complex values the tolerance applies to the modulus. `0.4 + 0.4i` has
 *   both parts at or below 0.5 but modulus 0.57, and `2i` has a zero real
 *   part. Both must be kept. `0.3 - 0.3i` has modulus 0.42 and is dropped.
 */
template <std::size_t dimension>
void test_from_dense(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  ::boba::Tensor<dimension, ::boba::host_space, double> dense(sizes);
  auto dense_view = dense.view();
  ::boba::loop<::boba::host_space, 1>(0_z, dense.size(), [=](::boba::index_t linear_index)
  {
    const double sign = linear_index % 2 == 0 ? 1.0 : -1.0;
    dense_view(linear_index) = sign * static_cast<double>(linear_index) / 4.0;
  });

  auto nonzeros = ::boba::from_dense(dense);
  pass_or_fail_bool(check, nonzeros.nnz() == dense.size() - 1);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(nonzeros), dense), 1.0e-14);
  validate_host_linear_indices(nonzeros, check);

  // Linear indices 0, 1, and 2 have magnitudes 0, 0.25, and 0.5, all <= 0.5.
  auto above_tolerance = ::boba::from_dense(dense, 0.5);
  ::boba::Tensor<dimension, ::boba::host_space, double> expected_above_tolerance(dense);
  auto expected_above_tolerance_view = expected_above_tolerance.view();
  ::boba::loop<::boba::host_space, 1>(0_z, 3_z, [=](::boba::index_t linear_index)
  {
    expected_above_tolerance_view(linear_index) = 0.0;
  });
  pass_or_fail_bool(check, above_tolerance.nnz() == dense.size() - 3);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(above_tolerance), expected_above_tolerance), 1.0e-14);

  ::boba::Tensor<dimension, space, double> dense_space(dense);
  auto sparse_space = ::boba::from_dense(dense_space, 0.5);
  pass_or_fail_bool(check, sparse_space.nnz() == above_tolerance.nnz());
  validate_view(sparse_space, above_tolerance, check);

  using complex_t = ::boba::complex<double>;
  ::boba::Tensor<dimension, ::boba::host_space, complex_t> complex_dense(sizes);
  complex_dense.fill_with_zeros();
  auto complex_dense_view = complex_dense.view();
  complex_dense_view(0) = complex_t{0.3, -0.3};
  complex_dense_view(1) = complex_t{0.4, 0.4};
  complex_dense_view(2) = complex_t{0.0, 2.0};
  auto complex_sparse = ::boba::from_dense(complex_dense, 0.5);
  auto complex_sparse_view = complex_sparse.const_view();
  pass_or_fail_bool(check, complex_sparse.nnz() == 2);
  pass_or_fail_bool(check, complex_sparse_view.entry_linear_index(0) == 1);
  pass_or_fail_bool(check, complex_sparse_view.entry_linear_index(1) == 2);
}

/**
 * \brief Tests construction from per-dimension coordinate lists.
 *
 * The coordinates are linearized and passed to detail::sort_and_coalesce().
 * The input is `{last, middle, zero, middle}`:
 * - It is in descending key order, so the strictly-increasing early exit must
 *   not trigger and a real sort must run.
 * - The two `middle_index` entries are not adjacent. Merging them requires
 *   sorting first, so an implementation that only merges neighbors would keep
 *   both. They must sum to `1.0 + 0.5 = 1.5` in a single entry.
 *
 * The host path uses std::stable_sort. The `space` path uses the CUB/hipCUB
 * radix sort and reduce-by-key when available. Both must give the same result.
 */
template <std::size_t dimension>
void test_coordinate_list_construction(bool& check)
{
  const auto sizes = make_sizes<dimension>();
  const auto zero_index = ::boba::filled_array<dimension>(static_cast<::boba::index_t>(0));
  const auto middle_index = make_middle_index(sizes);
  const auto last_index = make_last_index(sizes);

  const ::boba::Array<::boba::Array<::boba::index_t, dimension>, 4> entries{last_index, middle_index, zero_index, middle_index};
  const ::boba::Array<double, 4> entry_values{3.0, 1.0, -2.0, 0.5};

  typename ::boba::SparseTensor<dimension, ::boba::host_space, double>::index_list_array_t host_lists;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    host_lists[d].resize(4);
    auto host_list_view = host_lists[d].view();
    for (::boba::index_t entry = 0; entry < 4; ++entry)
    {
      host_list_view(entry) = entries[entry][d];
    }
  }
  ::boba::Vector<::boba::host_space, double> host_values({4});
  auto host_values_view = host_values.view();
  for (::boba::index_t entry = 0; entry < 4; ++entry)
  {
    host_values_view(entry) = entry_values[entry];
  }

  ::boba::SparseTensor<dimension, ::boba::host_space, double> expected(sizes);
  expected.set(zero_index, -2.0);
  expected.set(middle_index, 1.5);
  expected.set(last_index, 3.0);

  ::boba::SparseTensor<dimension, ::boba::host_space, double> host_sparse(sizes, host_lists, host_values);
  pass_or_fail_bool(check, host_sparse.nnz() == 3);
  validate_host_linear_indices(host_sparse, check);
  validate_view(host_sparse, expected, check);

  typename ::boba::SparseTensor<dimension, space, double>::index_list_array_t space_lists;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    space_lists[d] = host_lists[d];
  }
  ::boba::Vector<space, double> space_values(host_values);
  ::boba::SparseTensor<dimension, space, double> space_sparse(sizes, space_lists, space_values);
  pass_or_fail_bool(check, space_sparse.nnz() == 3);
  validate_view(space_sparse, expected, check);
}

/**
 * \brief Tests large unsorted linear-index construction with heavy duplication.
 *
 * 10000 entries map onto keys below `size() / 3` = 810 of the 2431 positions in
 * 17 x 13 x 11, so each key repeats about 12 times, spread across the input.
 * On `space` this runs the radix sort over `linear_index_bits(size())` key bits
 * and the reduce-by-key merge. A host std::map gives the reference.
 *
 * All values are small integers. The device reduction may associate the sums
 * differently from the host, but integer sums are exact either way, so a
 * 1e-14 tolerance is valid.
 *
 * The sorted result is then used to build a new tensor. Its keys are already
 * strictly increasing, so the early exit must return them unchanged.
 */
void test_large_linear_index_construction(bool& check)
{
  std::cout << "Testing large SparseTensor linear-index construction" << std::endl;
  const ::boba::Array<::boba::index_t, 3> sizes{17, 13, 11};
  const ::boba::index_t size = ::boba::product(sizes);
  const ::boba::index_t number_entries = 10000;

  ::boba::Vector<::boba::host_space, ::boba::index_t> host_keys({number_entries});
  ::boba::Vector<::boba::host_space, double> host_values({number_entries});
  std::map<::boba::index_t, double> reference;
  auto host_keys_view = host_keys.view();
  auto host_values_view = host_values.view();
  for (::boba::index_t entry = 0; entry < number_entries; ++entry)
  {
    const ::boba::index_t key = (entry * 7919 + 13) % (size / 3);
    const double value = static_cast<double>((entry % 5) + 1);
    host_keys_view(entry) = key;
    host_values_view(entry) = value;
    reference[key] += value;
  }

  ::boba::Tensor<3, ::boba::host_space, double> expected(sizes);
  expected.fill_with_zeros();
  auto expected_view = expected.view();
  for (auto const& [key, value] : reference)
  {
    expected_view(key) = value;
  }

  ::boba::Vector<space, ::boba::index_t> space_keys(host_keys);
  ::boba::Vector<space, double> space_values(host_values);
  ::boba::SparseTensor<3, space, double> space_sparse(sizes, std::move(space_keys), std::move(space_values));
  ::boba::SparseTensor<3, ::boba::host_space, double> result(space_sparse);

  // All reference values are positive, so equal dense tensors imply equal keys.
  pass_or_fail_bool(check, result.nnz() == static_cast<::boba::index_t>(reference.size()));
  validate_host_linear_indices(result, check);
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(result), expected), 1.0e-14);

  ::boba::SparseTensor<3, space, double> resorted(sizes, space_sparse.linear_indices(), space_sparse.values());
  pass_or_fail_bool(check, resorted.nnz() == space_sparse.nnz());
  pass_or_fail(check, ::boba::norm_difference_inf(::boba::to_dense(resorted), ::boba::to_dense(space_sparse)), 1.0e-14);
}

/**
 * \brief Tests linear-index construction edge cases on `space`.
 *
 * Each case targets a boundary of detail::sort_and_coalesce():
 * - Zero entries: the device path must return before the bounds check
 *   (max_reduce over no keys) and before calling CUB with zero items.
 * - One entry: a single key is trivially sorted, so the early exit runs and
 *   no sort is launched.
 * - Six copies of one key: the whole input is one run, so reduce-by-key must
 *   report exactly one unique key holding `6 * 0.5 = 3.0`.
 * - Complex values with keys `{4, 1, 4}`: the device sum operator must add
 *   boba::complex values, giving `(1 + 2i) + (3 - 5i) = 4 - 3i` at key 4, and
 *   key 1 must move ahead of it. In the 5 x 4 tensor, key 4 is `(4, 0)` and
 *   key 1 is `(1, 0)`.
 */
void test_linear_index_construction_edge_cases(bool& check)
{
  std::cout << "Testing SparseTensor linear-index construction edge cases" << std::endl;
  const ::boba::Array<::boba::index_t, 2> sizes{5, 4};

  ::boba::SparseTensor<2, space, double> empty(sizes, ::boba::Vector<space, ::boba::index_t>({0}), ::boba::Vector<space, double>({0}));
  pass_or_fail_bool(check, empty.nnz() == 0);

  ::boba::Vector<::boba::host_space, ::boba::index_t> one_key({1});
  ::boba::Vector<::boba::host_space, double> one_value({1});
  auto one_key_view = one_key.view();
  auto one_value_view = one_value.view();
  one_key_view(0) = 7;
  one_value_view(0) = 2.5;
  ::boba::SparseTensor<2, space, double> single(sizes, ::boba::Vector<space, ::boba::index_t>(one_key), ::boba::Vector<space, double>(one_value));
  pass_or_fail_bool(check, single.nnz() == 1);
  pass_or_fail(check, single.sum_reduce() - 2.5, 1.0e-14);

  ::boba::Vector<::boba::host_space, ::boba::index_t> same_keys({6});
  ::boba::Vector<::boba::host_space, double> same_values({6});
  same_keys.fill_with(11);
  same_values.fill_with(0.5);
  ::boba::SparseTensor<2, space, double> merged(sizes, ::boba::Vector<space, ::boba::index_t>(same_keys), ::boba::Vector<space, double>(same_values));
  pass_or_fail_bool(check, merged.nnz() == 1);
  pass_or_fail(check, merged.sum_reduce() - 3.0, 1.0e-14);

  using complex_t = ::boba::complex<double>;
  ::boba::Vector<::boba::host_space, ::boba::index_t> complex_keys({3});
  ::boba::Vector<::boba::host_space, complex_t> complex_values({3});
  auto complex_keys_view = complex_keys.view();
  auto complex_values_view = complex_values.view();
  complex_keys_view(0) = 4;
  complex_keys_view(1) = 1;
  complex_keys_view(2) = 4;
  complex_values_view(0) = complex_t{1.0, 2.0};
  complex_values_view(1) = complex_t{0.0, -1.0};
  complex_values_view(2) = complex_t{3.0, -5.0};
  ::boba::SparseTensor<2, space, complex_t> complex_sparse(sizes, ::boba::Vector<space, ::boba::index_t>(complex_keys), ::boba::Vector<space, complex_t>(complex_values));
  ::boba::SparseTensor<2, ::boba::host_space, complex_t> complex_host(complex_sparse);
  pass_or_fail_bool(check, complex_host.nnz() == 2);
  pass_or_fail(check, ::boba::abs(complex_host({4, 0}) - complex_t{4.0, -3.0}), 1.0e-14);
  pass_or_fail(check, ::boba::abs(complex_host({1, 0}) - complex_t{0.0, -1.0}), 1.0e-14);
}

/**
 * \brief Tests boba::sort on a Vector against std::sort.
 *
 * The 1000 keys `(389 i + 17) mod 251` take only 251 distinct values, so each
 * value appears about 4 times and the input is far from sorted.
 */
void test_vector_sort(bool& check)
{
  std::cout << "Testing boba::sort on Vector" << std::endl;
  const ::boba::index_t n = 1000;
  ::boba::Vector<::boba::host_space, ::boba::index_t> host_input({n});
  auto host_input_view = host_input.view();
  ::boba::loop<::boba::host_space, 1>(0_z, n, [=](::boba::index_t i)
  {
    host_input_view(i) = (i * 389 + 17) % 251;
  });
  ::boba::Vector<space, ::boba::index_t> space_input(host_input);
  ::boba::sort(space_input);
  ::boba::Vector<::boba::host_space, ::boba::index_t> sorted(space_input);

  std::vector<::boba::index_t> sorted_reference(host_input.data(), host_input.data() + n);
  std::sort(sorted_reference.begin(), sorted_reference.end());
  ::boba::Vector<::boba::host_space, ::boba::index_t> expected({n});
  std::copy(sorted_reference.begin(), sorted_reference.end(), expected.data());

  pass_or_fail_bool(check, sorted.size() == n);
  pass_or_fail(check, ::boba::norm_difference_inf(sorted, expected), 1.0e-14);
}

template <std::size_t dimension>
void test_sparse_tensor_dimension(bool& check)
{
  std::cout << "Testing SparseTensor dimension " << dimension << std::endl;
  test_set_and_erase<dimension>(check);
  test_convert_to_sparse_tensor<dimension>(check);
  test_from_dense<dimension>(check);
  test_coordinate_list_construction<dimension>(check);
}

int main()
{
  boba::init();

  bool check = true;

  test_sparse_tensor_dimension<1>(check);
  test_sparse_tensor_dimension<2>(check);
  test_sparse_tensor_dimension<3>(check);
  test_sparse_tensor_dimension<4>(check);
  test_large_linear_index_construction(check);
  test_linear_index_construction_edge_cases(check);
  test_vector_sort(check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}
