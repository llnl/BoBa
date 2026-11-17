// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"
#include "BOBA/tensors/Vector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace boba
{

template <std::size_t _dimension, ::boba::execution_space space, typename _data_t>
struct SparseTensor;

namespace detail
{

/**
 * \brief Applies the dense-to-sparse filter used by convert_to_SparseTensor.
 *
 * The filter must be callable as `filter(value, indices)`, where `value` is
 * the dense tensor entry and `indices` is its tensor multi-index. Returning
 * true stores the entry, even if the value is zero.
 */
template <typename Filter, typename data_t, typename index_array>
bool sparse_filter_accepts(Filter& filter, data_t const& value, index_array indices)
{
  static_assert(std::is_invocable_v<Filter&, data_t const&, index_array>,
                "filter must be callable as filter(value, indices)");
  return static_cast<bool>(filter(value, indices));
}

/**
 * \brief Returns the product of \p sizes, asserting that it fits in index_t.
 *
 * Sparse entries are keyed by their column-major linear index, so the dense
 * size of a sparse tensor must be representable even when it is never allocated.
 */
template <std::size_t dimension>
index_t checked_linear_size(Array<index_t, dimension> sizes)
{
  index_t total = 1;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    boba_always_assert(sizes[d] == 0 || total <= std::numeric_limits<index_t>::max() / sizes[d],
                       "SparseTensor extents overflow the linear index type");
    total *= sizes[d];
  }
  return total;
}

/**
 * \brief Converts per-dimension coordinate lists into column-major linear indices.
 */
template <execution_space space, std::size_t dimension>
Vector<space, index_t> linearize_index_lists(Array<Vector<space, index_t>, dimension> const& index_lists,
                                             Array<index_t, dimension> sizes)
{
  const index_t number_entries = dimension == 0 ? 0 : index_lists[0].size();
  Array<index_t const*, dimension> lists;
  for (std::size_t d = 0; d < dimension; ++d)
  {
    boba_always_assert_equal(index_lists[d].size(), number_entries, "Sparse index lists must have matching sizes");
    const auto index_list_view = index_lists[d].const_view();
    lists[d] = index_list_view.data();
  }

  Vector<space, index_t> linear_indices({number_entries});
  auto linear_view = linear_indices.view();
  const Multiindexer<dimension> indexer(sizes);
  ::boba::loop<space, 1>(0_z, number_entries, [=] __boba_host_device__(index_t entry)
  {
    Array<index_t, dimension> indices;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      indices[d] = lists[d][entry];
    }
    linear_view(entry) = indexer.index(indices);
  });
  return linear_indices;
}

/**
 * \brief Returns true when \p keys is strictly increasing; runs in \p space.
 */
template <execution_space space>
bool keys_strictly_increasing(Vector<space, index_t> const& keys)
{
  const index_t n = keys.size();
  if (n < 2)
  {
    return true;
  }
  const auto keys_view = keys.const_view();
  index_t const* key_data = keys_view.data();
  index_t number_out_of_order = 0;
  ::boba::sum_reduce<space>(number_out_of_order, index_t(1), n, [=] __boba_host_device__(index_t entry, sum_reducer_operator<index_t> & local_count)
  {
    if (!(key_data[entry - 1] < key_data[entry]))
    {
      local_count += index_t(1);
    }
  });
  return number_out_of_order == 0;
}

/**
 * \brief Number of low key bits needed to represent linear indices below \p size.
 */
inline int linear_index_bits(index_t size) noexcept
{
  int bits = 0;
  for (index_t remaining = size > 0 ? size - 1 : 0; remaining > 0; remaining >>= 1)
  {
    ++bits;
  }
  return bits > 0 ? bits : 1;
}

/**
 * \brief Sorts sparse entries by linear index and sums entries with equal indices.
 *
 * On return \p linear_indices is strictly increasing and \p values is aligned
 * with it. Both buffers shrink when duplicates are merged.
 *
 * Host buffers use std::stable_sort. CUDA/HIP buffers built with
 * BOBA_CUDA_LIBS/BOBA_HIP_LIBS use a CUB/hipCUB radix sort followed by
 * reduce-by-key, so data stays on device. Other device configurations copy to
 * host and back. Duplicates are summed in input order on both paths, but the
 * device reduction may associate the sums differently, so results can differ
 * in the last bits.
 *
 * Example: keys {5, 2, 5} with values {1, 3, 4} become keys {2, 5}, values {3, 5}.
 *
 * \param linear_indices Column-major linear indices; each must be below \p size.
 * \param values Values aligned with \p linear_indices.
 * \param size Dense size of the tensor, used for bounds checking.
 */
template <execution_space space, typename data_t>
void sort_and_coalesce(Vector<space, index_t>& linear_indices, Vector<space, data_t>& values, index_t size)
{
  boba_always_assert_equal(linear_indices.size(), values.size(), "Sparse buffers must have matching sizes");
  if constexpr (space == host_space)
  {
    const index_t number_entries = linear_indices.size();
    const auto linear_indices_view = linear_indices.const_view();
    const auto values_view = values.const_view();
    index_t const* keys = linear_indices_view.data();
    data_t const* vals = values_view.data();

    bool strictly_increasing = true;
    for (index_t entry = 1; entry < number_entries; ++entry)
    {
      strictly_increasing = strictly_increasing && keys[entry - 1] < keys[entry];
    }
    if (strictly_increasing)
    {
      if (number_entries > 0)
      {
        boba_always_assert(keys[number_entries - 1] < size, "Sparse linear index out of bounds");
      }
      return;
    }

    std::vector<index_t> order(static_cast<std::size_t>(number_entries));
    std::iota(order.begin(), order.end(), index_t{0});
    std::stable_sort(order.begin(), order.end(), [keys](index_t a, index_t b)
    {
      return keys[a] < keys[b];
    });
    boba_always_assert(keys[order.back()] < size, "Sparse linear index out of bounds");

    Vector<host_space, index_t> sorted_keys({number_entries});
    Vector<host_space, data_t> sorted_values({number_entries});
    auto sorted_keys_view = sorted_keys.view();
    auto sorted_values_view = sorted_values.view();
    index_t* out_keys = sorted_keys_view.data();
    data_t* out_vals = sorted_values_view.data();
    index_t number_unique = 0;
    for (index_t const source : order)
    {
      if (number_unique > 0 && out_keys[number_unique - 1] == keys[source])
      {
        out_vals[number_unique - 1] += vals[source];
      }
      else
      {
        out_keys[number_unique] = keys[source];
        out_vals[number_unique] = vals[source];
        ++number_unique;
      }
    }
    sorted_keys.resize(number_unique);
    sorted_values.resize(number_unique);
    linear_indices = std::move(sorted_keys);
    values = std::move(sorted_values);
  }
  else if constexpr (cub_available<space>())
  {
    const index_t number_entries = linear_indices.size();
    if (number_entries == 0)
    {
      return;
    }
    const auto linear_indices_view = linear_indices.const_view();
    const auto values_view = values.const_view();
    boba_always_assert(reductions::max_reduce<space>(linear_indices_view.data(), number_entries) < size,
                       "Sparse linear index out of bounds");
    if (keys_strictly_increasing(linear_indices))
    {
      return;
    }

    Vector<space, index_t> sorted_keys({number_entries});
    Vector<space, data_t> sorted_values({number_entries});
    auto sorted_keys_view = sorted_keys.view();
    auto sorted_values_view = sorted_values.view();
    sort_pairs<space>(linear_indices_view.data(),
                      sorted_keys_view.data(),
                      values_view.data(),
                      sorted_values_view.data(),
                      number_entries,
                      linear_index_bits(size));

    Vector<space, index_t> unique_keys({number_entries});
    Vector<space, data_t> unique_values({number_entries});
    auto unique_keys_view = unique_keys.view();
    auto unique_values_view = unique_values.view();
    const index_t number_unique = reduce_by_key<space>(sorted_keys_view.data(),
                                                       unique_keys_view.data(),
                                                       sorted_values_view.data(),
                                                       unique_values_view.data(),
                                                       number_entries);
    unique_keys.resize(number_unique);
    unique_values.resize(number_unique);
    linear_indices = std::move(unique_keys);
    values = std::move(unique_values);
  }
  else
  {
    Vector<host_space, index_t> host_indices(linear_indices);
    Vector<host_space, data_t> host_values(values);
    sort_and_coalesce(host_indices, host_values, size);
    linear_indices = host_indices;
    values = host_values;
  }
}

/**
 * \brief Writes sorted sparse entries into a zero-filled dense tensor of the same extents.
 */
template <std::size_t dimension, execution_space space, typename data_t>
void scatter_to_dense(index_t const* linear_indices,
                      data_t const* values,
                      index_t nnz,
                      Tensor<dimension, space, data_t>& dense)
{
  auto dense_view = dense.view();
  ::boba::loop<space, 1>(0_z, nnz, [=] __boba_host_device__(index_t entry)
  {
    dense_view(linear_indices[entry]) = values[entry];
  });
}

/**
 * \brief Applies \p transform to every stored value in place.
 */
template <execution_space space, typename data_t, typename unary_function_t>
void transform_values_in_place(Vector<space, data_t>& values, unary_function_t transform)
{
  auto values_view = values.view();
  ::boba::loop<space, 1>(0_z, values.size(), [=] __boba_host_device__(index_t entry)
  {
    values_view(entry) = transform(values_view(entry));
  });
}


/**
 * \brief Keeps sparse entries whose value satisfies \p keep, preserving order.
 *
 * Device path: flag each entry, exclusive-scan the flags into output offsets,
 * then scatter kept entries. Requires \p keep to be callable on device.
 * Example: values {3, 0, 5, 0} with keep = nonzero give flags {1, 0, 1, 0},
 * offsets {0, 1, 1, 2}, and output entries {0, 2}.
 */
template <execution_space space, typename data_t, typename predicate_t>
void compact_entries_device(Vector<space, index_t>& linear_indices, Vector<space, data_t>& values, predicate_t keep)
{
  const index_t n = values.size();
  if (n == 0)
  {
    return;
  }
  Vector<space, index_t> flags({n});
  auto flags_view = flags.view();
  auto values_in = values.const_view();
  ::boba::loop<space, 1>(0_z, n, [=] __boba_host_device__(index_t entry)
  {
    flags_view(entry) = keep(values_in(entry)) ? index_t(1) : index_t(0);
  });

  Vector<space, index_t> offsets({n});
  auto offsets_view = offsets.view();
  exclusive_scan<space>(flags_view.data(), offsets_view.data(), n);

  index_t last_offset = 0;
  index_t last_flag = 0;
  ::boba::detail::memcpy<host_space, space>(&last_offset, offsets_view.data() + (n - 1), 1);
  ::boba::detail::memcpy<host_space, space>(&last_flag, flags_view.data() + (n - 1), 1);
  ::boba::detail::synchronize<space>();
  const index_t number_kept = last_offset + last_flag;
  if (number_kept == n)
  {
    return;
  }

  Vector<space, index_t> kept_keys({number_kept});
  Vector<space, data_t> kept_values({number_kept});
  auto kept_keys_view = kept_keys.view();
  auto kept_values_view = kept_values.view();
  auto keys_in = linear_indices.const_view();
  ::boba::loop<space, 1>(0_z, n, [=] __boba_host_device__(index_t entry)
  {
    if (flags_view(entry) != 0)
    {
      kept_keys_view(offsets_view(entry)) = keys_in(entry);
      kept_values_view(offsets_view(entry)) = values_in(entry);
    }
  });
  linear_indices = std::move(kept_keys);
  values = std::move(kept_values);
}

/**
 * \brief Returns a sparse tensor with only the entries whose value satisfies \p keep.
 *
 * Retained entries keep their order, so the result stays sorted. CUDA/HIP
 * tensors with CUB/hipCUB are compacted on device (\p keep must be device
 * callable); other non-host configurations round-trip through host.
 */
template <typename sparse_tensor_t, typename predicate_t>
sparse_tensor_t copy_entries_if(sparse_tensor_t const& tensor, predicate_t keep)
{
  constexpr execution_space space = sparse_tensor_t::get_space();
  using data_t = typename sparse_tensor_t::data_t;
  if constexpr (space == host_space)
  {
    auto keys = tensor.linear_indices().const_view();
    auto values = tensor.values().const_view();
    index_t number_kept = 0;
    for (index_t entry = 0; entry < tensor.nnz(); ++entry)
    {
      if (keep(values(entry)))
      {
        ++number_kept;
      }
    }

    Vector<host_space, index_t> kept_keys({number_kept});
    Vector<host_space, data_t> kept_values({number_kept});
    auto kept_keys_view = kept_keys.view();
    auto kept_values_view = kept_values.view();
    index_t kept_entry = 0;
    for (index_t entry = 0; entry < tensor.nnz(); ++entry)
    {
      if (keep(values(entry)))
      {
        kept_keys_view(kept_entry) = keys(entry);
        kept_values_view(kept_entry) = values(entry);
        ++kept_entry;
      }
    }
    return sparse_tensor_t(tensor.sizes(), std::move(kept_keys), std::move(kept_values), tensor.name());
  }
  else if constexpr (cub_available<space>())
  {
    Vector<space, index_t> kept_keys = tensor.linear_indices();
    Vector<space, data_t> kept_values = tensor.values();
    compact_entries_device(kept_keys, kept_values, keep);
    return sparse_tensor_t(tensor.sizes(), std::move(kept_keys), std::move(kept_values), tensor.name());
  }
  else
  {
    using host_tensor_t = SparseTensor<sparse_tensor_t::dimension, host_space, data_t>;
    host_tensor_t host_tensor(tensor);
    return sparse_tensor_t(copy_entries_if(host_tensor, keep));
  }
}

} // namespace detail

/**
 * \brief COO sparse tensor storing one column-major linear index and one value per entry.
 *
 * Invariant: linear indices are strictly increasing, so entries are unique and
 * sorted in the same order as the dense Tensor linearization. Per-dimension
 * coordinates are decoded on demand from the linear index.
 *
 * Execution notes:
 *  - Element access via const_view() (binary search) and per-entry loops such as
 *    permute() and tensor_reduction() remapping run in this tensor's space.
 *  - Anything that re-establishes the invariant -- the bulk constructors,
 *    permute(), tensor_reduction() -- calls detail::sort_and_coalesce(), and
 *    entry removal (filter_small_entries(), nonnegative_part(), ...) calls
 *    detail::copy_entries_if(). Both run on device with CUB/hipCUB
 *    (BOBA_CUDA_LIBS/BOBA_HIP_LIBS) and otherwise copy non-host data to host.
 *  - set(), erase(), contains(), and operator()(indices) are host-only; they
 *    use binary search but insert/erase by shifting, which is O(nnz).
 */
template <std::size_t _dimension, ::boba::execution_space space, typename _data_t>
struct SparseTensor
{
  static constexpr std::size_t dimension = _dimension;

  using data_t = _data_t;
  using value_t = data_t;
  using real_data_t = real_type_t<data_t>;
  using index_array = ::boba::Array<index_t, dimension>;
  using indexer_t = Multiindexer<dimension>;
  using index_list_t = Vector<space, index_t>;
  using index_list_array_t = ::boba::Array<index_list_t, dimension>;
  using linear_indices_t = Vector<space, index_t>;
  using values_list_t = Vector<space, data_t>;
  using accessor = ::boba::DefaultAccessor<data_t>;
  using view_t = SparseTensorView<accessor, dimension>;
  using const_accessor = ::boba::DefaultAccessor<data_t const>;
  using const_view_t = SparseTensorView<const_accessor, dimension>;

  static constexpr std::string_view object_type_name = "SparseTensor";

  SparseTensor() = default;
  SparseTensor(SparseTensor const&) = default;
  SparseTensor(SparseTensor&&) = default;
  SparseTensor& operator=(SparseTensor const&) = default;
  SparseTensor& operator=(SparseTensor&&) = default;
  ~SparseTensor() = default;

  /**
   * \brief Constructs an empty sparse tensor with the requested extents.
   */
  explicit SparseTensor(index_array sizes, std::string_view name = object_type_name)
      : m_sizes(sizes),
        m_strides(indexer_t::precompute_strides(sizes)),
        m_name(name)
  {
    ::boba::detail::checked_linear_size(m_sizes);
  }

  /**
   * \brief Constructs a sparse tensor from per-dimension coordinate lists and values.
   *
   * Entries may be given in any order. Entries with equal coordinates are summed.
   */
  SparseTensor(index_array sizes,
               index_list_array_t const& index_lists,
               values_list_t values,
               std::string_view name = object_type_name)
      : SparseTensor(sizes, ::boba::detail::linearize_index_lists(index_lists, sizes), std::move(values), name)
  {
  }

  /**
   * \brief Constructs a sparse tensor from column-major linear indices and values.
   *
   * Entries may be given in any order. Entries with equal linear indices are summed.
   * See detail::sort_and_coalesce() for where sorting runs.
   */
  SparseTensor(index_array sizes,
               linear_indices_t linear_indices,
               values_list_t values,
               std::string_view name = object_type_name)
      : m_sizes(sizes),
        m_strides(indexer_t::precompute_strides(sizes)),
        m_name(name),
        m_linear_indices(std::move(linear_indices)),
        m_values(std::move(values))
  {
    ::boba::detail::sort_and_coalesce(m_linear_indices, m_values, ::boba::detail::checked_linear_size(m_sizes));
  }

  /**
   * \brief Copy constructor from a sparse tensor in another execution space.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  SparseTensor(SparseTensor<dimension, rhs_space, data_t> const& rhs)
      : m_sizes(rhs.sizes()),
        m_strides(rhs.strides()),
        m_name(rhs.name()),
        m_linear_indices(rhs.linear_indices()),
        m_values(rhs.values())
  {
  }

  /**
   * \brief Copy assignment from a sparse tensor in another execution space.
   */
  template <execution_space rhs_space>
    requires(space != rhs_space)
  SparseTensor& operator=(SparseTensor<dimension, rhs_space, data_t> const& rhs)
  {
    m_sizes = rhs.sizes();
    m_strides = rhs.strides();
    m_name = rhs.name();
    m_linear_indices = rhs.linear_indices();
    m_values = rhs.values();
    return *this;
  }

  /**
   * \brief Converts a tensor multi-index into a linear index.
   */
  constexpr index_t index(index_array indices) const noexcept
  {
    return indexer_t::index(m_strides, indices);
  }

  /**
   * \brief Converts a linear index into a tensor multi-index.
   */
  constexpr index_array multiindex(index_t linear_index) const noexcept
  {
    return indexer_t::multiindex(m_sizes, linear_index);
  }

  /**
   * \brief Validates a tensor multi-index.
   */
  void assert_indices(index_array indices) const noexcept
  {
    boba_assert_nonnegative(indices, "Negative index");
    boba_assert_lt(indices, m_sizes, "Out of bounds");
  }

  /**
   * \brief Inserts or updates an explicitly stored entry on host.
   *
   * Zero values are stored if passed here. Call erase() to remove an entry.
   */
  template <execution_space s = space>
    requires(s == host_space)
  void set(index_array indices, data_t value)
  {
    assert_indices(indices);
    set_indices(indices, value);
  }

  /**
   * \brief Erases an entry on host if it is explicitly stored.
   */
  template <execution_space s = space>
    requires(s == host_space)
  void erase(index_array indices)
  {
    assert_indices(indices);
    erase_indices(indices);
  }

  /**
   * \brief Returns whether an entry is explicitly stored on host.
   */
  template <execution_space s = space>
    requires(s == host_space)
  bool contains(index_array indices) const
  {
    assert_indices(indices);
    return find_position(indices).found;
  }

  /**
   * \brief Returns the stored value on host or zero if absent.
   */
  template <execution_space s = space>
    requires(s == host_space)
  data_t operator()(index_array indices) const
  {
    assert_indices(indices);
    const auto position = find_position(indices);
    if (position.found)
    {
      const auto values = m_values.const_view();
      return values(position.offset);
    }
    return data_t{};
  }

  /**
   * \brief Renames this object.
   */
  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  /**
   * \brief Returns the execution space where sparse buffers are stored.
   */
  static constexpr execution_space get_space() noexcept
  {
    return space;
  }

  /**
   * \brief Returns the compile-time tensor dimension.
   */
  static constexpr std::size_t get_dimension() noexcept
  {
    return dimension;
  }

  /**
   * \brief Returns true when the dense tensor shape has zero total size.
   */
  constexpr bool empty() const noexcept
  {
    return size() == 0;
  }

  /**
   * \brief Returns the extent of one tensor dimension.
   */
  constexpr index_t sizes(index_t d) const noexcept
  {
    return m_sizes[d];
  }

  /**
   * \brief Returns all tensor extents.
   */
  constexpr index_array sizes() const noexcept
  {
    return m_sizes;
  }

  /**
   * \brief Returns the dense tensor size implied by the extents.
   */
  constexpr index_t size() const noexcept
  {
    return ::boba::product(m_sizes);
  }

  constexpr index_t get_number_elements() const noexcept
  {
    return size();
  }

  constexpr index_t get_full_size() const noexcept
  {
    return size();
  }

  /**
   * \brief Returns the dense linearization stride of one tensor dimension.
   */
  constexpr index_t strides(index_t d) const noexcept
  {
    return m_strides[d];
  }

  /**
   * \brief Returns all dense linearization strides.
   */
  constexpr index_array strides() const noexcept
  {
    return m_strides;
  }

  /**
   * \brief Returns the number of explicitly stored sparse entries.
   */
  index_t nnz() const noexcept
  {
    return m_values.size();
  }

  /**
   * \brief Alias of nnz() matching the dense classes' naming; prefer nnz().
   *
   * Unlike the dense `number_nonzeros(tolerance)`, this counts stored entries,
   * including explicitly stored zeros, and takes no tolerance.
   */
  index_t number_nonzeros() const noexcept
  {
    return nnz();
  }

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Returns the strictly increasing column-major linear index of each entry.
   *
   * The list has nnz() entries and is aligned with values().
   */
  linear_indices_t const& linear_indices() const noexcept
  {
    return m_linear_indices;
  }

  /**
   * \brief Returns the sparse value list aligned with linear_indices().
   */
  values_list_t const& values() const noexcept
  {
    return m_values;
  }

  /**
   * \brief Returns the mutable sparse value list.
   *
   * Values may be modified freely; the sparsity pattern (linear_indices()) is
   * fixed, so the sorted-unique invariant is preserved.
   */
  values_list_t& values() noexcept
  {
    return m_values;
  }

  /**
   * \brief Returns a pointer-backed view of this sparse tensor.
   */
  view_t view() noexcept
  {
    const auto linear_indices_view = m_linear_indices.const_view();
    auto values_view = m_values.view();
    return {linear_indices_view.data(), values_view.data(), nnz(), m_sizes, m_strides};
  }

  /**
   * \brief Returns a const pointer-backed view of this sparse tensor.
   */
  const_view_t view() const noexcept
  {
    return const_view();
  }

  /**
   * \brief Returns a const pointer-backed view of this sparse tensor.
   */
  const_view_t const_view() const noexcept
  {
    const auto linear_indices_view = m_linear_indices.const_view();
    const auto values_view = m_values.const_view();
    return {linear_indices_view.data(), values_view.data(), nnz(), m_sizes, m_strides};
  }

  // ---------------------------------------------------------------------------
  // Section: Reductions
  //
  // Reductions match the dense tensor returned by boba::to_dense(): unstored entries
  // count as zeros whenever nnz() < size().
  // ---------------------------------------------------------------------------

  /**
   * \brief Sums all entries.
   */
  data_t sum_reduce() const
  {
    const auto values_view = m_values.const_view();
    return reductions::sum_reduce<space>(values_view.data(), nnz());
  }

  /**
   * \brief Returns the largest entry, including implicit zeros.
   */
  data_t max_reduce() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "max_reduce is disabled for complex data type");
    const auto values_view = m_values.const_view();
    const data_t stored_max = reductions::max_reduce<space>(values_view.data(), nnz());
    return (has_implicit_zeros() && stored_max < data_t{}) ? data_t{} : stored_max;
  }

  /**
   * \brief Returns the smallest entry, including implicit zeros.
   */
  data_t min_reduce() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "min_reduce is disabled for complex data type");
    const auto values_view = m_values.const_view();
    const data_t stored_min = reductions::min_reduce<space>(values_view.data(), nnz());
    return (has_implicit_zeros() && data_t{} < stored_min) ? data_t{} : stored_min;
  }

  /**
   * \brief Returns the largest entry magnitude.
   */
  real_data_t max_abs_reduce() const
  {
    if (nnz() == 0)
    {
      return real_data_t{};
    }
    const auto values_view = m_values.const_view();
    return reductions::max_abs_reduce<space>(values_view.data(), nnz());
  }

  /**
   * \brief Returns the smallest entry magnitude, including implicit zeros.
   */
  real_data_t min_abs_reduce() const
  {
    if (has_implicit_zeros())
    {
      return real_data_t{};
    }
    const auto values_view = m_values.const_view();
    return reductions::min_abs_reduce<space>(values_view.data(), nnz());
  }

  // ---------------------------------------------------------------------------
  // Section: Value transforms
  // ---------------------------------------------------------------------------

  /**
   * \brief Returns a copy with f(x) = max(x, 0) applied and resulting zeros removed.
   */
  SparseTensor nonnegative_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonnegative_part is disabled for complex data type");
    return transformed_without_zeros([] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(x);
    });
  }

  /**
   * \brief Returns a copy with f(x) = max(-x, 0) applied and resulting zeros removed.
   */
  SparseTensor nonpositive_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonpositive_part is disabled for complex data type");
    return transformed_without_zeros([] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(-x);
    });
  }

  /**
   * \brief Applies \p transform to every stored value in place.
   *
   * The sparsity pattern is unchanged, so \p transform should satisfy f(0) = 0
   * for the result to match applying it to boba::to_dense(). Values that become zero
   * stay stored; call filter_small_entries() to drop them.
   */
  template <typename unary_function_t>
  void transform_values(unary_function_t&& transform)
  {
    ::boba::detail::transform_values_in_place(m_values, std::forward<unary_function_t>(transform));
  }

  // ---------------------------------------------------------------------------
  // Section: Scalar arithmetic
  //
  // Scaling keeps the sparsity pattern, so scaling by zero leaves explicit zeros.
  // ---------------------------------------------------------------------------

  SparseTensor& operator*=(data_t const scalar)
  {
    m_values *= scalar;
    return *this;
  }

  SparseTensor operator*(data_t const scalar) const
  {
    SparseTensor output{*this};
    output *= scalar;
    return output;
  }

  SparseTensor& operator/=(data_t const scalar)
  {
    m_values /= scalar;
    return *this;
  }

  SparseTensor operator/(data_t const scalar) const
  {
    SparseTensor output{*this};
    output /= scalar;
    return output;
  }

  SparseTensor operator-() const
  {
    SparseTensor output{*this};
    output *= data_t{-1};
    return output;
  }

  /**
   * \brief Returns a copy without entries whose stored value is exactly zero.
   */
  SparseTensor without_exact_zeros() const
  {
    return ::boba::detail::copy_entries_if(*this, [] __boba_host_device__(data_t value)
    {
      return ::boba::abs(value) > real_data_t{};
    });
  }

private:
  bool has_implicit_zeros() const noexcept
  {
    return nnz() < size();
  }

  // Device lambdas are created in the public callers; nvcc rejects extended
  // lambdas defined inside private member functions.
  template <typename unary_function_t>
  SparseTensor transformed_without_zeros(unary_function_t&& transform) const
  {
    SparseTensor transformed{*this};
    transformed.transform_values(std::forward<unary_function_t>(transform));
    return transformed.without_exact_zeros();
  }

  struct find_result
  {
    index_t offset = 0;
    bool found = false;
  };

  // Host-only: the binary search in entry_lower_bound() is host/device callable,
  // but here it runs on host over host-resident keys.
  template <execution_space s = space>
    requires(s == host_space)
  find_result find_position(index_array indices) const
  {
    const index_t linear_index = index(indices);
    const auto sparse_view = const_view();
    const index_t offset = sparse_view.entry_lower_bound(linear_index);
    return {offset, offset < nnz() && sparse_view.entry_linear_index(offset) == linear_index};
  }

  template <execution_space s = space>
    requires(s == host_space)
  void set_indices(index_array indices, data_t value)
  {
    const auto position = find_position(indices);
    if (position.found)
    {
      auto values = m_values.view();
      values(position.offset) = value;
      return;
    }

    const auto old_nnz = nnz();
    m_linear_indices.resize(old_nnz + 1);
    m_values.resize(old_nnz + 1);

    auto keys = m_linear_indices.view();
    auto values = m_values.view();
    for (index_t entry = old_nnz; entry > position.offset; --entry)
    {
      keys(entry) = keys(entry - 1);
      values(entry) = values(entry - 1);
    }
    keys(position.offset) = index(indices);
    values(position.offset) = value;
  }

  template <execution_space s = space>
    requires(s == host_space)
  void erase_indices(index_array indices)
  {
    const auto position = find_position(indices);
    if (!position.found)
    {
      return;
    }

    auto keys = m_linear_indices.view();
    auto values = m_values.view();
    const auto old_nnz = nnz();
    for (index_t entry = position.offset; entry + 1 < old_nnz; ++entry)
    {
      keys(entry) = keys(entry + 1);
      values(entry) = values(entry + 1);
    }
    m_linear_indices.resize(old_nnz - 1);
    m_values.resize(old_nnz - 1);
  }

  index_array m_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  index_array m_strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  std::string m_name = "SparseTensor";
  linear_indices_t m_linear_indices;
  values_list_t m_values;
};

/**
 * \brief Returns the dense tensor with the same extents and stored entries.
 *
 * Unstored entries are zero. The result lives in the same execution space as \p sparse.
 */
template <std::size_t dimension, execution_space space, typename data_t>
Tensor<dimension, space, data_t> to_dense(SparseTensor<dimension, space, data_t> const& sparse)
{
  Tensor<dimension, space, data_t> dense(sparse.sizes());
  dense.fill_with_zeros();
  const auto sparse_view = sparse.const_view();
  ::boba::detail::scatter_to_dense(sparse_view.linear_indices_ptr(), sparse_view.values_ptr(), sparse.nnz(), dense);
  return dense;
}

/**
 * \brief Identifies whether a type is a SparseTensor specialization.
 */
template <typename tensor_t>
struct is_sparse_tensor : std::false_type
{
};

template <std::size_t dimension, execution_space space, typename data_t>
struct is_sparse_tensor<SparseTensor<dimension, space, data_t>> : std::true_type
{
};

/**
 * \brief Concept satisfied by SparseTensor specializations, including cv/ref-qualified types.
 */
template <typename tensor_t>
concept IsSparseTensorType = is_sparse_tensor<std::remove_cvref_t<tensor_t>>::value;

/**
 * \brief Converts a dense tensor to a sparse tensor using a user filter.
 *
 * The filter is called as `filter(value, indices)` for each dense entry.
 * Entries for which the filter returns true are stored exactly as provided,
 * including values equal to zero.
 *
 * \note Always runs on host: non-host tensors are copied to host, filtered,
 * and copied back, so \p filter may be a host-only callable.
 */
template <std::size_t dimension, execution_space space, typename data_t, typename Filter>
SparseTensor<dimension, space, data_t>
convert_to_SparseTensor(Tensor<dimension, space, data_t> const& tensor, Filter&& filter)
{
  if constexpr (space == host_space)
  {
    using sparse_tensor_t = SparseTensor<dimension, space, data_t>;
    using index_array = typename sparse_tensor_t::index_array;

    Filter& filter_ref = filter;
    auto tensor_view = tensor.const_view();
    index_t nnz = 0;
    for (index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
    {
      const data_t& value = tensor_view(linear_index);
      index_array indices = tensor.multiindex(linear_index);
      if (::boba::detail::sparse_filter_accepts(filter_ref, value, indices))
      {
        ++nnz;
      }
    }

    typename sparse_tensor_t::linear_indices_t linear_indices({nnz});
    typename sparse_tensor_t::values_list_t values({nnz});

    index_t sparse_index = 0;
    auto linear_indices_view = linear_indices.view();
    auto values_view = values.view();
    for (index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
    {
      const data_t& value = tensor_view(linear_index);
      index_array indices = tensor.multiindex(linear_index);
      if (::boba::detail::sparse_filter_accepts(filter_ref, value, indices))
      {
        linear_indices_view(sparse_index) = linear_index;
        values_view(sparse_index) = value;
        ++sparse_index;
      }
    }

    return sparse_tensor_t(tensor.sizes(), std::move(linear_indices), std::move(values));
  }
  else
  {
    Tensor<dimension, host_space, data_t> host_tensor(tensor);
    auto host_sparse = convert_to_SparseTensor(host_tensor, std::forward<Filter>(filter));
    return SparseTensor<dimension, space, data_t>(host_sparse);
  }
}

/**
 * \brief Imports a dense tensor into a sparse tensor using an absolute tolerance.
 *
 * Matches `from_dense` for sparse matrices: entries with
 * `abs(value) <= tolerance` are omitted, so the default stores only nonzeros.
 * Runs on host like convert_to_SparseTensor().
 *
 * \param dense Dense tensor to import.
 * \param tolerance Nonnegative finite retention tolerance.
 */
template <std::size_t dimension, execution_space space, typename data_t>
SparseTensor<dimension, space, data_t>
from_dense(Tensor<dimension, space, data_t> const& dense, real_type_t<data_t> tolerance = 0)
{
  using real_data_t = real_type_t<data_t>;
  boba_always_assert(tolerance >= real_data_t{0}, "Sparse tolerance must be nonnegative");
  boba_always_assert(std::isfinite(static_cast<double>(tolerance)), "Sparse tolerance must be finite");
  return convert_to_SparseTensor(dense, [tolerance](data_t const& value, auto)
  {
    return ::boba::abs(value) > tolerance;
  });
}

} // namespace boba
