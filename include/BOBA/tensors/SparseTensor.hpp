// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"
#include "BOBA/tensors/Vector.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace boba
{

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

} // namespace detail

/**
 * \brief Sparse tensor storing one index list per tensor dimension and one value list.
 */
template <std::size_t _dimension, ::boba::execution_space space, typename _data_t>
struct SparseTensor
{
  static constexpr std::size_t dimension = _dimension;

  using data_t = _data_t;
  using value_t = data_t;
  using index_array = ::boba::Array<index_t, dimension>;
  using indexer_t = Multiindexer<dimension>;
  using index_list_t = Vector<space, index_t>;
  using index_list_array_t = ::boba::Array<index_list_t, dimension>;
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
  }

  /**
   * \brief Constructs a sparse tensor from sparse index-list and value buffers.
   */
  SparseTensor(index_array sizes,
               index_list_array_t index_lists,
               values_list_t values,
               std::string_view name = object_type_name)
      : m_sizes(sizes),
        m_strides(indexer_t::precompute_strides(sizes)),
        m_name(name),
        m_index_lists(std::move(index_lists)),
        m_values(std::move(values))
  {
    for (std::size_t d = 0; d < dimension; ++d)
    {
      boba_always_assert_equal(m_index_lists[d].size(), m_values.size(), "Sparse buffers must have matching sizes");
    }
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
        m_values(rhs.values_tensor())
  {
    for (std::size_t d = 0; d < dimension; ++d)
    {
      m_index_lists[d] = rhs.index_list(d);
    }
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
    for (std::size_t d = 0; d < dimension; ++d)
    {
      m_index_lists[d] = rhs.index_list(d);
    }
    m_values = rhs.values_tensor();
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
      return m_values.const_view()(position.offset);
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
  index_t number_nonzeros() const noexcept
  {
    return m_values.size();
  }

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Returns the sparse index list for tensor dimension \p d.
   *
   * The list has number_nonzeros() entries and is aligned with values_tensor().
   */
  index_list_t const& index_list(std::size_t d) const noexcept
  {
    return m_index_lists[d];
  }

  /**
   * \brief Returns all per-dimension sparse index lists.
   */
  index_list_array_t const& index_lists() const noexcept
  {
    return m_index_lists;
  }

  /**
   * \brief Returns the sparse value list aligned with the index lists.
   */
  values_list_t const& values_tensor() const noexcept
  {
    return m_values;
  }

  /**
   * \brief Returns a pointer-backed view of this sparse tensor.
   */
  view_t view() noexcept
  {
    return {index_list_pointers(), m_values.view().data(), number_nonzeros(), m_sizes, m_strides};
  }

  /**
   * \brief Returns a const pointer-backed view of this sparse tensor.
   */
  const_view_t view() const noexcept
  {
    return {index_list_pointers(), m_values.const_view().data(), number_nonzeros(), m_sizes, m_strides};
  }

  /**
   * \brief Returns a const pointer-backed view of this sparse tensor.
   */
  const_view_t const_view() const noexcept
  {
    return {index_list_pointers(), m_values.const_view().data(), number_nonzeros(), m_sizes, m_strides};
  }

private:
  struct find_result
  {
    index_t offset = 0;
    bool found = false;
  };

  template <execution_space s = space>
    requires(s == host_space)
  find_result find_position(index_array indices) const
  {
    index_t offset = 0;
    while (offset < number_nonzeros() && entry_is_less_than(offset, indices))
    {
      ++offset;
    }
    return {offset, offset < number_nonzeros() && entry_matches(offset, indices)};
  }

  template <execution_space s = space>
    requires(s == host_space)
  void set_indices(index_array indices, data_t value)
  {
    const auto position = find_position(indices);
    if (position.found)
    {
      m_values.view()(position.offset) = value;
      return;
    }

    const auto old_number_nonzeros = number_nonzeros();
    for (std::size_t d = 0; d < dimension; ++d)
    {
      m_index_lists[d].resize({old_number_nonzeros + 1});
    }
    m_values.resize({old_number_nonzeros + 1});

    auto values = m_values.view();
    for (index_t entry = old_number_nonzeros; entry > position.offset; --entry)
    {
      for (std::size_t d = 0; d < dimension; ++d)
      {
        m_index_lists[d].view()(entry) = m_index_lists[d].const_view()(entry - 1);
      }
      values(entry) = values(entry - 1);
    }
    for (std::size_t d = 0; d < dimension; ++d)
    {
      m_index_lists[d].view()(position.offset) = indices[d];
    }
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

    auto values = m_values.view();
    const auto old_number_nonzeros = number_nonzeros();
    for (index_t entry = position.offset; entry + 1 < old_number_nonzeros; ++entry)
    {
      for (std::size_t d = 0; d < dimension; ++d)
      {
        m_index_lists[d].view()(entry) = m_index_lists[d].const_view()(entry + 1);
      }
      values(entry) = values(entry + 1);
    }
    for (std::size_t d = 0; d < dimension; ++d)
    {
      m_index_lists[d].resize({old_number_nonzeros - 1});
    }
    m_values.resize({old_number_nonzeros - 1});
  }

  bool entry_is_less_than(index_t entry, index_array indices) const
  {
    return entry_linear_index(entry) < indexer_t::index(m_strides, indices);
  }

  bool entry_matches(index_t entry, index_array indices) const
  {
    for (std::size_t d = 0; d < dimension; ++d)
    {
      if (m_index_lists[d].const_view()(entry) != indices[d])
      {
        return false;
      }
    }
    return true;
  }

  ::boba::Array<index_t const*, dimension> index_list_pointers() const noexcept
  {
    ::boba::Array<index_t const*, dimension> output;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      output[d] = m_index_lists[d].const_view().data();
    }
    return output;
  }

  index_t entry_linear_index(index_t entry) const
  {
    index_t linear_index = 0;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      linear_index += m_strides[d] * m_index_lists[d].const_view()(entry);
    }
    return linear_index;
  }

  index_array m_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  index_array m_strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  std::string m_name = "SparseTensor";
  index_list_array_t m_index_lists;
  values_list_t m_values;
};

/**
 * \brief Converts a dense tensor to a sparse tensor using a user filter.
 *
 * The filter is called as `filter(value, indices)` for each dense entry.
 * Entries for which the filter returns true are stored exactly as provided,
 * including values equal to zero.
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
    index_t number_nonzeros = 0;
    for (index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
    {
      const data_t& value = tensor_view(linear_index);
      index_array indices = tensor.multiindex(linear_index);
      if (::boba::detail::sparse_filter_accepts(filter_ref, value, indices))
      {
        ++number_nonzeros;
      }
    }

    typename sparse_tensor_t::index_list_array_t index_lists;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      index_lists[d].resize({number_nonzeros});
    }
    typename sparse_tensor_t::values_list_t values({number_nonzeros});

    index_t sparse_index = 0;
    auto values_view = values.view();
    for (index_t linear_index = 0; linear_index < tensor.size(); ++linear_index)
    {
      const data_t& value = tensor_view(linear_index);
      index_array indices = tensor.multiindex(linear_index);
      if (::boba::detail::sparse_filter_accepts(filter_ref, value, indices))
      {
        for (std::size_t d = 0; d < dimension; ++d)
        {
          index_lists[d].view()(sparse_index) = indices[d];
        }
        values_view(sparse_index) = value;
        ++sparse_index;
      }
    }

    return sparse_tensor_t(tensor.sizes(), std::move(index_lists), std::move(values));
  }
  else
  {
    Tensor<dimension, host_space, data_t> host_tensor(tensor);
    auto host_sparse = convert_to_SparseTensor(host_tensor, std::forward<Filter>(filter));
    return SparseTensor<dimension, space, data_t>(host_sparse);
  }
}

} // namespace boba
