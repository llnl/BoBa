// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <random>
#include <sstream>
#include <stdexcept>

namespace boba
{

/**
 * \brief
 * Arbitrary dimensional Tensor. Dynamically allocated.
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
struct Tensor
{
  using data_t = _data_t;
  using data_reference = data_t&;
  using data_const_reference = data_t const&;
  using data_pointer = data_t*;
  using data_const_pointer = data_t const*;
  using real_data_t = real_type_t<data_t>;
  using index_array = ::boba::Array<index_t, dimension>;
  using indexer_type = Multiindexer<dimension>;
  using accessor = ::boba::DefaultAccessor<data_t>;
  using view_type = TensorView<accessor, dimension>;
  using const_accessor = ::boba::DefaultAccessor<data_t const>;
  using const_view_type = TensorView<const_accessor, dimension>;
  using atomic_accessor = ::boba::atomics::Accessor<space, data_t>;
  using atomic_view_type = TensorView<atomic_accessor, dimension>;

  static constexpr std::string_view object_type_name = "Tensor";

  Tensor() = default;

  // Copy constructor
  Tensor(Tensor const& rhs)
      : m_allocator(rhs.m_allocator),
        m_data(allocate(static_cast<size_t>(rhs.size()))),
        m_capacity(static_cast<size_t>(rhs.size())),
        m_sizes(rhs.sizes()),
        m_strides(rhs.strides()),
        m_name(rhs.name())
  {
    copy(rhs);
  }

  // Copy ctor from const view
  Tensor(const_view_type rhs)
      : m_data(allocate(static_cast<size_t>(rhs.size()))),
        m_capacity(static_cast<size_t>(rhs.size())),
        m_sizes(rhs.sizes()),
        m_strides(rhs.strides())
  {
    copy(rhs);
  }

  // Copy ctor from view
  Tensor(view_type rhs)
      : m_data(allocate(static_cast<size_t>(rhs.size()))),
        m_capacity(static_cast<size_t>(rhs.size())),
        m_sizes(rhs.sizes()),
        m_strides(rhs.strides())
  {
    copy(rhs);
  }

  // Copy constructor, different spaces
  template <execution_space rhs_space>
    requires(space != rhs_space)
  Tensor(Tensor<dimension, rhs_space, data_t> const& rhs)
      : m_data(allocate(static_cast<size_t>(rhs.size()))),
        m_capacity(static_cast<size_t>(rhs.size())),
        m_sizes(rhs.sizes()),
        m_strides(rhs.strides()),
        m_name(rhs.name())
  {
    copy(rhs);
  }

  // Copy constructor, different data_t
  template <typename rhsdata_t>
    requires(!std::same_as<data_t, rhsdata_t>)
  Tensor(Tensor<dimension, space, rhsdata_t> const& rhs)
      : m_allocator(rhs.m_allocator),
        m_data(allocate(static_cast<size_t>(rhs.size()))),
        m_capacity(static_cast<size_t>(rhs.size())),
        m_sizes(rhs.sizes()),
        m_strides(rhs.strides()),
        m_name(rhs.name())
  {
    copy(rhs);
  }

  // Move constructor
  Tensor(Tensor&& rhs) noexcept
      : m_allocator(rhs.m_allocator),
        m_data(std::exchange(rhs.m_data, nullptr)),
        m_capacity(std::exchange(rhs.m_capacity, 0)),
        m_sizes(std::exchange(rhs.m_sizes, ::boba::filled_array<dimension>(static_cast<index_t>(0)))),
        m_strides(std::exchange(rhs.m_strides, ::boba::filled_array<dimension>(static_cast<index_t>(0)))),
        m_name(std::exchange(rhs.m_name, "invalid"))
  {
  }

  // Move constructor, difference spaces
  template <execution_space rhs_space>
    requires(space != rhs_space)
  Tensor(Tensor<dimension, rhs_space, data_t>&& rhs)
  {
    this->resize(rhs.sizes());
    this->copy(rhs);
    rhs.resize(::boba::filled_array<dimension>(static_cast<index_t>(0)));
    m_name = std::exchange(rhs.m_name, "invalid");
  }

  // Copy assignment
  Tensor& operator=(Tensor const& rhs)
  {
    boba_always_assert(this != &rhs, "Tensor self-copy-assignment is not supported.");

    const auto rhs_capacity = static_cast<size_t>(rhs.size());
    if (m_allocator != rhs.m_allocator || rhs_capacity > this->m_capacity)
    {
      deallocate(this->m_data);
      m_allocator = rhs.m_allocator;
      this->m_data = allocate(rhs_capacity);
      this->m_capacity = rhs_capacity;
    }

    m_sizes = rhs.sizes();
    m_strides = rhs.strides();
    m_name = rhs.name();

    this->copy(rhs);

    return *this;
  }

  // Copy assignment, difference spaces
  template <execution_space rhs_space>
    requires(space != rhs_space)
  Tensor& operator=(Tensor<dimension, rhs_space, data_t> const& rhs)
  {
    checkpoint_objects();

    deallocate(this->m_data);
    const auto rhs_capacity = static_cast<size_t>(rhs.size());
    this->m_data = allocate(rhs_capacity);
    this->m_capacity = rhs_capacity;

    m_sizes = rhs.sizes();
    m_strides = rhs.strides();
    m_name = rhs.name();

    copy(rhs);

    return *this;
  }

  // Move assignment
  Tensor& operator=(Tensor&& rhs) noexcept
  {
    if (this == &rhs)
    {
      return *this;
    }

    deallocate(m_data);
    m_allocator = rhs.m_allocator;
    m_data = std::exchange(rhs.m_data, nullptr);
    m_capacity = std::exchange(rhs.m_capacity, 0);
    m_sizes = std::exchange(rhs.m_sizes, ::boba::filled_array<dimension>(static_cast<index_t>(0)));
    m_strides = std::exchange(rhs.m_strides, ::boba::filled_array<dimension>(static_cast<index_t>(0)));
    m_name = std::exchange(rhs.m_name, "invalid");
    return *this;
  }

  // Move assignment, difference spaces
  template <execution_space rhs_space>
    requires(space != rhs_space)
  Tensor& operator=(Tensor<dimension, rhs_space, data_t>&& rhs)
  {
    this->resize(rhs.sizes());
    this->copy(rhs);
    rhs.resize(::boba::filled_array<dimension>(static_cast<index_t>(0)));
    m_name = std::exchange(rhs.m_name, "invalid");
    return *this;
  }

  Tensor(index_array sizes)
      : m_data(allocate(static_cast<size_t>(::boba::product(sizes)))),
        m_capacity(static_cast<size_t>(::boba::product(sizes))),
        m_sizes(sizes),
        m_strides(indexer_type::precompute_strides(sizes))
  {
  }

  Tensor(index_array sizes, std::string_view name)
      : m_data(allocate(static_cast<size_t>(::boba::product(sizes)))),
        m_capacity(static_cast<size_t>(::boba::product(sizes))),
        m_sizes(sizes),
        m_strides(indexer_type::precompute_strides(sizes)),
        m_name(name)
  {
  }

  ~Tensor() noexcept
  {
    deallocate(m_data);
  }

  // ---------------------------------------------------------------------------
  // Member objects
  // ---------------------------------------------------------------------------

  detail::Allocator_t m_allocator = detail::get_default_allocator<space>();
  data_pointer m_data = nullptr;
  size_t m_capacity = 0;
  index_array m_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  index_array m_strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  std::string m_name = "Tensor";

  // ---------------------------------------------------------------------------
  // Section: Indexing
  // ---------------------------------------------------------------------------

  constexpr index_t index(index_array indices) const noexcept
  {
    return indexer_type::index(m_strides, indices);
  }

  constexpr index_array multiindex(index_t index) const noexcept
  {
    return indexer_type::multiindex(m_sizes, index);
  }

  void assert_indices(index_array indices) const noexcept
  {
    boba_assert_nonnegative(indices, "Negative index");
    boba_assert_lt(indices, m_sizes, "Out of bounds");
  }

  // ---------------------------------------------------------------------------
  // Section: Allocation
  // ---------------------------------------------------------------------------

  data_pointer allocate(size_t nelems)
  {
    return ::boba::detail::malloc<space, data_t>(nelems, m_allocator);
  }

  // deallocate (without changing sizes)
  void deallocate(data_pointer& data) noexcept
  {
    if (data != nullptr)
    {
      ::boba::detail::free<space>(data, m_allocator);
    }
  }

  void invalidating_reallocate(std::size_t new_capacity)
  {
    deallocate(this->m_data);
    this->m_data = allocate(new_capacity);
    this->m_capacity = new_capacity;
  }

  // ---------------------------------------------------------------------------
  // Section: Read/write
  // ---------------------------------------------------------------------------

  template <execution_space s = space>
    requires(s == host_space)
  data_reference operator()(index_array indices) noexcept
  {
    assert_indices(indices);
    return m_data[index(m_strides, indices)];
  }

  template <execution_space s = space>
    requires(s == host_space)
  data_const_reference operator()(index_array indices) const noexcept
  {
    assert_indices(indices);
    return m_data[index(m_strides, indices)];
  }

  // ---------------------------------------------------------------------------
  // Section: Modifiers
  // ---------------------------------------------------------------------------

  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  // ---------------------------------------------------------------------------
  // Section: Getters
  // ---------------------------------------------------------------------------

  static constexpr execution_space get_space() noexcept
  {
    return space;
  }

  static constexpr std::size_t get_dimension() noexcept
  {
    return dimension;
  }

  constexpr bool empty() const noexcept
  {
    return size() == 0;
  }

  constexpr index_t sizes(index_t d) const noexcept
  {
    return m_sizes[d];
  }

  constexpr index_array sizes() const noexcept
  {
    return m_sizes;
  }

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

  constexpr index_t strides(index_t d) const noexcept
  {
    return m_strides[d];
  }

  constexpr index_array strides() const noexcept
  {
    return m_strides;
  }

  constexpr size_t capacity() const noexcept
  {
    return m_capacity;
  }

  std::string const& name() const noexcept
  {
    return m_name;
  }

  constexpr data_pointer data() noexcept
  {
    return m_data;
  }

  constexpr data_const_pointer data() const noexcept
  {
    return m_data;
  }

  constexpr data_const_pointer const_data() const noexcept
  {
    return m_data;
  }

  constexpr view_type view() noexcept
  {
    return {data(), m_sizes, m_strides};
  }

  constexpr const_view_type view() const noexcept
  {
    return {const_data(), m_sizes, m_strides};
  }

  constexpr const_view_type const_view() const noexcept
  {
    return {const_data(), m_sizes, m_strides};
  }

  constexpr atomic_view_type atomic_view() noexcept
  {
    return {data(), m_sizes, m_strides};
  }

  // -------------------------------------------------------------------------------------
  // Section: Shaping
  // -------------------------------------------------------------------------------------

  void reshape(const Array<index_t, dimension> new_sizes) noexcept
  {
    // consistent with the MATLAB reshape command

    auto new_size = ::boba::product(new_sizes);
    auto new_strides = indexer_type::precompute_strides(new_sizes);

    boba_always_assert_equal(new_size, this->size(), "total size cannot change");

    m_sizes = new_sizes;
    m_strides = new_strides;
    m_capacity = static_cast<size_t>(new_size);
    checkpoint_objects();
  }

  template <size_t rhs_dimension>
  void reshape(const Tensor<rhs_dimension, space, data_t>& rhs) noexcept
  {
    // consistent with the MATLAB reshape command
    boba_always_assert_equal(rhs.size(), size(), "reshape sizes must match");
    boba::detail::memcpy<space, space>(this->data(), rhs.data(), static_cast<size_t>(rhs.size()));
  }

  // ---------------------------------------------------------------------------
  // Section: Indexing
  // ---------------------------------------------------------------------------

  // index = i0 + s0*(i1 + s1*(i2 + s2*(i3 + .... )))
  // index = i0 + (s0)*i1 + (s0*s1)*i2 + (s0*s1*s2)*i3 + ....
  //  strides[0] = 1
  //  strides[1] = s0
  //  strides[2] = s0*s1
  //  ...
  //  strides[i] = s0*...*s{i-1}

  static constexpr index_t index(index_array strides, index_array indices) noexcept
  {
    index_t index = 0;
    for (std::size_t d = 0; d < dimension; ++d)
    {
      index += strides[d] * indices[d];
    }
    return index;
  }

  // ---------------------------------------------------------------------------
  // Section: Copy
  // ---------------------------------------------------------------------------

  /*
   * \brief
   * Copies data from an equivalent Tensor
   */

  void copy(Tensor<dimension, space, data_t> const& rhs)
  {
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for copy.");
    if (rhs.size() == 0)
    {
      return;
    }
    BOBA_CALI_OBJECT_MARK

    boba::detail::memcpy<space, space>(this->data(), rhs.const_data(), static_cast<size_t>(rhs.size()));
  }

  /*
   * \brief
   * Copies data from an equivalent Tensor's views
   * WARNING - this does not check that the spaces are the same
   */

  void copy(const_view_type rhs)
  {
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for copy.");
    if (rhs.size() == 0)
    {
      return;
    }
    BOBA_CALI_OBJECT_MARK

    boba::detail::memcpy<space, space>(this->data(), rhs.const_data(), static_cast<size_t>(rhs.size()));
  }

  /*
   * Copies data from an equivalent Tensor's views
   * WARNING - this does not check that the spaces are the same
   */

  void copy(view_type rhs)
  {
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for copy.");
    if (rhs.size() == 0)
    {
      return;
    } // sizes are the same
    BOBA_CALI_OBJECT_MARK

    boba::detail::memcpy<space, space>(this->data(), rhs.const_data(), static_cast<size_t>(rhs.size()));
  }

  /*
   * Copies data from a Tensor of a different space
   */

  template <::boba::execution_space rhs_space>
    requires(rhs_space != space)
  void copy(Tensor<dimension, rhs_space, data_t> const& rhs)
  {
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for copy.");
    if (rhs.size() == 0)
    {
      return;
    } // sizes are the same
    BOBA_CALI_OBJECT_MARK

    boba::detail::memcpy<space, rhs_space>(this->data(), rhs.const_data(), static_cast<size_t>(rhs.size()));
  }

  /*
   * Copies data from a Tensor of differing data_t
   */
  template <typename rhsdata_t>
    requires(!std::same_as<data_t, rhsdata_t>)
  void copy(Tensor<dimension, space, rhsdata_t> const& rhs)
  {
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for copy.");
    if (rhs.size() == 0)
    {
      return;
    } // sizes are the same
    BOBA_CALI_OBJECT_MARK

    data_pointer this_data = this->data();
    auto rhs_data = rhs.const_data();

    ::boba::detail::loop<space>(0, size(), [=] __boba_host_device__(index_t i)
    {
      this_data[i] = static_cast<data_t>(rhs_data[i]);
    });
  }

  // ---------------------------------------------------------------------------
  // Section: Operations
  // ---------------------------------------------------------------------------

  /**
   * Returns the largest of the relative elementwise differences between this and rhs.
   * $$ value = \max\limits_i |a_i - r_i| / ( (|a_i| + |r_i|) / 2 )  $$
   */

  real_data_t max_pointwise_relative_difference(Tensor const& rhs) const
  {
    BOBA_CALI_OBJECT_MARK
    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for max_pointwise_relative_difference.");

    data_const_pointer this_data = this->const_data();
    data_const_pointer rhs_data = rhs.const_data();

    real_data_t pointwise_relative_difference = 0.0;

    ::boba::max_reduce<space>(pointwise_relative_difference, index_t(0), size(), [=] __boba_host_device__(index_t i, max_reducer_operator<real_data_t> & local_pointwise_relative_difference)
    {
      data_t x_rhs = rhs_data[i];
      data_t x_this = this_data[i];
      bool is_near_zero = is_tiny(abs(x_rhs)) or is_tiny(abs(x_this));
      if (not(is_near_zero))
      {
        real_data_t local_difference = abs(x_rhs - x_this);
        real_data_t local_average = (abs(x_rhs) + abs(x_this)) / static_cast<real_data_t>(2.0);

        real_data_t local_relative_difference = local_difference / local_average;
        local_pointwise_relative_difference.max(local_relative_difference);
      }
    });

    return pointwise_relative_difference;
  }

  template <::boba::execution_space rhs_space>
    requires(rhs_space != space)
  data_t max_pointwise_relative_difference(Tensor<dimension, rhs_space, data_t> const& rhs) const
  {
    BOBA_CALI_OBJECT_MARK

    // Copy to this space
    Tensor rhs_in_this_space{rhs};

    return this->max_pointwise_relative_difference(rhs_in_this_space);
  }

  // ---------------------------------------------------------------------------
  // Section: Sizing
  // ---------------------------------------------------------------------------

  /**
   * \brief
   * Resizes the Tensor and copies the old data into the new Tensor, using the corresponding multiindices
   * Note that this may truncate certain dimensions.  Some data will be uninitialized.
   */

  void resize(const index_array new_sizes) noexcept
  {
    if (this->sizes() == new_sizes)
    {
      return; // No change
    }
    if (this->size() == product(new_sizes))
    {
      this->reshape(new_sizes); // total size is contant, no reallocation required
      return;
    }

    // resizes the object
    // copies old data into new
    checkpoint_objects();
    auto old_sizes = this->sizes();
    auto old_data = this->data();
    auto old_view = this->const_view();

    checkpoint_objects();
    auto new_size = ::boba::product(new_sizes);
    auto new_capacity = static_cast<size_t>(new_size);
    auto new_strides = indexer_type::precompute_strides(new_sizes);
    auto new_data = allocate(new_capacity);
    view_type new_view{new_data, new_sizes, new_strides};

    checkpoint_objects();
    indexer_type intersection_view{min(old_sizes, new_sizes)};

    checkpoint_objects();
    ::boba::loop<space, 1>(intersection_view.size(),
                           [=] __boba_host_device__(index_t i)
    {
      auto idcs = intersection_view.multiindex(i);
      new_view(idcs) = old_view(idcs);
    });

    checkpoint_objects();
    deallocate(old_data);

    checkpoint_objects();
    this->m_sizes = new_sizes;
    this->m_strides = new_strides;
    this->m_capacity = new_capacity;
    this->m_data = new_data;
    checkpoint_objects();
  }

  void reset(index_array new_sizes) noexcept
  {
    checkpoint_objects();
    auto new_size = static_cast<size_t>(::boba::product(new_sizes));
    boba_always_assert_le(new_size, this->m_capacity, "insufficient capacity");

    this->m_sizes = new_sizes;
    this->m_strides = indexer_type::precompute_strides(this->m_sizes);
    checkpoint_objects();
  }

  void clear() noexcept
  {
    checkpoint_objects();
    this->m_sizes = ::boba::filled_array<dimension>(static_cast<index_t>(0));
    this->m_strides = ::boba::filled_array<dimension>(static_cast<index_t>(0));
  }

  // ---------------------------------------------------------------------------
  // Section: Read/write
  // ---------------------------------------------------------------------------

  void fill_with(data_t x)
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_fill_with");
    data_pointer this_data = this->data();
    ::boba::detail::loop<space>(0, size(), [=] __boba_host_device__(index_t i)
    {
      this_data[i] = x;
    });
    BOBA_CALI_OBJECT_END("tensor_fill_with");
  }

  template <typename _real_type>
    requires(!std::same_as<_real_type, data_t>)
  void fill_with(_real_type x)
  {
    fill_with(::boba::PotentiallyComplex<data_t>::value(x));
  }

  void fill_with_zeros()
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_set_to_zero");
    data_pointer this_data = this->data();
    if constexpr (space == host_space)
    {
      ::boba::detail::host_memset(this_data, 0, static_cast<size_t>(size()));
    }
    else if constexpr (space == execution_space::CUDA)
    {
      ::boba::detail::cuda_memset(this_data, 0, static_cast<size_t>(size()));
    }
    else if constexpr (space == execution_space::HIP)
    {
      ::boba::detail::hip_memset(this_data, 0, static_cast<size_t>(size()));
    }
    BOBA_CALI_OBJECT_END("tensor_set_to_zero");
  }

  /**
   * @brief Fills this Tensor with random values between lower_value and upper_value
   * For floats and doubles, the random value is in the range of (lower_value, upper_value).
   * For ints and size_t, the random value is in the range [0, upper_value - 1]
   * For complex floats and doubles, this is not defined
   * @param[in] lower_value
   * @param[in] upper_value
   */

  void fill_with_random(data_t lower_value, data_t upper_value)
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_fill_with_random");
    if constexpr (space == host_space)
    {
      if constexpr (std::is_same_v<data_t, size_t> or std::is_same_v<data_t, int>)
      {
        // Seed and construct random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        auto M = min(sizes());
        // Define distribution that generates integers in [lower_value, upper_value]
        std::uniform_int_distribution<data_t> distrib(lower_value, upper_value);
        // Write to data
        std::generate(data(), data() + size(), [&]
        {
          return distrib(gen);
        });
      }
      else if constexpr (std::is_same_v<data_t, double> or std::is_same_v<data_t, float>)
      {
        // Seed and construct random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        // Define distribution from (0.0, 1.0)
        std::uniform_real_distribution<data_t> distrib(lower_value, upper_value);
        // Write to data
        std::generate(data(), data() + size(), [&]
        {
          return distrib(gen);
        });
      }
      else
      {
        boba_error("Not defined for this data type");
      }
    }
    else
    {
      Tensor<dimension, host_space, data_t> host_tensor(this->sizes());
      host_tensor.fill_with_random(lower_value, upper_value);
      host_tensor.rename(this->name());
      *this = host_tensor;
    }
    BOBA_CALI_OBJECT_END("tensor_fill_with_random");
  }

  /**
   * @brief Fills this Tensor with random values
   * For floats and doubles, the random value is in the range of (0.0, 1.0).
   * For complex floats and doubles, the random value is a complex number such that |x| < 1.0
   * For ints and size_t, the random value is between [0, min(sizes()))
   */

  void fill_with_random()
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_fill_with_random");
    if constexpr (space == host_space)
    {
      if constexpr (std::is_same_v<data_t, size_t> or std::is_same_v<data_t, int>)
      {
        fill_with_random(0, min(sizes()));
      }
      else if constexpr (std::is_same_v<data_t, double> or std::is_same_v<data_t, float>)
      {
        fill_with_random(0, 1.0);
      }
      else
      {
        // Generate a set of random theta_k, R_k
        // Set k'th complex value to x_k = R_k*cos(theta_k) + i*R_k*sin(theta_k);
        boba::Tensor<1, host_space, real_data_t> theta({size()});
        boba::Tensor<1, host_space, real_data_t> radius({size()});
        theta.fill_with_random();
        radius.fill_with_random();

        auto this_view = view();
        auto theta_view = theta.const_view();
        auto radius_view = radius.const_view();

        boba::loop<host_space, 1>(size(), [=](index_t i)
        {
          auto th = theta_view(i);
          auto R = radius_view(i);
          auto real_part = R * boba::cos(th);
          auto imag_part = R * boba::sin(th);
          data_t random_value{real_part, imag_part};
          this_view(i) = random_value;
        });
      }
    }
    else
    {
      Tensor<dimension, host_space, data_t> host_tensor(this->sizes());
      host_tensor.fill_with_random();
      host_tensor.rename(this->name());
      *this = host_tensor;
    }
    BOBA_CALI_OBJECT_END("tensor_fill_with_random");
  }

  void multiply_scalar(data_t x)
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_multiply_scalar");
    data_pointer this_data = this->data();
    ::boba::loop<space, 1>(0, size(), [=] __boba_host_device__(index_t i)
    {
      this_data[i] *= x;
    });
    BOBA_CALI_OBJECT_END("tensor_multiply_scalar");
  }

  /**
   * Reset to the basis Tensor e_{i0, ... } which is zeros except for a single 1 at mid
   */

  void set_to_basis(index_array mid)
  {
    BOBA_CALI_OBJECT_BEGIN("tensor_set_to_basis");
    auto this_view = this->view();
    this->fill_with_zeros();
    auto id = this_view.index(mid);

    ::boba::detail::loop<space>(0, 1, [=] __boba_host_device__(index_t i)
    {
      this_view(id) = ::boba::PotentiallyComplex<data_t>::value(1.0);
      detail::ignore(i);
    });
    BOBA_CALI_OBJECT_END("tensor_set_to_basis");
  }

  // ---------------------------------------------------------------------------
  // Section: Printing
  // ---------------------------------------------------------------------------

  void print_dims() const
  {
    std::cout << name() << " is size a " << dimension << "-Tensor of size ";
    if (dimension > 0)
    {
      std::cout << this->m_sizes[0];
      for (size_t d = 1; d < dimension; d++)
      {
        std::cout << " x " << this->m_sizes[d];
      }
    }
    std::cout << std::endl;
  }

  void print(size_t indent = 0) const;

  // ---------------------------------------------------------------------------
  // Section: Operators
  // ---------------------------------------------------------------------------

  /**
   * \brief
   * this += rhs
   */

  Tensor& operator+=(Tensor const& rhs)
  {
    BOBA_CALI_OBJECT_MARK

    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for operator+=.");

    auto rhs_view = rhs.const_view();
    auto this_view = this->view();
    ::boba::loop<space, 1>(size(),
                           [=] __boba_host_device__(size_t i)
    {
      this_view(i) += rhs_view(i);
    });

    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  Tensor operator+(Tensor const& rhs)
  {
    BOBA_CALI_OBJECT_MARK
    Tensor output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  Tensor& operator-=(Tensor const& rhs)
  {
    BOBA_CALI_OBJECT_MARK

    if (not(sizes() == rhs.sizes()))
      throw std::length_error("Tensor shapes must match for operator-=.");

    auto rhs_view = rhs.const_view();
    auto this_view = this->view();
    ::boba::loop<space, 1>(size(),
                           [=] __boba_host_device__(size_t i)
    {
      this_view(i) -= rhs_view(i);
    });

    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  Tensor operator-(Tensor const& rhs)
  {
    BOBA_CALI_OBJECT_MARK
    Tensor output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * negation operator
   */

  Tensor operator-()
  {
    BOBA_CALI_MARK
    Tensor output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  Tensor& operator*=(data_t const scalar)
  {
    BOBA_CALI_OBJECT_MARK
    multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  Tensor operator*(data_t const scalar) const
  {
    BOBA_CALI_OBJECT_MARK
    Tensor output{*this};
    output *= scalar;
    return output;
  }

  /**
   * \brief
   * this /= scalar
   */

  Tensor& operator/=(data_t const scalar)
  {
    BOBA_CALI_OBJECT_MARK
    multiply_scalar(::boba::PotentiallyComplex<data_t>::value(1.0) / scalar);
    return *this;
  }

  /**
   * \brief
   * output = this / scalar
   */

  Tensor operator/(data_t const scalar) const
  {
    BOBA_CALI_OBJECT_MARK
    Tensor output{*this};
    output /= scalar;
    return output;
  }

  // ---------------------------------------------------------------------------
  // Section: Nonnegative / nonpositive part
  // ---------------------------------------------------------------------------

  template <typename output_type, typename unary_function_t>
  output_type unary_transform_copy(unary_function_t&& transform) const
  {
    BOBA_CALI_OBJECT_MARK

    output_type output(this->sizes());

    auto this_view = this->const_view();
    auto output_view = output.view();

    ::boba::detail::loop<space>(static_cast<index_t>(0), this->size(), [=] __boba_host_device__(index_t i)
    {
      output_view(i) = transform(this_view(i));
    });

    return output;
  }

  /**
   * Create a new Tensor where you apply the function f(x) = max(x, 0) to each entry
   */

  Tensor nonnegative_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonnegative_part is disabled for complex data type");
    return this->template unary_transform_copy<Tensor>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(x);
    });
  }

  /**
   * Create a new Tensor where you apply the function f(x) = max(-x, 0) to each entry.
   */

  Tensor nonpositive_part() const
  {
    static_assert(std::is_same_v<data_t, real_data_t>, "nonpositive_part is disabled for complex data type");
    return this->template unary_transform_copy<Tensor>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(-x);
    });
  }

  // ---------------------------------------------------------------------------
  // Section: Reductions
  // ---------------------------------------------------------------------------

  /**
   * Sum all elements of this Tensor.
   */

  data_t sum_reduce() const
  {
    return reductions::sum_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the maximum element of this Tensor.
   */

  data_t max_reduce() const
  {
    return reductions::max_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the maximum magnitude element of this Tensor.
   */

  real_data_t max_abs_reduce() const
  {
    return reductions::max_abs_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the maximum element of this Tensor and the corresponding long index.
   */

  std::pair<data_t, index_t> max_loc_reduce() const
  {
    return reductions::max_loc_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the maximum absolute value element of this Tensor and the corresponding long index.
   */

  std::pair<real_data_t, index_t> max_abs_loc_reduce() const
  {
    return reductions::max_abs_loc_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the minimum element of this Tensor.
   */

  data_t min_reduce() const
  {
    return reductions::min_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the minimum of the absolute values of the elements of this Tensor.
   */

  real_data_t min_abs_reduce() const
  {
    return reductions::min_abs_reduce<space>(const_data(), size());
  }

  /**
   * Calculates the minimum element of this Tensor and the corresponding long index.
   */

  std::pair<data_t, index_t> min_loc_reduce() const
  {
    return reductions::min_loc_reduce(min_loc_reduce(), size());
  }

  /**
   * Counts the number of nonzero elements based on a tolerance, like MATLAB's nnz
   */

  index_t number_nonzeros(const real_data_t tolerance = 1.0e-15) const
  {
    auto this_data = this->const_data();

    index_t value = 0;

    ::boba::sum_reduce<space>(value, index_t(0), size(), [=] __boba_host_device__(index_t i, sum_reducer_operator<index_t> & local_value)
    {
      if (this_data[i] > tolerance)
      {
        local_value += 1_z;
      }
    });

    return value;
  }

  /**
   * Computes the sparsity ratio consistent with matrix sparsity
   */

  real_data_t sparsity(const real_data_t tolerance = 1.0e-15) const
  {
    real_data_t full_size = this->get_full_size();
    real_data_t nnz = this->number_nonzeros(tolerance);
    return full_size / nnz;
  }

  /**
   * A no-op that lets you treat a Tensor like a tensor decomposition
   */

  void round(const real_data_t tolerance = 1.0e-15) const
  {
    ::boba::detail::ignore(tolerance);
  }

  /**
   * A no-op that lets you treat a Tensor like a tensor decomposition
   */

  void orthogonalize() const
  {
  }

  /**
   * A no-op that lets you treat a Tensor like a tensor decomposition
   */

  float compression_rate() const
  {
    return 1.0;
  }

  /**
   * An no-op that lets you treat a Tensor like a tensor decomposition
   */
  std::string ranks_string() const
  {
    return "n/a";
  }
};

/**
 * \brief
 * output = a + b
 */

template <size_t dimension, execution_space space, typename _data_t>
Tensor<dimension, space, _data_t> operator+(const Tensor<dimension, space, _data_t>& a, const Tensor<dimension, space, _data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Tensor<dimension, space, _data_t> output = a;
  output += b;
  return output;
}

/**
 * \brief
 * output = a - b
 */

template <size_t dimension, execution_space space, typename _data_t>
Tensor<dimension, space, _data_t> operator-(const Tensor<dimension, space, _data_t>& a, const Tensor<dimension, space, _data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Tensor<dimension, space, _data_t> output = a;
  output -= b;
  return output;
}

/**
 * \brief
 *  output = scalar * tensor_A
 */

template <size_t dimension, execution_space space, typename _data_t>
Tensor<dimension, space, _data_t> operator*(_data_t const scalar, Tensor<dimension, space, _data_t> tensor_A)
{
  BOBA_CALI_OBJECT_MARK
  return tensor_A * scalar;
}

} // namespace boba
