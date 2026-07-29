// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include "BOBA/boba.hpp"

#include <algorithm>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace boba
{

/**
 * \brief Vector implemented as a 1-tensor.
 *
 * Not to be confused with std::vector.
 */

// ------------------------------------
template <execution_space space, typename _data_t>
struct Vector : Tensor<1, space, _data_t>
{
  using base = Tensor<1, space, _data_t>;

  using base::const_view;
  using base::resize;
  using base::view;
  using typename base::data_t;
  using typename base::index_array;

  static constexpr std::string_view object_type_name = "Vector";

  /**
   * \brief Constructs a Vector with the requested size and optional name.
   * \param sizes Extent of the Vector.
   * \param name Object name used in diagnostics and I/O helpers.
   */
  explicit Vector(index_array sizes = ::boba::filled_array<1>(static_cast<index_t>(0)),
                  std::string_view name = object_type_name)
      : base(std::move(sizes), name)
  {
  }

  /**
   * \brief Constructs a host Vector from a std::vector.
   * \param vec Host data copied into this Vector.
   */
  template <execution_space s = space>
    requires(s == host_space)
  Vector(const std::vector<data_t>& vec)
  {
    this->resize({vec.size()});
    auto this_data = this->data();

    for (index_t i = 0; i < this->size(); i++)
    {
      this_data[i] = vec[i];
    }
  }

  /**
   * \brief copy constructor
   */
  Vector(Vector const&) = default;

  /**
   * \brief move constructor
   */
  Vector(Vector&&) = default;

  /**
   * \brief copy assignment operator
   */
  Vector& operator=(Vector const&) = default;

  /**
   * \brief move assignment operator
   */
  Vector& operator=(Vector&&) = default;

  /**
   * \brief copy constructor for a different execution space
   * \param rhs Vector to copy from.
   */
  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  Vector(Vector<rhs_space, data_t> const& rhs)
      : base(rhs)
  {
  }

  // Conversion constructor that works with boba::Tensor<1,...>
  // It creates a temporary Vector that will either be copied (lvalue)
  // or moved (rvalue) depending on the calling code
  /**
   * \brief Constructs a Vector from a rank-1 tensor.
   * \param input Rank-1 tensor used to initialize this Vector.
   */
  explicit Vector(base input)
      : base(std::move(input))
  {
  }

  // -------------------------------------------------------------------------------------
  // Operators
  // -------------------------------------------------------------------------------------

  /**
   * \brief copy assignment operator for a different execution space
   * \param rhs Vector to copy from.
   * \return This Vector after assignment.
   */
  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  Vector& operator=(Vector<rhs_space, data_t> const& rhs)
  {
    static_cast<base&>(*this) = static_cast<Tensor<1, rhs_space, data_t> const&>(rhs);
    return *this;
  }

  /**
   * \brief Adds another Vector elementwise.
   * \param rhs Vector to add.
   * \return This Vector after the addition.
   */
  Vector& operator+=(Vector const& rhs)
  {
    static_cast<base&>(*this) += static_cast<base const&>(rhs);
    return *this;
  }

  /**
   * \brief Returns the elementwise sum of two vectors.
   * \param rhs Right-hand operand.
   * \return The elementwise sum.
   */
  [[nodiscard]]
  Vector operator+(Vector const& rhs)
  {
    BOBA_CALI_OBJECT_MARK
    Vector output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief Subtracts another Vector elementwise.
   * \param rhs Vector to subtract.
   * \return This Vector after the subtraction.
   */
  Vector& operator-=(Vector const& rhs)
  {
    static_cast<base&>(*this) -= static_cast<base const&>(rhs);
    return *this;
  }

  /**
   * \brief Returns the elementwise difference of two vectors.
   * \param rhs Right-hand operand.
   * \return The elementwise difference.
   */
  [[nodiscard]]
  Vector operator-(Vector const& rhs)
  {
    Vector output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief Returns the additive inverse of this Vector.
   * \return A Vector with every element negated.
   */
  [[nodiscard]]
  Vector operator-()
  {
    Vector output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief Multiplies every entry by a scalar.
   * \param scalar Scalar multiplier.
   * \return This Vector after scaling.
   */
  Vector& operator*=(data_t const scalar)
  {
    this->multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief Returns a scaled copy of this Vector.
   * \param scalar Scalar multiplier.
   * \return The scaled Vector.
   */
  [[nodiscard]]
  Vector operator*(data_t const scalar) const
  {
    Vector output{*this};
    output *= scalar;
    return output;
  }

  /**
   * \brief Divides every entry by a scalar.
   * \param scalar Scalar divisor.
   * \return This Vector after scaling.
   */
  Vector& operator/=(data_t const scalar)
  {
    static_cast<base&>(*this) /= scalar;
    return *this;
  }

  /**
   * \brief Returns a scaled copy of this Vector.
   * \param scalar Scalar divisor.
   * \return The scaled Vector.
   */
  [[nodiscard]]
  Vector operator/(data_t const scalar) const
  {
    Vector output{*this};
    output /= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Norms
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes the Vector 1-norm, \f$\|x\|_1\f$.
   * \return Sum of absolute values of all Vector entries.
   */
  [[nodiscard]]
  data_t vector_one_norm() const
  {
    BOBA_CALI_OBJECT_MARK

    if (this->size() == 0)
    {
      return data_t{0};
    }

    auto this_data = this->const_data();

    data_t value = 0.0;

    ::boba::sum_reduce<space>(value, index_t(0), this->size(), [=] __boba_host_device__(index_t i, sum_reducer_operator<data_t> & local_value)
    {
      local_value += ::boba::abs(this_data[i]);
    });

    return value;
  }

  /**
   * \brief Specialization of resize for CUDA-friendly scalar syntax.
   * \param new_sizes New Vector length.
   *
   * See https://lc.llnl.gov/gitlab/boba/boba/-/issues/202
   */
  void resize(const index_t new_sizes) noexcept
  {
    index_array arr{new_sizes};
    base::resize(arr);
  }

  /**
   * \brief Returns the elementwise positive part of this Vector.
   * \return Vector with entries replaced by \f$\max(x, 0)\f$.
   */
  [[nodiscard]]
  Vector nonnegative_part() const
  {
    static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "nonnegative_part is disabled for complex data type");
    return this->template unary_transform_copy<Vector>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(x);
    });
  }

  /**
   * \brief Returns the elementwise negative part magnitude of this Vector.
   * \return Vector with entries replaced by \f$\max(-x, 0)\f$.
   */
  [[nodiscard]]
  Vector nonpositive_part() const
  {
    static_assert(std::is_same_v<data_t, real_type_t<data_t>>, "nonpositive_part is disabled for complex data type");
    return this->template unary_transform_copy<Vector>(
      [] __boba_host_device__(data_t x)
    {
      return ::boba::positive_part(-x);
    });
  }

  /**
   * \brief Copies a half-open interval of entries into a new Vector.
   * \param rows Half-open interval `[rows[0], rows[1])` to copy.
   * \return Vector containing the selected entries.
   */
  [[nodiscard]]
  Vector get_subvector(Array<index_t, 2> rows) const
  {
    BOBA_CALI_OBJECT_MARK
    boba_assert_lt(rows[0], rows[1], "Incorrect ordering.");

    Vector output({rows[1] - rows[0]});

    auto output_view = output.view();
    auto this_view = this->const_view();

    ::boba::loop<space, 1>(output_view.size(),
                           [=] __boba_host_device__(index_t i)
    {
      output_view(i) = this_view({rows[0] + i});
    });
    return output;
  }

  /**
   * \brief Replaces a half-open interval of entries with another Vector.
   * \param rows Half-open interval `[rows[0], rows[1])` to overwrite.
   * \param replacement Vector whose entries are copied into the interval.
   */
  void replace_subvector(Array<index_t, 2> rows, const Vector& replacement)
  {
    BOBA_CALI_OBJECT_MARK
    boba_assert_equal(rows[1] - rows[0], replacement.size(), "Incorrect ordering.");

    auto replacement_view = replacement.const_view();
    auto this_view = this->view();

    ::boba::loop<space, 1>(replacement_view.size(),
                           [=] __boba_host_device__(index_t i)
    {
      this_view({rows[0] + i}) = replacement_view(i);
    });
  }
};

// -------------------------------------------------------------------------------------
// Section: Helper functions
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Create a Vector filled with a specified value
 */

template <execution_space space, typename data_t>
Vector<space, data_t> filled_vector(index_t size, data_t value)
{
  ::boba::Vector<space, data_t> filled({size});
  filled.fill_with(value);
  return filled;
}

/**
 * \brief
 * output = a + b
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator+(const Vector<space, data_t>& a, const Vector<space, data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Vector<space, data_t> output = a;
  output += b;
  return output;
}

/**
 * \brief
 * output = a - b
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator-(const Vector<space, data_t>& a, const Vector<space, data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Vector<space, data_t> output = a;
  output -= b;
  return output;
}

/**
 * \brief
 * output = a * b
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(const Vector<space, data_t>& a, const Vector<space, data_t>& b)
{
  BOBA_CALI_OBJECT_MARK
  Vector<space, data_t> output = a;
  output *= b;
  return output;
}

/**
 * \brief
 *  output = scalar * vector_A
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(data_t const scalar, Vector<space, data_t> vector_A)
{
  BOBA_CALI_OBJECT_MARK
  return vector_A * scalar;
}

/**
 * \brief
 *  Sorts the Vector according to std::sort
 */

template <execution_space space, typename data_t>
  requires(space == host_space)
void sort(Vector<space, data_t>& input_vector)
{
  if constexpr (space == host_space)
  {
    std::sort(input_vector.data(), input_vector.data() + input_vector.size());
  }
  else
  {
    Vector<host_space, data_t> host_vector{input_vector};
    sort(host_vector);
    input_vector = host_vector;
  }
}

} // namespace boba
