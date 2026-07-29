// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

namespace boba
{

/**
 *  Structure for supporting block systems (e.g. multiple fields)
 */

template <typename vector_t>
struct BlockVector
{
  using data_t = typename vector_t::data_t;
  using real_data_t = typename vector_t::real_data_t;

  size_t block_size{0};
  std::vector<vector_t> vector_blocks;
  std::string m_name = "BlockVector";

  /// \brief default constructor
  BlockVector() = default;

  /**
   * \brief Construct a block vector with a fixed number of blocks.
   *
   * \param[in] _size number of blocks
   */
  BlockVector(size_t _size)
  {
    block_size = _size;
    vector_blocks.resize(_size);
  }

  /**
   * \brief Access an element of the vector.
   *
   * \param[in] id block index
   * \return reference to the requested block
   */
  vector_t& operator()(size_t id)
  {
    return vector_blocks.at(id);
  }

  /**
   * \brief Access an element of the vector, const.
   *
   * \param[in] id block index
   * \return const reference to the requested block
   */
  vector_t const& operator()(size_t id) const
  {
    return vector_blocks.at(id);
  }

  BlockVector operator-(const BlockVector& rhs) const
  {
    BlockVector output(rhs.block_size);
    for (size_t element = 0; element < output.block_size; element++)
    {
      output(element) = vector_blocks.at(element);
      output(element) -= rhs(element);
    }
    return output;
  }

  BlockVector operator+(const BlockVector& rhs) const
  {
    BlockVector output(rhs.block_size);
    for (size_t element = 0; element < output.block_size; element++)
    {
      output(element) = vector_blocks.at(element);
      output(element) += rhs(element);
    }
    return output;
  }

  BlockVector& operator-=(const BlockVector& rhs)
  {
    for (size_t element = 0; element < rhs.block_size; element++)
    {
      this->operator()(element) -= rhs(element);
    }
    return *this;
  }

  BlockVector& operator+=(const BlockVector& rhs)
  {
    for (size_t element = 0; element < block_size; element++)
    {
      this->operator()(element) += rhs(element);
    }
    return *this;
  }

  BlockVector& operator*=(data_t const x)
  {
    BOBA_CALI_MARK
    for (size_t element = 0; element < block_size; element++)
    {
      this->vector_blocks.at(element) *= x;
    }
    return *this;
  }

  BlockVector operator*(data_t const x) const
  {
    BOBA_CALI_MARK
    BlockVector output(block_size);
    for (size_t element = 0; element < block_size; element++)
    {
      output(element) = this->vector_blocks.at(element);
    }
    output *= x;
    return output;
  }

  /**
   * \brief Get the block vector name.
   *
   * \return current name
   */
  std::string const& name() const noexcept
  {
    return m_name;
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  /**
   * \brief Get the total compressed element count across all blocks.
   *
   * \return total number of stored elements
   */
  [[nodiscard]]
  size_t get_number_elements() const
  {
    size_t count = 0;
    for (size_t element = 0; element < block_size; element++)
    {
      count += vector_blocks.at(element).get_number_elements();
    }
    return count;
  }

  /**
   * \brief Get the total full size across all blocks.
   *
   * \return total full size
   */
  [[nodiscard]]
  size_t get_full_size() const
  {
    size_t count = 0;
    for (size_t element = 0; element < block_size; element++)
    {
      count += vector_blocks.at(element).get_full_size();
    }
    return count;
  }

  /**
   * \brief Return the ratio of full size to compressed size, truncated to two digits.
   *
   * \return truncated compression rate
   */
  [[nodiscard]]
  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  void round()
  {
    for (size_t element = 0; element < block_size; element++)
    {
      vector_blocks.at(element).round();
    }
  }

  void orthogonalize()
  {
    for (size_t element = 0; element < block_size; element++)
    {
      vector_blocks.at(element).orthogonalize();
    }
  }

  /**
   *  Consistent with tensor fill_with
   */
  void fill_with(data_t x)
  {
    for (size_t element = 0; element < block_size; element++)
    {
      vector_blocks.at(element).fill_with(x);
    }
  }

  /**
   *  Consistent with tensor fill_with_zeros
   */
  void fill_with_zeros()
  {
    for (size_t element = 0; element < block_size; element++)
    {
      vector_blocks.at(element).fill_with_zeros();
    }
  }

  void savetxt(const std::string& name, std::string ext) const
  {
    for (size_t element = 0; element < block_size; element++)
    {
      boba::savetxt(vector_blocks.at(element), name + "_" + std::to_string(element) + ext);
    }
  }
};

template <typename vector_t>
BlockVector<vector_t> operator*(const double x, const BlockVector<vector_t>& vec)
{
  BOBA_CALI_MARK
  return vec * x;
}

template <typename vector_t>
typename vector_t::data_t inner_product(
  const BlockVector<vector_t>& vec_a,
  const BlockVector<vector_t>& vec_b)
{
  BOBA_CALI_MARK
  checkpoint();
  typename vector_t::data_t ip = 0.0;
  for (size_t element = 0; element < vec_a.block_size; element++)
  {
    ip += inner_product(vec_a(element), vec_b(element));
  }
  checkpoint();
  return ip;
}

template <typename vector_t, typename data_t>
BlockVector<vector_t> add_reduce(
  const std::vector<BlockVector<vector_t>>& sequence,
  Vector<execution_space::CPU, data_t>& coeff)
{
  BOBA_CALI_MARK
  checkpoint();
  BlockVector<vector_t> output(sequence.at(0).block_size);
  for (size_t element = 0; element < output.block_size; element++)
  {
    std::vector<vector_t> element_sequence;
    for (auto& item : sequence)
    {
      element_sequence.push_back(item(element));
    }
    output(element) = add_reduce(element_sequence, coeff);
  }
  checkpoint();
  return output;
}

template <typename vector_a_t>
typename vector_a_t::data_t norm_frobenius(
  const BlockVector<vector_a_t>& block_vec)
{
  BOBA_CALI_MARK
  typename vector_a_t::data_t output = 0.0;
  for (size_t element = 0; element < block_vec.block_size; element++)
  {
    auto norm_segment = ::boba::norm_frobenius(block_vec.vector_blocks.at(element));
    output += ::boba::pow(norm_segment, 2.0);
  }
  return ::boba::sqrt(output);
}

template <typename vector_a_t, typename vector_b_t>
typename vector_a_t::data_t norm_difference_frobenius(
  const BlockVector<vector_a_t>& block_vec_a,
  const BlockVector<vector_b_t>& block_vec_b)
{
  BOBA_CALI_MARK
  checkpoint();
  typename vector_a_t::data_t result = 0.0;
  static_assert(std::is_same_v<typename vector_a_t::data_t, typename vector_b_t::data_t>);

  for (size_t element = 0; element < block_vec_a.block_size; element++)
  {
    auto element_norm = norm_difference_frobenius(block_vec_a(element), block_vec_b(element));
    result += ::boba::pow(element_norm, 2.0);
  }
  checkpoint();
  return boba::sqrt(result);
}

} // namespace boba
