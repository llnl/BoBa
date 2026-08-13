// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tucker decomposition
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
struct Tucker
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "Tucker";
  ::boba::Array<size_t, dimension + 1> max_ranks = ::boba::filled_array<dimension + 1>(highest_value<size_t>());

  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);

  ::boba::Array<Matrix<space, data_t>, dimension> cores;
  Tensor<dimension, space, data_t> R_core;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize.
   * Instantiated object is the zero Tucker.
   */

  Tucker(::boba::Array<index_t, dimension> input_sizes)
  {
    this->resize(input_sizes);
  }

  // Default constructor
  Tucker() = default;

  // Copy constructor
  Tucker(Tucker const&) = default;

  // Move constructor
  Tucker(Tucker&&) = default;

  // Copy assignment
  Tucker& operator=(Tucker const&) = default;

  // Move assignment
  Tucker& operator=(Tucker&&) = default;

  /**
   * \brief copy constructor for a different execution space
   */

  template <execution_space rhs_space>
  Tucker(Tucker<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        max_ranks(rhs.max_ranks),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        cores(::boba::typed_array<Matrix<space, data_t>>(rhs.cores)),
        R_core(Tensor<dimension, space, data_t>(rhs.cores))
  {
  }

  template <execution_space rhs_space>
  Tucker& operator=(Tucker<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    max_ranks = rhs.max_ranks;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    cores = ::boba::typed_array<Matrix<space, data_t>>(rhs.cores);
    R_core = rhs.R_core;
    return *this;
  }

  ~Tucker() = default;

  /**
   * \brief
   * Resizes the Tucker decomposition and reinitializes every core to zero.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   */

  void resize(::boba::Array<index_t, dimension> input_sizes)
  {
    checkpoint();
    boba_assert(input_sizes > 0, "invalid size");
    for (size_t d = 0; d < dimension; d++)
    {
      cores[d].resize({input_sizes[d], 1});
      cores[d].fill_with_zeros();
    }
    R_core.resize(filled_array<dimension>(1_z));
    R_core.fill_with_zeros();
    checkpoint();
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
    R_core.rename(m_name + "_R_core");
  }

  // -------------------------------------------------------------------------------------
  // Printing
  // -------------------------------------------------------------------------------------

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    R_core.print();
    for (size_t d = 0; d < dimension; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      std::cout << write_indent(indent + 2) << cores[d].name() << std::endl
                << write_indent(indent + 3) << "ranks = " << get_ranks(d)
                << std::endl
                << write_indent(indent + 3) << "size = " << sizes(d) << std::endl;
      cores[d].print(indent + 1);
    }
  }

  /**
   * \brief The Tucker ranks (extents of the R_core).
   *
   * \return rank extents
   */
  [[nodiscard]]
  Array<size_t, dimension> ranks() const
  {
    return R_core.sizes();
  }

  /**
   * \brief Return a string describing the Tucker ranks in the format `(r0, r1, ...)`.
   *
   * \return ranks as a formatted string
   */
  [[nodiscard]]
  std::string ranks_string() const
  {
    return (" ( " + make_delimited_string(ranks()) + " ) ");
  }

  /**
   * \brief A string describing the size/length of each core.
   *
   * \return sizes as a formatted string
   */
  [[nodiscard]]
  std::string sizes_string() const
  {
    std::string string_of_sizes = "";
    string_of_sizes += " ( " + std::to_string(sizes(0));
    for (size_t d = 1; d < get_number_cores(); d++)
    {
      string_of_sizes += ", " + std::to_string(sizes(d));
    }
    string_of_sizes += " )";
    return string_of_sizes;
  }

  // -------------------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------------------

  /**
   * \brief Get the Tucker name.
   *
   * \return current name
   */

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Get the number of cores.
   *
   * \return number of cores
   */

  static constexpr size_t get_number_cores()
  {
    return get_dimension();
  }

  /**
   * \brief Get the dimensionality of the Tucker object.
   *
   * \return tensor dimension
   */

  static constexpr size_t get_dimension()
  {
    return dimension;
  }

  [[nodiscard]]
  size_t get_ranks(size_t i) const
  {
    boba_always_assert_lt(i, dimension, "invalid core number");
    return this->cores[i].sizes(1);
  }

  [[nodiscard]]
  size_t get_number_elements(size_t d) const
  {
    return this->cores[d].size();
  }

  [[nodiscard]]
  size_t get_number_elements() const
  {
    size_t sum_elements = R_core.size();
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += get_number_elements(d);
    }
    return sum_elements;
  }

  [[nodiscard]]
  double get_full_size() const
  {
    // returns a double since full_sizes can be huge for high dimensions
    double full_size = 1.0;
    for (size_t d = 0; d < dimension; d++)
    {
      full_size *= sizes(d);
    }
    return full_size;
  }

  /**
   * \brief Returns the ratio of the full size over the compressed size, truncated to two digits
   */

  [[nodiscard]]
  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  Array<size_t, dimension> sizes() const noexcept
  {
    ::boba::Array<size_t, dimension> this_sizes{0};
    for (size_t d = 0; d < dimension; d++)
    {
      this_sizes[d] = sizes(d);
    }
    return this_sizes;
  }

  size_t sizes(size_t d) const noexcept
  {
    return this->cores[d].sizes(0);
  }

  // -------------------------------------------------------------------------------------
  // Section: Views
  // -------------------------------------------------------------------------------------

  // TODO<feature>

  // -------------------------------------------------------------------------------------
  // Section: Write
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Sets this into a rank-one Tucker filled with x, equivalent to a tensor filled with x
   */

  void fill_with(data_t x)
  {
    BOBA_CALI_MARK

    R_core.resize(filled_array<dimension>(1_z));
    R_core.fill_with(x);

    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].resize({this->cores[d].sizes(0), 1});
      this->cores[d].fill_with(1.0);
    }
  }

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  void fill_with(_real_data_t x)
  {
    BOBA_CALI_MARK
    data_t complex_x{x, static_cast<_real_data_t>(0)};
    fill_with(complex_x);
  }

  /**
   * \brief
   * Sets this into a rank-one Tucker filled with zeros
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK
    this->fill_with(0.0);
  }

  // -------------------------------------------------------------------------------------
  // Section: Unroll
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Re-creates the full tensor corresponding to this decomposition
   */

  boba::Tensor<dimension, space, data_t> decompress() const
  {
    BOBA_CALI_MARK

    auto decompressed = R_core;

    for (size_t d = 0; d < dimension; d++)
    {
      auto this_decompressed = tensor_contraction<1>(decompressed, this->cores[d], {0}, {1});
      decompressed = this_decompressed;
    }

    return decompressed;
  }

// -------------------------------------------------------------------------------------
// Section: Partial Unroll
// -------------------------------------------------------------------------------------
#if 0

/**
 * \brief
 * TODO
 */

  // TODO<feature> partial_decompress
  Tucker<dimension-1, space> partial_decompress(
    const size_t dimension_to_decompress) const
  {
    // TODO
  }
#endif

  // -------------------------------------------------------------------------------------
  // Section: Insert dimension
  // -------------------------------------------------------------------------------------

#if 0
/**
 * \brief
 * TODO
 */

  Tucker<dimension+1, space> insert_dimension(
    const size_t dimension_to_insert,
    const size_t new_size = 1) const
  {
    BOBA_CALI_MARK
    // TODO
  }
#endif

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Addition of two tucker decompositions
   */

  void tucker_add(Tucker const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_assert_equal(input.sizes(d), this->sizes(d), "incompatible sizes");

      auto new_core = concatenate_columns(this->cores[d], input.cores[d]);
      this->cores[d] = new_core;
    }
    checkpoint();

    auto new_R_sizes = this->R_core.sizes() + input.R_core.sizes();

    auto new_R_core = R_core;
    new_R_core.resize(new_R_sizes);
    new_R_core.fill_with_zeros();

    auto input_R_core_view = input.R_core.const_view();
    auto R_core_view = R_core.const_view();

    auto new_R_core_view = new_R_core.view();

    checkpoint();
    ::boba::loop<space, 1>(new_R_core_view.size(),
                           [=] __boba_host_device__(size_t i)
    {
      auto new_mid = new_R_core_view.multiindex(i);
      auto R_sizes = R_core_view.sizes();
      if (new_mid < R_sizes)
      {
        new_R_core_view(i) = R_core_view(new_mid);
      }
      else if (new_mid >= R_sizes)
      {
        new_R_core_view(i) = input_R_core_view(new_mid - R_sizes);
      }
    });

    R_core = new_R_core;

    checkpoint();
  }

  /**
   * \brief
   * this += rhs
   */

  Tucker& operator+=(Tucker const& rhs)
  {
    BOBA_CALI_MARK
    this->tucker_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  Tucker operator+(Tucker const& rhs) const
  {
    BOBA_CALI_MARK
    Tucker output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  Tucker& operator-=(Tucker const& rhs)
  {
    BOBA_CALI_MARK
    Tucker rhs_minus{rhs};
    rhs_minus *= static_cast<real_data_t>(-1);
    this->tucker_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  Tucker operator-(Tucker const& rhs) const
  {
    BOBA_CALI_MARK
    Tucker output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * negation operator
   */

  Tucker operator-()
  {
    BOBA_CALI_MARK
    Tucker output{*this};
    output *= static_cast<real_data_t>(-1);
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  Tucker& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    multiply_scalar(scalar);
    return *this;
  }

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  Tucker& operator*=(_real_data_t const scalar)
  {
    BOBA_CALI_MARK
    data_t complex_scalar{scalar, static_cast<_real_data_t>(0)};
    *this *= complex_scalar;
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  Tucker operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    Tucker output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Compress an arbitrary tensor of size product(sizes) to this Tucker.
   */

  void compress(const boba::Tensor<dimension, space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();

    R_core = input;
    round_svd_step(true);
  }

  /**
   * \brief
   * Uses svd_tolerance, svd_max_kept_values to perform the tensor-round operation on this.
   * This process is broken up in the right-moving QR phase and the left-moving SVD phase.
   */

  void round()
  {
    BOBA_CALI_MARK
    checkpoint();

    round_qr_step();
    round_svd_step();
  }

  /**
   * \brief
   * QR phase of tensor round
   */

  void round_qr_step()
  {
    for (size_t dp1 = dimension; dp1 > 0; dp1--)
    {
      auto d = dp1 - 1;
      QR<space, data_t> qr;
      qr(cores[d]);
      cores[d] = qr.Q;
      auto contract_R = tensor_contraction<1>(qr.R, R_core, {1}, {dimension - 1});
      R_core = contract_R;
    }
  }

  void round_svd_step(bool intial_compression = false)
  {
    //
    // Step 1: determine ranks
    //
    Array<Matrix<space, data_t>, dimension> U_matrices;
    for (size_t d = 0; d < dimension; d++)
    {
      // Unfold over mode d
      auto unfold_right = unfold(R_core, d);

      // Compute SVD
      SVD<space, data_t> svd;
      svd.tolerance_relative = svd_tolerance_relative;
      svd.tolerance_absolute = svd_tolerance_absolute;

      svd(unfold_right);
      U_matrices[d] = svd.U;

      // Form new core
      if (intial_compression)
      {
        this->cores[d] = svd.U;
      }
      else
      {
        auto new_core = this->cores[d] * svd.U;
        this->cores[d] = new_core;
      }
    }

    //
    // Step 2: form compressed Tucker
    //
    for (size_t d = dimension; d > 0; d--)
    {
      auto temp = R_core;
      R_core = tensor_contraction<1>(U_matrices[d - 1], temp, {0}, {dimension - 1});
    }
  }

  /**
   * \brief
   * orthogonalize the cores by performing a sequence of QRs along the cores
   */

  void orthogonalize()
  {
    BOBA_CALI_MARK
    checkpoint();
    this->round_qr_step();
  }

  // -------------------------------------------------------------------------------------
  // Section: Multiply scalar
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Multiplies the Tucker by x, such that A = x*A
   */

  void multiply_scalar(data_t x)
  {
    checkpoint();
    this->cores[0] *= x;
  }

  /**
   * \brief
   * Multiplies each core of the Tucker by its own scalar, such that A = product(scalars)*A
   */

  void multiply_scalars(const ::boba::Array<data_t, dimension> scalars)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d] *= scalars[d];
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Reduction
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Inner product of this and input Tucker
   */

  data_t inner_product(const Tucker<1, space, data_t>& input) const
  {
    BOBA_CALI_MARK
    checkpoint();
    static_assert(dimension == 1, "Inconsistent dimensions between this and input.");
    auto R_core_expansion = tensor_product(this->R_core, input.R_core);
    auto mode_expansion = mode_n_tensor_product(this->cores[0], input.cores[0], 1);
    auto R_core_reduce = tensor_contraction<1>(mode_expansion, R_core_expansion, {1}, {0});
    auto result = R_core_reduce.sum_reduce();
    return result;
  }

  template <size_t rhs_dimension>
    requires(rhs_dimension != 1_z)
  data_t inner_product(const Tucker<rhs_dimension, space, data_t>& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    auto R_core_expansion = tensor_product(this->R_core, input.R_core);

    checkpoint();
    for (size_t dp1 = dimension; dp1 > 0; dp1--)
    {
      auto d = dp1 - 1;
      checkpoint();

      auto mode_ip = tensor_contraction<1>(
        {"r", "i"}, this->cores[d], {"r", "j"}, input.cores[d], {"i", "j"});

      checkpoint();
      // D[i R*j] = C[i, j] i < R
      auto mode_ip_flat = flatten(mode_ip);

      checkpoint();
      // D[i + Rj] * R_core_expansion(..., i + Rj , ...)
      auto R_core_reduce = tensor_contraction<1>(mode_ip_flat, R_core_expansion, {0}, {d});

      auto R_core_expansion_sizes = R_core_expansion.sizes();
      checkpoint();
      R_core_expansion_sizes[d] = 1;
      R_core_expansion.resize(R_core_expansion_sizes);
      R_core_expansion.reshape(R_core_reduce);
      checkpoint();
    }

    boba_always_assert_equal(R_core_expansion.size(), 1_z, "Should resolve to size 1.");

    auto inner_product = R_core_expansion.sum_reduce();
    return inner_product;
  }

  // -------------------------------------------------------------------------------------
  // Section: Apply arbitrary function
  // -------------------------------------------------------------------------------------
  /*
    template <typename function_type>
    tucker<dimension, space> apply_function_elementwise(
      function_type function,
      Array<size_t, dimension> n_chunks_per_dimension)
    {
      BOBA_CALI_MARK
      checkpoint();
    }
  */

  /*
    // TODO<feature> extract_subtrain -> extract_subtensor, implement
    Tucker<dimension, space> extract_subtrain(
        const ::boba::Array<size_t, dimension> initial_index,
        const ::boba::Array<size_t, dimension> end_index) {
      BOBA_CALI_MARK
      checkpoint();
    }
  */
};

template <size_t dimension, ::boba::execution_space space, typename _data_t>
Tucker<dimension, space, _data_t> operator*(_data_t scalar, Tucker<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  Tucker<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

} // namespace boba
