// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Quantized Tensor train. This is used to represent a vector of length base^exponent.
 * The QTT is made up of exponent number cores each of size base.
\verbatim
    (1, r_0)     (r_0, r_1)    (r_1, r_2)    (r_d-2, r_d-1)  (r_d-1, 1)
 ___  ____     __   ____      ___  ____         ___  ____       ____
 \    |  |     \    |  |      \    |  |         \    |  |       |  |
  >   |  | (x)  >   |  | (x)   >   |  | (x) ...  >   |  |  (x)  |  |
 /__  |__|     /__  |__|      /__  |__|         /__  |__|       |__|
 r_0  base     r_1  base      r2   base       r_d-1  base       base
\endverbatim
 */

template <::boba::execution_space space, typename _data_t>
struct QuantizedTensorTrain
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "QuantizedTensorTrain";
  size_t base = 2;
  size_t exponent = 1;

  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);

  size_t svd_max_kept_values = highest_value<size_t>();

  std::vector<Tensor<3, space, data_t>> cores;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize. Instantiated object is the zero tensor train.
   */

  QuantizedTensorTrain(size_t input_base, size_t input_exponent)
  {
    this->resize(input_base, input_exponent);
  }

  // Default constructor
  QuantizedTensorTrain() = default;

  // Copy constructor
  QuantizedTensorTrain(QuantizedTensorTrain const&) = default;

  // Move constructor
  QuantizedTensorTrain(QuantizedTensorTrain&&) = default;

  // Copy assignment
  QuantizedTensorTrain& operator=(QuantizedTensorTrain const&) = default;

  // Move assignment
  QuantizedTensorTrain& operator=(QuantizedTensorTrain&&) = default;

  /**
   * \brief copy constructor for a different execution space
   */

  template <execution_space rhs_space>
  QuantizedTensorTrain(QuantizedTensorTrain<rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        base(rhs.base),
        exponent(rhs.exponent),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        svd_max_kept_values(rhs.svd_max_kept_values),
        cores(rhs.cores)
  {
  }

  /**
   * \brief copy assignment operator for a different execution space
   */

  template <execution_space rhs_space>
  QuantizedTensorTrain& operator=(QuantizedTensorTrain<rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    base = rhs.base;
    exponent = rhs.exponent;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    svd_max_kept_values = rhs.svd_max_kept_values;
    for (size_t d = 0; d < rhs.cores.size(); d++)
    {
      cores[d] = rhs.cores[d];
    }
    return *this;
  }

  ~QuantizedTensorTrain() = default;

  /**
   * \brief Resize the quantized tensor train and reinitialize every core to zero.
   *
   * Unlike `Tensor::resize`, this does not preserve overlapping entries.
   *
   * \param[in] _base logical base size
   * \param[in] _exponent number of cores
   */

  void resize(size_t _base, size_t _exponent)
  {
    checkpoint();
    boba_assert_positive(base, "invalid base");
    boba_assert_positive(exponent, "invalid exponent");
    base = _base;
    exponent = _exponent;
    cores.resize(exponent);
    for (auto& core : cores)
    {
      core.resize({1_z, base, 1_z});
      core.fill_with_zeros();
    }
    checkpoint();
  }
  /**
   * \brief Rename the quantized tensor train and its cores.
   *
   * \param[in] new_name new object name
   */

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < exponent; d++)
    {
      this->cores.at(d).rename(m_name + "_core_" + std::to_string(d));
    }
  }

  // -------------------------------------------------------------------------------------
  // Printing
  // -------------------------------------------------------------------------------------

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    for (size_t d = 0; d < exponent; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      std::cout << write_indent(indent + 2) << cores[d].name() << std::endl
                << write_indent(indent + 3) << "ranks_left = " << get_ranks_left(d)
                << std::endl
                << write_indent(indent + 3) << "ranks_right = " << get_ranks_right(d)
                << std::endl
                << write_indent(indent + 3) << "size = " << sizes(d) << std::endl;
      cores[d].print(indent + 1);
    }
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
    string_of_sizes += " ( " + std::to_string(get_core_size(0));
    for (size_t d = 1; d < get_number_cores(); d++)
    {
      string_of_sizes += ", " + std::to_string(get_core_size(d));
    }
    string_of_sizes += " )";
    return string_of_sizes;
  }

  /**
   * \brief A std::vector describing the tensor train ranks.
   *
   * \return rank extents for each core boundary
   */
  [[nodiscard]]
  std::vector<size_t> ranks() const
  {
    std::vector<size_t> ranks;
    auto dimension = get_number_cores();
    ranks.resize(dimension + 1);
    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] = get_ranks_left(d);
    }
    ranks[dimension] = 1;
    return ranks;
  }

  /**
   * \brief A string describing the tensor train ranks.
   *
   * \return ranks as a formatted string
   */
  [[nodiscard]]
  std::string ranks_string() const
  {
    return (" ( " + make_delimited_string(ranks()) + " ) ");
  }

  /**
   * \brief A std::vector describing the size/length of each core.
   *
   * \return core sizes
   */
  [[nodiscard]]
  std::vector<size_t> sizes() const
  {
    std::vector<size_t> sizes;
    auto dimension = get_number_cores();
    sizes.resize(dimension);
    for (size_t d = 0; d < get_number_cores(); d++)
    {
      sizes[d] = get_core_size(d);
    }
    return sizes;
  }

  /**
   * \brief
   * The size of the d'th core, which is always base
   */

  index_t sizes(index_t d) const
  {
    detail::ignore(d);
    return base;
  }

  // -------------------------------------------------------------------------------------
  // Diagnostics
  // -------------------------------------------------------------------------------------

  /**
   * \brief Returns number of cores
   */

  size_t get_number_cores() const
  {
    return exponent;
  }

  /**
   * \brief
   */

  size_t get_core_size(size_t i) const
  {
    ::boba::detail::ignore(i);
    return base;
  }

  /**
   * \brief
   */

  size_t get_ranks_left(size_t i) const
  {
    return this->cores.at(i).sizes(0);
  }

  /**
   * \brief
   */

  size_t get_ranks_right(size_t i) const
  {
    return this->cores.at(i).sizes(2);
  }

  /**
   * \brief
   */

  size_t get_number_elements() const
  {
    size_t sum_elements = 0;
    for (size_t d = 0; d < exponent; d++)
    {
      sum_elements += this->cores[d].get_number_elements();
    }
    return sum_elements;
  }

  /**
   * \brief
   * Returns a double since full_sizes can be huge
   */

  double get_full_size() const
  {
    return ::boba::pow(base, exponent);
  }
  /**
   * \brief Returns name of tt
   */

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \brief Returns the ratio of the full size over the compressed size, truncated to two digits
   */

  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  // -------------------------------------------------------------------------------------
  // Section: Write
  // -------------------------------------------------------------------------------------
  /**
   * \brief
   * Sets this into a rank-one train filled with x, equivalent to a tensor filled with x
   */

  void fill_with(data_t x)
  {
    BOBA_CALI_MARK
    this->cores.resize(exponent);
    this->cores.at(0).resize({1, base, 1});
    set_to_rank_one_scalar_core(this->cores.at(0), x);
    for (size_t d = 1; d < exponent; d++)
    {
      this->cores.at(d).resize({1, base, 1});
      set_to_rank_one_scalar_core(this->cores.at(d), 1.0);
    }
  }

  /**
   * \brief
   * Sets this into a rank-one train filled with zeros
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK
    this->cores.resize(exponent);
    for (size_t d = 0; d < exponent; d++)
    {
      this->cores.at(d).resize({1, base, 1});
      set_to_rank_one_scalar_core(this->cores.at(d), data_t(0));
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Unroll
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Unroll qtt into a vector.
   */

  Vector<space, data_t> decompress() const
  {
    BOBA_CALI_MARK
    checkpoint();

    Tensor<3, space, data_t> new_core({1, 1, 1});
    new_core.rename("new_core");
    new_core.fill_with(1.0);
    checkpoint();

    // TODO<optimization> judiciously choose the order

    for (size_t dpp = exponent; dpp > 0; dpp--)
    {
      auto d = dpp - 1;
      Array<size_t, 2> start_indices{0, 0};
      Array<size_t, 2> end_indices{base, new_core.sizes(1)};
      auto temp_core = partial_decompress_core(start_indices, end_indices, this->cores.at(d), new_core);
      new_core = temp_core;
      if (d == 0)
      {
        break;
      }
    }

    //
    // Recast tensor core into tensor
    //
    Vector<space, data_t> output({new_core.size()});
    output.reshape(new_core);

    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tensor train addition of this and subtrain
   */

  void add_subtrain(QuantizedTensorTrain const& subtrain)
  {
    BOBA_CALI_MARK
    boba_assert_equal(base, subtrain.base, "incompatible base");
    boba_assert_equal(exponent, subtrain.exponent, "incompatible exponent");
    checkpoint();

    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {
      add_subcore(this->cores.at(d), subtrain.cores.at(d), d, exponent, 0);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Operators
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * this += rhs
   */

  QuantizedTensorTrain& operator+=(QuantizedTensorTrain const& rhs)
  {
    BOBA_CALI_MARK
    this->add_subtrain(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  QuantizedTensorTrain operator+(QuantizedTensorTrain const& rhs) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrain output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  QuantizedTensorTrain& operator-=(QuantizedTensorTrain const& rhs)
  {
    BOBA_CALI_MARK
    QuantizedTensorTrain rhs_minus{rhs};
    rhs_minus *= -1.0;
    this->add_subtrain(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  QuantizedTensorTrain operator-(QuantizedTensorTrain const& rhs) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrain output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * negation operator
   */

  QuantizedTensorTrain operator-()
  {
    BOBA_CALI_MARK
    QuantizedTensorTrain output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  QuantizedTensorTrain& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  QuantizedTensorTrain operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    QuantizedTensorTrain output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Compress an arbitrary tensor of size product(sizes) to this tensor train.
   */

  void compress(const boba::Vector<space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    boba_always_assert_equal(input.size(), pow(base, exponent), "QTT not correctly set up to compress this tensor");
    this->fill_with_zeros();

    if (exponent == 1_z)
    {
      this->cores.at(0).resize({1_z, base, 1_z});
      this->cores.at(0).reshape(input);
      return;
    }

    std::vector<boba::SVD<space, data_t>> svd;
    svd.resize(exponent - 1);

    for (size_t d = 0; d < svd.size(); d++)
    {
      svd[d].tolerance_relative = svd_tolerance_relative;
      svd[d].tolerance_absolute = svd_tolerance_absolute;
      svd[d].max_kept_singular_values = svd_max_kept_values;
    }

    size_t sizes_product = input.size();
    checkpoint();
    size_t rows = 1;
    size_t cols = sizes_product;
    checkpoint();
    boba::Matrix<space, data_t> temp_fold({rows, cols});
    checkpoint();
    temp_fold.rename("folding_matrix");
    checkpoint();
    temp_fold.reshape(input);
    checkpoint();
    size_t ranks = 1;
    for (size_t d = 0; d < exponent - 1; d++)
    {
      checkpoint();
      size_t this_size = base;
      rows = this_size * ranks;
      cols = temp_fold.size() / rows;
      checkpoint();
      temp_fold.reshape({rows, cols});
      checkpoint();
      svd[d](temp_fold);
      checkpoint();
      ranks = svd[d].significant_singular_values;
      if (ranks == 0)
      {
        this->fill_with_zeros();
        return;
      }
      checkpoint();
      apply_as_diagonal_right_in_place(svd[d].S, svd[d].U);
      checkpoint();
      this->cores.at(d) = write_to_core_from_left_fold(svd[d].U, get_core_size(d));
      checkpoint();
      temp_fold = svd[d].V.transpose();
      checkpoint();
    }
    checkpoint();
    this->cores.at(exponent - 1) = write_to_core_from_right_fold(temp_fold, sizes(exponent - 1));
    checkpoint();
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
    if (exponent == 1)
    {
      bool left_rank_one = this->get_ranks_left(0) == 1;
      bool right_rank_one = this->get_ranks_right(0) == 1;
      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core = tensor_reduction<2>({"l", "i", "r"}, this->cores[0], {"i"});
      this->cores[0] = reshape<3>(new_core, {1_z, sizes(0), 1_z});
    }
    boba::Matrix<space, data_t> unfold_right;
    std::vector<::boba::QR<space, data_t>> qr;
    qr.resize(exponent);
    round_qr_step(qr, unfold_right);
    round_svd_step(qr, unfold_right);
  }

  /**
   * \brief
   * QR phase of tensor round
   */

  void round_qr_step(
    std::vector<QR<space, data_t>>& qr,
    boba::Matrix<space, data_t>& unfold_right)
  {
    BOBA_CALI_MARK
    checkpoint();
    // TODO<optimization> the cores are later overwritten, so they could be cleared away
    auto unfold_left = compute_unfold_left(this->cores[0]);
    set_to_rank_one_scalar_core(this->cores[0], data_t(0.0));
    for (size_t d = 0; d < exponent - 1; d++)
    {
      checkpoint();
      unfold_right = compute_unfold_right(this->cores[d + 1]);
      set_to_rank_one_scalar_core(this->cores[d + 1], data_t(0.0));
      checkpoint();
      qr[d](unfold_left);
      checkpoint();
      qr[d].apply_R_left_in_place(unfold_right);
      qr[d].R.resize({1, 1});
      checkpoint();
      if (d + 2 < exponent)
      {
        unfold_left = compute_unfold_left_from_unfold_right(unfold_right, sizes(d + 1));
      }
      checkpoint();
    }
    checkpoint();
  }

  void round_svd_step(
    std::vector<QR<space, data_t>>& qr,
    boba::Matrix<space, data_t>& unfold_right)
  {
    BOBA_CALI_MARK

    std::vector<boba::SVD<space, data_t>> svd;
    svd.resize(exponent - 1);
    for (size_t d = 0; d < exponent - 1; d++)
    {
      svd[d].tolerance_relative = svd_tolerance_relative;
      svd[d].tolerance_absolute = svd_tolerance_absolute;
      svd[d].max_kept_singular_values = svd_max_kept_values;
    }

    for (size_t d = exponent - 1; d > 0; d--)
    {
      checkpoint();
      svd[d - 1](unfold_right);
      if (svd[d - 1].significant_singular_values == 0)
      {
        this->fill_with_zeros();
        return;
      }
      apply_as_diagonal_right_in_place(svd[d - 1].S, svd[d - 1].U);
      checkpoint();
      unfold_right = svd[d - 1].V.transpose();
      this->cores[d] = write_to_core_from_right_fold(unfold_right, sizes(d));
      svd[d - 1].V.resize({1, 1});
      checkpoint();
      auto unfold_left = qr[d - 1].apply_householder_left(svd[d - 1].U);
      qr[d - 1].Q.resize({1, 1});
      svd[d - 1].U.resize({1, 1});
      checkpoint();
      if (d > 1)
      {
        unfold_right = compute_unfold_right_from_unfold_left(unfold_left, sizes(d - 1));
      }
      else if (d == 1)
      {
        this->cores[0] = write_to_core_from_left_fold(unfold_left, sizes(0));
      }
      checkpoint();
    }
    checkpoint();
  }

  /**
   * \brief
   * orthogonalize the cores by performing a sequence of QRs along the cores
   */

  void orthogonalize()
  {
    BOBA_CALI_MARK
    checkpoint();
    auto dimension = get_number_cores();
    if (dimension == 1)
    {
      bool left_rank_one = this->get_ranks_left(0) == 1;
      bool right_rank_one = this->get_ranks_right(0) == 1;
      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core = tensor_reduction<2>({"l", "i", "r"}, this->cores[0], {"i"});
      this->cores[0] = reshape<3>(new_core, {1_z, sizes(0), 1_z});
    }

    // Do this procedure for all cores except last
    for (size_t d = 0; d < dimension - 1; d++)
    {
      auto unfold_left = compute_unfold_left(this->cores[d]);
      boba::QR<space, data_t> qr;
      qr(unfold_left);
      this->cores[d] = write_to_core_from_left_fold(qr.Q, sizes(d));
      auto temp = boba::tensor_contraction<1>({"k", "l"}, qr.R, {"l", "i", "r"}, this->cores[d + 1], {"k", "i", "r"});
      this->cores[d + 1] = temp;
      checkpoint();
    }
    checkpoint();
  }

  // -------------------------------------------------------------------------------------
  // Section: Multiply scalar
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Multiplies the tensor train by x, such that A = x*A
   */

  void multiply_scalar(data_t x)
  {
    checkpoint();
    this->cores.at(0).multiply_scalar(x);
  }

  /**
   * \brief
   * Multiplies each core of the tensor train by its own scalar, such that A = product(scalars)*A
   */

  void multiply_scalars(const Vector<host_space, data_t> scalars)
  {
    checkpoint();
    auto scalars_view = scalars.const_view();
    for (size_t d = 0; d < exponent; d++)
    {
      this->cores.at(d).multiply_scalar(scalars_view(d));
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Reduction
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * QTT dot product
   */

  data_t inner_product(const QuantizedTensorTrain& input) const
  {
    BOBA_CALI_MARK
    // little-l 2 norm
    checkpoint();
    boba::Vector<space, data_t> in;
    in.rename("in");
    boba::Vector<space, data_t> out;
    out.rename("out");
    for (size_t dp1 = exponent; dp1 > 0; dp1--)
    {
      auto d = dp1 - 1;
      // Compute new core sizes
      checkpoint();
      auto this_core_view = this->cores.at(d).const_view();
      const size_t this_ranks_left = this->get_ranks_left(d);
      const size_t this_ranks_right = this->get_ranks_right(d);
      const size_t this_size = this->get_core_size(d);
      checkpoint();
      auto in_core_view = input.cores.at(d).const_view();
      const size_t in_ranks_left = input.get_ranks_left(d);
      const size_t in_ranks_right = input.get_ranks_right(d);
      boba_assert_equal(input.get_core_size(d), this_size, "incompatible sizes");
      // Initialize reduction tensor
      const size_t new_ranks_left = in_ranks_left * this_ranks_left;
      const size_t new_ranks_right = in_ranks_right * this_ranks_right;
      checkpoint();
      if (d == exponent - 1)
      {
        boba_assert_equal(new_ranks_right, 1_z, "whoops");
        in.resize({1});
        in.fill_with(1.0);
      }
      else
      {
        in = out;
      }
      out.resize({new_ranks_left});
      out.fill_with_zeros();
      boba_assert_equal(input.get_ranks_left(d), in_ranks_left, "incompatible sizes");
      boba_assert_equal(input.get_ranks_right(d), in_ranks_right, "incompatible sizes");
      boba_assert_equal(this->get_ranks_left(d), this_ranks_left, "incompatible sizes");
      boba_assert_equal(this->get_ranks_right(d), this_ranks_right, "incompatible sizes");
      checkpoint();
      auto rank_left_view = ::boba::Multiindexer<2>({this_ranks_left, in_ranks_left});
      auto rank_right_view = ::boba::Multiindexer<2>({this_ranks_right, in_ranks_right});

      boba::Matrix<space, data_t> temp({new_ranks_left, new_ranks_right});
      temp.fill_with_zeros();
      auto temp_atomic_view = temp.atomic_view();

      ::boba::loop<space, 3>({new_ranks_left, new_ranks_right, this_size},
                             [=] __boba_host_device__(Array<size_t, 3> ijk)
      {
        auto [i, j, k] = ijk;
        auto rank_left_mid = rank_left_view.multiindex(i);
        auto rank_right_mid = rank_right_view.multiindex(j);
        const auto value_this = this_core_view({rank_left_mid[0], k, rank_right_mid[0]});
        const auto value_in = in_core_view({rank_left_mid[1], k, rank_right_mid[1]});
        temp_atomic_view({i, j}) += value_this * value_in;
      });

      out = temp * in;
      if (d == 0)
      {
        break;
      }
    }
    checkpoint();
    boba_assert_equal(out.size(), 1_z, "end result should be a scalar");
    auto inner_product = out.sum_reduce();
    checkpoint();
    return inner_product;
  }

  // -------------------------------------------------------------------------------------
  // Section: Extract rank one terms
  // -------------------------------------------------------------------------------------

  QuantizedTensorTrain<space, data_t> extract_rank_one_TensorTrain()
  {
    BOBA_CALI_MARK

    std::vector<size_t> rank_indices(exponent + 1_z, 0);
    QuantizedTensorTrain rank_one(base, exponent);

    rank_one = extract_rank_one_TensorTrain(rank_indices);

    return rank_one;
  }

  QuantizedTensorTrain<space, data_t> extract_rank_one_TensorTrain(
    const std::vector<size_t> rank_indices)
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < exponent; d++)
    {
      boba_always_assert_ge(rank_indices[d], 0_z, "Must be positive.");
      boba_always_assert_lt(rank_indices[d], get_ranks_left(d), "Attempting to access an out of bounds rank.");
      boba_always_assert_lt(rank_indices[d + 1], get_ranks_right(d), "Attempting to access an out of bounds rank.");
    }

    checkpoint();
    QuantizedTensorTrain rank_one(base, exponent);
    rank_one.fill_with_zeros();

    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {

      checkpoint();
      auto core_view = cores[d].const_view();
      checkpoint();
      auto rank_one_view = rank_one.cores.at(d).view();

      auto rank_index_left = rank_indices[d];
      auto rank_index_right = rank_indices[d + 1];

      checkpoint();
      ::boba::loop<space, 1>(base,
                             [=] __boba_host_device__(size_t index)
      {
        auto x = core_view({rank_index_left, index, rank_index_right});
        rank_one_view({0_z, index, 0_z}) = x;
      });
    }

    return rank_one;
  }
};

// -------------------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Write the tt to file in a way consistent with Tensor::write_to_file
 */

template <::boba::execution_space space, typename _data_t>
void write_to_file(const QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  auto dimension = tt.get_number_cores();
  for (size_t d = 0; d < dimension; d++)
  {
    boba::write_to_file(tt.cores[d], print_filename + "_core_" + std::to_string(d));
  }
  ::boba::Tensor<1, host_space, size_t> temp({1});
  temp.fill_with(tt.exponent);
  boba::write_to_file(temp, print_filename + "_exponent");
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <::boba::execution_space space, typename _data_t>
void read_from_file(QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }
  ::boba::Tensor<1, host_space, size_t> temp({1});
  boba::read_from_file(temp, print_filename + "_exponent");
  auto dimension = temp({0});
  tt.cores.resize(dimension);
  tt.exponent = dimension;
  for (size_t d = 0; d < dimension; d++)
  {
    boba::read_from_file(tt.cores[d], print_filename + "_core_" + std::to_string(d));
  }
  tt.base = tt.cores[0].sizes(1);
}

/**
 * \brief
 * Write the tt to a MATLAB mat-file as a cell array of cores.
 *
 * The resulting cell array can be converted to Oseledet's TT-toolbox tt_tensor class object:
 * ```matlab
 * >> load <filename>.mat; % creates a cell array named `filename` in the workspace
 * >> tt = cell2core(tt_tensor, filename);
 * ```
 */

template <::boba::execution_space space, typename _data_t>
void write_to_mat_file(const QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename = "")
{
  std::string print_filename = "";
  auto dimension = tt.get_number_cores();
  if (filename.empty())
  {
    print_filename = tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "w");
  matlab_file.write_cell_array(print_filename, tt.cores);
  ::boba::Tensor<1, host_space, size_t> temp({1_z});
  temp.fill_with(tt.exponent);
  matlab_file.write_array(print_filename + "_exponent", temp);
}

/**
 * \brief
 * Read from a file generated by write_to_mat_file
 */

template <::boba::execution_space space, typename _data_t>
void read_from_mat_file(QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename = "")
{
  std::ofstream file;
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = tt.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  detail::MatFile matlab_file(print_filename, "r");
  ::boba::Tensor<2, space, size_t> temp({1_z, 1_z});
  matlab_file.read_array(print_filename + "_exponent", temp);
  tt.exponent = temp({0_z, 0_z});
  tt.cores.resize(tt.get_number_cores());
  matlab_file.read_cell_array(print_filename, tt.cores);
  tt.base = tt.cores[0].sizes(1);
}

/**
 * \brief
 * Write the tt to a HDF5 file
 */

template <::boba::execution_space space, typename _data_t>
void write_to_hdf5_file(const QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = tt.name();
  }

  detail::HDF5File h5_file(filename, "w");
  auto dimension = tt.get_number_cores();
  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.write_array(object_name + "_core_" + std::to_string(d), tt.cores[d]);
  }
  h5_file.write_int(object_name + "_exponent", tt.exponent);
}

/**
 * \brief
 * Read from a file generated by write_to_hdf5_file
 */

template <::boba::execution_space space, typename _data_t>
void read_from_hdf5_file(QuantizedTensorTrain<space, _data_t>& tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = tt.name();
  }

  detail::HDF5File h5_file(filename, "r");
  tt.exponent = static_cast<index_t>(h5_file.read_int(object_name + "_exponent"));

  tt.cores.resize(tt.get_number_cores());
  auto dimension = tt.get_number_cores();
  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.read_array(object_name + "_core_" + std::to_string(d), tt.cores[d]);
  }
  tt.base = tt.cores[0].sizes(1);
}

} // namespace boba
