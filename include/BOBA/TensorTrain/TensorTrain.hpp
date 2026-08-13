// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tensor train. Choo choo!
\verbatim
    (1, r_0)     (r_0, r_1)    (r_1, r_2)    (r_d-2, r_d-1)  (r_d-1, 1)
 ___  ____     __   ____      ___  ____         ___  ____       ____
 \    |  |     \    |  |      \    |  |         \    |  |       |  |
  >   |  | (x)  >   |  | (x)   >   |  | (x) ...  >   |  |  (x)  |  |
 /__  |__|     /__  |__|      /__  |__|         /__  |__|       |__|
 r_0 sizes[0]  r_1  sizes[1]   r2  sizes[2]   r_d-1  sizes[d-2]  sizes[d-1]
\endverbatim
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
struct TensorTrain
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "TensorTrain";
  ::boba::Array<size_t, dimension + 1> max_ranks = ::boba::filled_array<dimension + 1>(highest_value<size_t>());

  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);

  ::boba::Array<Tensor<3, space, _data_t>, dimension> cores;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct from array of sizes and max ranks, then call resize. Instantiated object is the zero tensor train.
   */

  TensorTrain(::boba::Array<index_t, dimension> input_sizes)
  {
    this->resize(input_sizes);
  }

  // Default constructor
  TensorTrain() = default;

  // Copy constructor
  TensorTrain(TensorTrain const&) = default;

  // Move constructor
  TensorTrain(TensorTrain&&) = default;

  // Copy assignment
  TensorTrain& operator=(TensorTrain const&) = default;

  // Move assignment
  TensorTrain& operator=(TensorTrain&&) = default;

  /**
   * \brief copy constructor for a different execution space
   */

  template <execution_space rhs_space>
  TensorTrain(TensorTrain<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        max_ranks(rhs.max_ranks),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        cores(::boba::typed_array<Tensor<3, space, _data_t>>(rhs.cores))
  {
  }

  /**
   * \brief copy assignment operator for a different execution space
   */

  template <execution_space rhs_space>
  TensorTrain& operator=(TensorTrain<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    max_ranks = rhs.max_ranks;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    cores = ::boba::typed_array<Tensor<3, space, _data_t>>(rhs.cores);
    return *this;
  }

  ~TensorTrain() = default;

  /**
   * \brief Resize the tensor train and reinitialize every core to zero.
   *
   * Unlike `Tensor::resize`, this does not preserve overlapping entries.
   *
   * \param[in] input_sizes tensor extents for each core
   */

  void resize(::boba::Array<index_t, dimension> input_sizes)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      cores[d].resize({1, input_sizes[d], 1});
      cores[d].fill_with_zeros();
    }
    checkpoint();
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
  }

  // -------------------------------------------------------------------------------------
  // Printing
  // -------------------------------------------------------------------------------------

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    for (size_t d = 0; d < dimension; d++)
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
   * \brief Get the tensor train name.
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
   * \brief Get the dimensionality of the tensor train object.
   *
   * \return tensor dimension
   */

  static constexpr size_t get_dimension()
  {
    return dimension;
  }

  [[nodiscard]]
  size_t get_ranks_left(size_t i) const
  {
    boba_always_assert_lt(i, dimension, "invalid core number");
    return this->cores[i].sizes(0);
  }

  [[nodiscard]]
  size_t get_ranks_right(size_t i) const
  {
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(2);
  }

  [[nodiscard]]
  Array<size_t, dimension> get_ranks_left() const
  {
    Array<size_t, dimension> ranks;
    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] = get_ranks_left(d);
    }
    return ranks;
  }

  [[nodiscard]]
  Array<size_t, dimension> get_ranks_right() const
  {
    Array<size_t, dimension> ranks;
    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] = get_ranks_right(d);
    }
    return ranks;
  }

  [[nodiscard]]
  size_t get_number_elements() const
  {
    size_t sum_elements = 0;
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += this->cores[d].get_number_elements();
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

  size_t ranks(size_t dim) const noexcept
  {
    return (dim < dimension) ? this->get_ranks_left(dim) : this->get_ranks_right(dimension - 1);
  }

  Array<size_t, dimension + 1> ranks() const noexcept
  {
    ::boba::Array<size_t, dimension + 1> this_ranks{0};
    for (size_t d = 0; d < dimension + 1; d++)
    {
      this_ranks[d] = ranks(d);
    }
    return this_ranks;
  }

  index_t sizes(index_t d) const noexcept
  {
    return this->cores[d].sizes(1);
  }

  // -------------------------------------------------------------------------------------
  // Section: Views
  // -------------------------------------------------------------------------------------

  using core_view_type = ::boba::TensorView<::boba::DefaultAccessor<data_t>, 3>;

  /**
   * \brief
   * Get an array of views of the tt cores
   */

  ::boba::Array<core_view_type, dimension> get_core_views() const
  {
    ::boba::Array<core_view_type, dimension> views;
    for (size_t d = 0; d < dimension; d++)
    {
      views[d] = this->cores[d].view();
    }
    return views;
  }

  using core_const_view_type = ::boba::TensorView<::boba::DefaultAccessor<data_t const>, 3>;

  /**
   * \brief
   * Get an array of const views of the tt cores
   */

  ::boba::Array<core_const_view_type, dimension> get_core_const_views() const
  {
    ::boba::Array<core_const_view_type, dimension> views;
    for (size_t d = 0; d < dimension; d++)
    {
      views[d] = this->cores[d].const_view();
    }
    return views;
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
    set_to_rank_one_scalar_core(this->cores[0], x);
    for (size_t d = 1; d < dimension; d++)
    {
      set_to_rank_one_scalar_core(this->cores[d], 1.0);
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
   * Sets this into a rank-one train filled with zeros
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK
    for (size_t d = 0; d < dimension; d++)
    {
      set_to_rank_one_scalar_core(this->cores[d], 0.0);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Unroll
  // -------------------------------------------------------------------------------------

  data_t unroll_value(
    const Array<size_t, dimension> indices) const
  {
    Tensor<1, space, data_t> temp({1});
    temp.fill_with(1);

    for (size_t d = 0; d < dimension; d++)
    {
      auto ranks_left = get_ranks_left(d);
      auto ranks_right = get_ranks_right(d);
      Tensor<2, space, data_t> slice({ranks_left, ranks_right});
      auto slice_view = slice.view();
      auto core_view = this->cores[d].const_view();
      auto slice_id = indices[d];

      ::boba::loop<space, 2>({ranks_left, ranks_right},
                             [=] __boba_host_device__(::boba::Array<size_t, 2> mids)
      {
        auto [left, right] = mids;
        slice_view(mids) = core_view({left, slice_id, right});
      });

      auto old_temp = temp;
      temp = tensor_contraction<1>(
        {"ranks_left"}, old_temp, {"ranks_left", "ranks_right"}, slice, {"ranks_right"});
    }

    boba_always_assert_equal(temp.size(), 1_z, "assume scalar for now");

    return temp.sum_reduce();
  }

  Tensor<dimension, space, data_t> unroll_subtensor(
    const Array<size_t, dimension>& start,
    const Array<size_t, dimension>& end) const
  {
    BOBA_CALI_MARK
    checkpoint();

    Array<size_t, dimension> subtensor_sizes;
    for (size_t d = 0; d < dimension; d++)
    {
      subtensor_sizes[d] = end[d] - start[d];
    }

    if constexpr (dimension == 1)
    {
      Tensor<dimension, space, data_t> subtensor(subtensor_sizes);
      subtensor.fill_with(0.0);
      size_t offset = start[0];
      auto subtensor_view = subtensor.atomic_view();
      auto this_view = this->cores[0].const_view();

      ::boba::loop<space, 3>({subtensor.size(), this->get_ranks_left(0), this->get_ranks_right(0)},
                             [=] __boba_host_device__(Array<size_t, 3> ijk)
      {
        auto [i, l, r] = ijk;
        subtensor_view(i) += this_view({l, i + offset, r});
      });

      return subtensor;
    }

    Tensor<3, space, _data_t> new_core({1, 1, 1});
    new_core.rename("new_core");
    new_core.fill_with(1.0);
    checkpoint();

    // TODO<optimization> judiciously choose the order
    for (size_t d = dimension; d > 0; d--)
    {
      checkpoint();
      Array<size_t, 2> partial_start{start[d - 1], 0};
      Array<size_t, 2> partial_end{end[d - 1], new_core.sizes(1)};
      auto temp_core = partial_decompress_core(partial_start, partial_end, this->cores[d - 1], new_core);
      new_core = temp_core;
    }

    //
    // Recast tensor core into tensor
    //
    checkpoint();
    Tensor<dimension, space, data_t> output(subtensor_sizes);
    output.reshape(new_core);

    checkpoint();
    return output;
  }

  boba::Tensor<dimension, space, data_t> decompress() const
  {
    BOBA_CALI_MARK
    return unroll_subtensor(::boba::filled_array<dimension>(0_z), sizes());
  }

  // -------------------------------------------------------------------------------------
  // Section: Partial Unroll
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Maps a tensor train to a lower-dimensional tensor train
   * Example: say you have a ttrain of cores A B C D. Calling this function with
   * dimension_to_decompress = 2 corresponds to unrolling B and C into one core.
   * The new ttrain will be A B*C D, where * is the unroll operation.
   */

  TensorTrain<dimension - 1, space, data_t> partial_decompress(
    const size_t dimension_to_decompress) const
  {
    BOBA_CALI_MARK
    checkpoint();

    //
    // determine new ttrain parameters
    //
    constexpr size_t new_dimensions = dimension - 1;
    Array<size_t, new_dimensions> new_ttrain_sizes;
    for (size_t d = 0; d < dimension_to_decompress - 1; d++)
    {
      new_ttrain_sizes[d] = sizes(d);
    }
    new_ttrain_sizes[dimension_to_decompress - 1] = sizes(dimension_to_decompress - 1) * sizes(dimension_to_decompress);

    for (size_t d = dimension_to_decompress + 1; d < dimension; d++)
    {
      new_ttrain_sizes[d - 1] = sizes(d);
    }

    //
    // Write new train, starting with the cores before the decompressed core
    //
    TensorTrain<new_dimensions, space, data_t> new_ttrain(new_ttrain_sizes);
    for (size_t d = 0; d < dimension_to_decompress - 1; d++)
    {
      new_ttrain.cores[d] = this->cores[d];
    }
    checkpoint();
    //
    // Handle the decompressed cores
    //
    {
      auto new_core = partial_decompress_core(this->cores[dimension_to_decompress - 1], this->cores[dimension_to_decompress]);
      new_ttrain.cores[dimension_to_decompress - 1] = new_core;
    }
    //
    // The remaining non-decompressed cores
    //
    checkpoint();
    for (size_t d = dimension_to_decompress + 1; d < dimension; d++)
    {
      new_ttrain.cores[d - 1] = this->cores[d];
    }
    checkpoint();
    for (size_t d = 0; d < new_dimensions - 1; d++)
    {
      boba_always_assert_equal(new_ttrain.get_ranks_right(d), new_ttrain.get_ranks_left(d + 1), "Ranks mismatch");
    }
    checkpoint();
    return new_ttrain;
  }

  // -------------------------------------------------------------------------------------
  // Section: Insert dimension
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Maps a ttrain to a higher-dimensional ttrain.
   * Example: say you have a ttrain of cores A B D. Calling this function with
   * dimension_to_insert = 2 corresponds to inserting before D.
   * The new ttrain will be A B C D, where C = 1 in the long dimension.
   * This implies the new ttrain's conrresponding tensor to be equal to the old one modulo a reshape.
   */

  TensorTrain<dimension + 1, space, data_t> insert_dimension(
    const size_t dimension_to_insert,
    const size_t new_size = 1) const
  {
    BOBA_CALI_MARK
    checkpoint();
    //
    // determine new ttrain parameters
    //
    constexpr size_t new_dimensions = dimension + 1;
    Array<size_t, new_dimensions> new_ttrain_sizes;
    checkpoint();
    for (size_t d = 0; d < dimension_to_insert; d++)
    {
      new_ttrain_sizes[d] = sizes(d);
    }
    new_ttrain_sizes[dimension_to_insert] = new_size;
    for (size_t d = dimension_to_insert; d < dimension; d++)
    {
      new_ttrain_sizes[d + 1] = sizes(d);
    }

    //
    // write new train
    //
    TensorTrain<new_dimensions, space, data_t> new_ttrain(new_ttrain_sizes);
    for (size_t d = 0; d < dimension_to_insert; d++)
    {
      new_ttrain.cores[d] = this->cores[d];
    }
    checkpoint();
    size_t new_ranks_left = 1;
    size_t new_ranks_right = 1;
    if (dimension_to_insert > 0)
    {
      new_ranks_left = get_ranks_right(dimension_to_insert - 1);
    }
    if (dimension_to_insert < dimension)
    {
      new_ranks_right = get_ranks_left(dimension_to_insert);
    }

    Array<size_t, 3> resize{new_ranks_left, new_size, new_ranks_right};

    new_ttrain.cores[dimension_to_insert].resize(resize);

    if ((new_ranks_left == 1) or (new_ranks_right == 1))
    {
      new_ttrain.cores[dimension_to_insert].fill_with(1.0);
    }
    else
    {
      auto new_core_view = new_ttrain.cores[dimension_to_insert].view();
      checkpoint();
      ::boba::loop<space, 3>(resize,
                             [=] __boba_host_device__(Array<index_t, 3> ijk)
      {
        auto rank_left = ijk[0];
        auto rank_right = ijk[2];
        auto value = ::boba::PotentiallyComplex<_data_t>::value(0.0);
        if (rank_left == rank_right)
        {
          value = ::boba::PotentiallyComplex<_data_t>::value(1.0);
        }
        new_core_view(ijk) = value;
      });
    }
    checkpoint();
    for (size_t d = dimension_to_insert; d < dimension; d++)
    {
      new_ttrain.cores[d + 1] = this->cores[d];
    }
    for (size_t d = 0; d < new_dimensions - 1; d++)
    {
      boba_always_assert_equal(new_ttrain.get_ranks_right(d), new_ttrain.get_ranks_left(d + 1), "Ranks mismatch");
    }
    checkpoint();
    return new_ttrain;
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tensor train addition of this and subtrain, where subtrain is smaller than or the same size as this.
   * The subtrain must have an offset and size for each dimension such that offset+size fits into this.
   * Zeros are padded to subtrain so that addition is valid.
   */

  void add_subtrain(
    TensorTrain const& subtrain,
    const ::boba::Array<size_t, dimension> initial_index)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_assert(sizes(d) >= (initial_index[d] + subtrain.sizes(d)), "Subtrain size exceeds parent");
    }

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      add_subcore(this->cores[d], subtrain.cores[d], d, dimension, initial_index[d]);
    }
  }

  /**
   * \brief
   */

  void TensorTrain_add(
    TensorTrain const& input,
    const ::boba::Array<size_t, dimension> initial_indices)
  {
    BOBA_CALI_MARK
    checkpoint();
    this->add_subtrain(input, initial_indices);
    checkpoint();
  }

  /**
   * \brief
   * Tensor train addition.
   */

  void TensorTrain_add(TensorTrain const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      boba_assert_equal(input.sizes(d), this->sizes(d), "incompatible sizes");
    }
    checkpoint();

    auto initial_indices = ::boba::filled_array<dimension>(0_z);

    checkpoint();
    this->add_subtrain(input, initial_indices);
    checkpoint();
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrain& operator+=(TensorTrain const& rhs)
  {
    BOBA_CALI_MARK
    this->TensorTrain_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TensorTrain operator+(TensorTrain const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrain output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrain& operator-=(TensorTrain const& rhs)
  {
    BOBA_CALI_MARK
    TensorTrain rhs_minus{rhs};
    rhs_minus *= static_cast<real_data_t>(-1);
    this->TensorTrain_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TensorTrain operator-(TensorTrain const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrain output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * negation operator
   */

  TensorTrain operator-()
  {
    BOBA_CALI_MARK
    TensorTrain output{*this};
    output *= static_cast<real_data_t>(-1);
    return output;
  }

  /**
   * \brief
   * this *= scalar
   */

  TensorTrain& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    multiply_scalar(scalar);
    return *this;
  }

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  TensorTrain& operator*=(_real_data_t const scalar)
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

  TensorTrain operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    TensorTrain output{*this};
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

  void compress(const boba::Tensor<dimension, space, data_t>& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    if constexpr (boba::is_ci_mode())
    {
      // TODO<bugfix>
      boba_always_assert(input.sizes() == sizes(), "tensor train not correctly set up to compress this tensor");
    }

    if constexpr (dimension == 1_z)
    {
      cores[0].resize({1_z, sizes(0), 1_z});
      cores[0].reshape(input);
      return;
    }

    ::boba::Array<boba::SVD<space, data_t>, dimension - 1> svd;

    for (size_t d = 0; d < dimension - 1; d++)
    {
      svd[d].tolerance_relative = svd_tolerance_relative;
      svd[d].tolerance_absolute = svd_tolerance_absolute;
      svd[d].max_kept_singular_values = max_ranks[1 + d];
    }

    size_t sizes_product = product(sizes());
    checkpoint();
    size_t rows = 1;
    size_t cols = sizes_product;
    checkpoint();
    boba::Matrix<space, _data_t> temp_fold({rows, cols});
    checkpoint();
    temp_fold.rename("folding_matrix");
    checkpoint();
    temp_fold.reshape(input);
    checkpoint();

    size_t ranks = 1;
    for (size_t d = 0; d < dimension - 1; d++)
    {
      checkpoint();
      size_t this_size = sizes(d);
      rows = this_size * ranks;
      cols = temp_fold.size() / rows;
      checkpoint();
      temp_fold.reshape({rows, cols});
      checkpoint();
      svd[d](temp_fold);
      apply_as_diagonal_right_in_place(svd[d].S, svd[d].U);
      checkpoint();
      ranks = svd[d].significant_singular_values;
      if (ranks == 0)
      {
        this->fill_with_zeros();
        return;
      }
      this->cores[d] = write_to_core_from_left_fold(svd[d].U, sizes(d));
      temp_fold = svd[d].V.transpose();
      checkpoint();
    }
    checkpoint();
    this->cores[dimension - 1] = write_to_core_from_right_fold(temp_fold, sizes(dimension - 1));
    checkpoint();
  }

  /**
   * \brief
   * Left-orthogonalize this tensor train by a left-to-right QR sweep.
   *
   * After this operation, all cores except the last are left-orthogonal. The
   * triangular factor from each QR factorization is absorbed into the left rank
   * index of the next core, so the represented tensor is unchanged up to
   * floating-point roundoff.
   */

  void orthogonalize()
  {
    BOBA_CALI_MARK
    checkpoint();
    if constexpr (dimension == 1)
    {
      bool left_rank_one = this->get_ranks_left(0) == 1;
      bool right_rank_one = this->get_ranks_right(0) == 1;

      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core = tensor_reduction_double_index({"l", "i", "r"}, this->cores[0], {"i"});
      this->cores[0] = reshape<3>(new_core, {1, sizes(0), 1});
      return;
    }

    // Reuse the QR workspace throughout the sweep
    boba::QR<space, data_t> qr;

    // Perform the left-to-right QR sweep
    for (size_t d = 0; d < dimension - 1; d++)
    {
      checkpoint();

      // Scope the unfolding so it is released after the QR step finishes
      {
        auto unfold_left = compute_unfold_left(this->cores[d]);
        qr(unfold_left);
      }

      // Q is the left unfolding of the current core
      checkpoint();
      this->cores[d] = write_to_core_from_left_fold(qr.Q, sizes(d));

      // Absorb R into the next core via contraction
      checkpoint();
      this->cores[d + 1] = boba::tensor_contraction<1>(
        {"k", "l"}, qr.R, {"l", "i", "r"}, this->cores[d + 1], {"k", "i", "r"});
    }
    checkpoint();
  }

  /**
   * \brief
   * Round this tensor train by left-orthogonalization followed by a right-to-left
   * truncated SVD sweep.
   *
   * The method first left-orthogonalizes the tensor train using a QR sweep. It
   * then performs a right-to-left SVD sweep, truncating each interface rank
   * according to svd_tolerance_relative, svd_tolerance_absolute, and max_ranks.
   * The resulting tensor train is right-orthogonal, with the first core as the
   * orthogonality center.
   */

  void round()
  {
    BOBA_CALI_MARK

    checkpoint();

    if constexpr (dimension == 1)
    {
      bool left_rank_one = this->get_ranks_left(0) == 1;
      bool right_rank_one = this->get_ranks_right(0) == 1;

      if (left_rank_one and right_rank_one)
      {
        return;
      }

      auto new_core = tensor_reduction_double_index({"l", "i", "r"}, this->cores[0], {"i"});
      this->cores[0] = reshape<3>(new_core, {1, sizes(0), 1});
      return;
    }
    // Reuse the SVD workspace during the sweep
    boba::SVD<space, data_t> svd;
    svd.tolerance_relative = svd_tolerance_relative;
    svd.tolerance_absolute = svd_tolerance_absolute;

    // First, we (left) orthogonalize the tensor train
    // TODO<optimization> later we should make orthogonalization a class attribute and update its value
    // so that orthogonalization becomes no-op for tensors which are already orthogonalized. Note that this
    // requires we adopt some convention, e.g., left/right orthogonality.
    checkpoint();
    this->orthogonalize();

    // Perform the right-to-left SVD sweep
    for (size_t d = dimension - 1; d > 0; d--)
    {
      checkpoint();

      // Set the maximum number of singular values retained for this mode
      svd.max_kept_singular_values = max_ranks[d];

      {
        auto unfold_right = compute_unfold_right(this->cores[d]);
        svd(unfold_right);
      }

      if (svd.significant_singular_values == 0_z)
      {
        this->fill_with_zeros();
        return;
      }

      // Write V^T back into the current core. The current core is now
      // right-orthogonal
      {
        auto unfold_right = svd.V.transpose();
        this->cores[d] = write_to_core_from_right_fold(unfold_right, sizes(d));
      }

      // Form U*S in-place and absorb it into the right rank index of the
      // preceding core
      checkpoint();
      apply_as_diagonal_right_in_place(svd.S, svd.U);

      checkpoint();
      this->cores[d - 1] =
        boba::tensor_contraction<1>(
          {"l", "i", "k"}, this->cores[d - 1], {"k", "r"}, svd.U, {"l", "i", "r"});
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
    this->cores[0] *= x;
  }

  /**
   * \brief
   * Multiplies each core of the tensor train by its own scalar, such that A = product(scalars)*A
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
   * Inner product of this and input TT
   */

  data_t inner_product(const TensorTrain<dimension, space, _data_t>& input) const
  {
    BOBA_CALI_MARK
    checkpoint();
    boba::Vector<space, data_t> in({1});
    in.fill_with(1.0);
    in.rename("in");
    boba::Vector<space, data_t> out;
    out.rename("out");

    // auto this_cores_views = this->get_core_const_views();
    // auto in_cores_views = input.get_core_const_views();

    for (size_t d = dimension; d > 0; d--)
    {
      auto contracted_cores = tensor_contraction<1>(
        {"l1", "i", "r1"}, this->cores[d - 1], {"l2", "i", "r2"}, input.cores[d - 1], {"l1", "l2", "r1", "r2"});

      const size_t this_ranks_left = this->get_ranks_left(d - 1);
      const size_t this_ranks_right = this->get_ranks_right(d - 1);
      const size_t in_ranks_left = input.get_ranks_left(d - 1);
      const size_t in_ranks_right = input.get_ranks_right(d - 1);
      const size_t new_ranks_left = in_ranks_left * this_ranks_left;
      const size_t new_ranks_right = in_ranks_right * this_ranks_right;
      boba::Matrix<space, _data_t> temp({new_ranks_left, new_ranks_right});
      temp.reshape(contracted_cores);

      out = temp * in;

      in = out;
    }
    checkpoint();
    boba_assert_equal(out.size(), 1_z, "end result should be a scalar");
    auto inner_product = out.sum_reduce();
    checkpoint();
    return inner_product;
  }

  // -------------------------------------------------------------------------------------
  // Section: Apply arbitrary function
  // -------------------------------------------------------------------------------------

  template <typename function_type>
  TensorTrain<dimension, space, _data_t> apply_function_elementwise(
    function_type function,
    Array<size_t, dimension> n_chunks_per_dimension)
  {
    BOBA_CALI_MARK
    checkpoint();
    Array<size_t, dimension> chunk_size;

    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_positive(n_chunks_per_dimension[d], "Chunks must be positive.");
      n_chunks_per_dimension[d] = min(n_chunks_per_dimension[d], this->sizes(d));
      chunk_size[d] = boba::ceil(double(this->sizes(d)) / double(n_chunks_per_dimension[d]));
      chunk_size[d] = boba::max(chunk_size[d], 1_z);
      n_chunks_per_dimension[d] = boba::ceil(double(this->sizes(d)) / double(chunk_size[d]));
    }

    auto chunks = ::boba::Multiindexer<dimension>(n_chunks_per_dimension);
    size_t total_chunks = chunks.size();

    checkpoint();
    TensorTrain<dimension, space, _data_t> output(this->sizes());

    for (size_t chunk_id = 0; chunk_id < total_chunks; chunk_id++)
    {
      auto chunk_multiindex = chunks.multiindex(chunk_id);

      Array<size_t, dimension> begin_indices = filled_array<dimension>(0_z);
      Array<size_t, dimension> replacement_indices = filled_array<dimension>(0_z);
      Array<size_t, dimension> end_indices = this->sizes();
      Array<size_t, dimension> this_chunk_size = this->sizes();

      for (size_t d = 0; d < dimension; d++)
      {
        size_t di = chunk_multiindex[d];
        begin_indices[d] = di * chunk_size[d];
        replacement_indices[d] = begin_indices[d];
        end_indices[d] = boba::min(begin_indices[d] + chunk_size[d], this->sizes(d));
        this_chunk_size[d] = end_indices[d] - begin_indices[d];
      }

      //
      BOBA_CALI_BEGIN("unroll");
      //
      checkpoint();
      auto chunk = this->unroll_subtensor(begin_indices, end_indices);
      auto chunk_view = chunk.view();

      //
      BOBA_CALI_SWITCH("unroll", "apply_function_to_chunk");
      //
      checkpoint();

      ::boba::loop<space, 1>({chunk.size()},
                             [=] __boba_host_device__(size_t i)
      {
        auto mid = chunk_view.multiindex(i);
        auto x = chunk_view(mid);
        chunk_view(mid) = function(x);
      });

      //
      BOBA_CALI_SWITCH("apply_function_to_chunk", "create_output_subtrain");
      //
      checkpoint();
      TensorTrain<dimension, space, _data_t> output_subtrain(this_chunk_size);
      output_subtrain.compress(chunk);
      checkpoint();
      output.add_subtrain(output_subtrain, replacement_indices);
      output.round();
    }

    checkpoint();
    output.rename(m_name);
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Extract rank one terms
  // -------------------------------------------------------------------------------------

  TensorTrain<dimension, space, _data_t> extract_rank_one_TensorTrain(
    const std::vector<size_t> rank_indices)
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < dimension; d++)
    {
      boba_always_assert_ge(rank_indices[d], 0_z, "Must be positive.");
      boba_always_assert_lt(rank_indices[d], get_ranks_left(d), "Attempting to access an out of bounds rank.");
      boba_always_assert_lt(rank_indices[d + 1], get_ranks_right(d), "Attempting to access an out of bounds rank.");
    }

    checkpoint();
    TensorTrain rank_one(sizes());
    rank_one.fill_with_zeros();

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      checkpoint();
      auto core_view = cores[d].const_view();
      checkpoint();
      auto rank_one_view = rank_one.cores.at(d).view();

      auto rank_index_left = rank_indices[d];
      auto rank_index_right = rank_indices[d + 1];

      checkpoint();
      ::boba::loop<space, 1>(sizes(d),
                             [=] __boba_host_device__(size_t index)
      {
        auto x = core_view({rank_index_left, index, rank_index_right});
        rank_one_view({0_z, index, 0_z}) = x;
      });
    }

    return rank_one;
  }
};

template <size_t dimension, ::boba::execution_space space, typename _data_t>
TensorTrain<dimension, space, _data_t> operator*(_data_t scalar, TensorTrain<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  TensorTrain<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

// -------------------------------------------------------------------------------------
// Section: Left/right folds
// -------------------------------------------------------------------------------------

// ------------------------------------
// Left fold functions
// ------------------------------------
template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_left(
  ::boba::Tensor<3, space, _data_t> const& core)
{
  BOBA_CALI_MARK
  checkpoint();
  auto new_unfold_left = unfold(core, std::vector<size_t>{0, 1}, std::vector<size_t>{2});
  return new_unfold_left;
}

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_right(
  ::boba::Tensor<3, space, _data_t> const& core)
{
  BOBA_CALI_MARK
  checkpoint();
  auto new_unfold_right = unfold(core, 0_z);
  return new_unfold_right;
}

template <execution_space space, typename _data_t>
::boba::Tensor<3, space, _data_t> write_to_core_from_left_fold(
  boba::Matrix<space, _data_t> const& unfold_left,
  const index_t size)
{
  BOBA_CALI_MARK
  checkpoint();

  const size_t new_rows = unfold_left.rows();
  const size_t new_cols = unfold_left.cols();

  const size_t new_ranks_left = new_rows / size;
  const size_t new_ranks_right = new_cols;

  boba_assert_equal(size * new_ranks_left, new_rows, "size mismatch");
  boba_assert_positive(new_ranks_left, "nonpositive ranks");
  boba_assert_positive(new_ranks_right, "nonpositive ranks");

  auto new_core = ::boba::fold<3>(unfold_left, ::boba::Array<size_t, 3>{new_ranks_left, size, new_ranks_right}, std::vector<size_t>{0, 1}, std::vector<size_t>{2});

  return new_core;
}

template <execution_space space, typename _data_t>
::boba::Tensor<3, space, _data_t> write_to_core_from_right_fold(
  boba::Matrix<space, _data_t> const& unfold_right,
  const index_t size)
{
  BOBA_CALI_MARK

  const size_t new_rows = unfold_right.rows();
  const size_t new_cols = unfold_right.cols();

  const size_t new_ranks_left = new_rows;
  const size_t new_ranks_right = new_cols / size;

  boba_assert_equal(size * new_ranks_right, new_cols, "size mismatch");
  boba_assert_positive(new_ranks_left, "nonpositive ranks");
  boba_assert_positive(new_ranks_right, "nonpositive ranks");

  auto new_core = ::boba::fold<3>(unfold_right, ::boba::Array<size_t, 3>{new_ranks_left, size, new_ranks_right}, 0_z);

  return new_core;
}

// ------------------------------------
// Right-to-left and vice versa
// ------------------------------------

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_right_from_unfold_left(
  boba::Matrix<space, _data_t> const& unfold_left,
  const index_t size)
{
  BOBA_CALI_MARK
  checkpoint();

  const size_t temp_rows = unfold_left.rows();
  const size_t temp_cols = unfold_left.cols();
  const size_t temp_ranks_left = temp_rows / size;
  const size_t temp_ranks_right = temp_cols;

  boba_assert_equal(size * temp_ranks_left, temp_rows, "size mismatch");

  auto new_temp = ::boba::fold<3>(unfold_left, ::boba::Array<size_t, 3>{temp_ranks_left, size, temp_ranks_right}, std::vector<size_t>{0, 1}, std::vector<size_t>{2});
  auto new_unfold_right = unfold(new_temp, 0_z);

  return new_unfold_right;
}

template <execution_space space, typename _data_t>
boba::Matrix<space, _data_t> compute_unfold_left_from_unfold_right(
  boba::Matrix<space, _data_t> const& unfold_right,
  index_t size)
{
  BOBA_CALI_MARK
  checkpoint();

  const auto temp_rows = unfold_right.rows();
  const auto temp_cols = unfold_right.cols();
  const auto temp_ranks_left = temp_rows;
  const auto temp_ranks_right = temp_cols / size;

  boba_assert_equal(size * temp_ranks_right, temp_cols, "size mismatch");

  auto new_temp = ::boba::fold<3>(unfold_right, ::boba::Array<size_t, 3>{temp_ranks_left, size, temp_ranks_right}, 0_z);
  auto new_unfold_left = unfold(new_temp, std::vector<size_t>{0, 1}, std::vector<size_t>{2});

  return new_unfold_left;
}

template <execution_space space, typename _data_t, typename scalar_type>
void set_to_rank_one_scalar_core(
  Tensor<3_z, space, _data_t>& tt_core,
  scalar_type scalar)
{
  BOBA_CALI_MARK
  checkpoint();
  auto size = tt_core.sizes(1);
  tt_core.resize({1_z, size, 1_z});
  tt_core.fill_with(::boba::PotentiallyComplex<_data_t>::value(scalar));
}

template <execution_space space, typename _data_t, typename scalar_type>
void set_to_rank_one_scalar_core(
  Tensor<4_z, space, _data_t>& ttm_core,
  scalar_type scalar)
{
  BOBA_CALI_MARK
  checkpoint();
  auto rows = ttm_core.sizes(1);
  auto cols = ttm_core.sizes(2);
  ttm_core.resize({1_z, rows, cols, 1_z});
  ttm_core.fill_with(::boba::PotentiallyComplex<_data_t>::value(scalar));
}

template <execution_space space, typename _data_t>
void set_to_identity_core(
  Tensor<4_z, space, _data_t>& ttm_core)
{
  BOBA_CALI_MARK
  auto rows = ttm_core.sizes(1);
  auto cols = ttm_core.sizes(2);
  ttm_core.resize({1_z, rows, cols, 1_z});
  auto this_view = ttm_core.view();
  ::boba::loop<space, 2>({rows, cols},
                         [=] __boba_host_device__(Array<size_t, 2> rc)
  {
    auto value = ::boba::PotentiallyComplex<_data_t>::value(0.0);
    size_t r = rc[0];
    size_t c = rc[1];
    if (r == c)
    {
      value = ::boba::PotentiallyComplex<_data_t>::value(1.0);
    }
    this_view({0, r, c, 0}) = value;
  });
}

// -------------------------------------------------------------------------------------
// Unroll cores
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Unrolls two compatible cores into a larger core, new_core{i,[l p], j} = sum_k this{i, l, k} x other_{k, p, j}
 * Compatible means that the ranks match, such as with adjacent cores in a train
 */

template <execution_space space, typename _data_t>
Tensor<3, space, _data_t> partial_decompress_core(
  const Tensor<3, space, _data_t>& left_core,
  const Tensor<3, space, _data_t>& right_core)
{
  Array<size_t, 2> start_indices{0, 0};
  Array<size_t, 2> end_indices{left_core.sizes(1), right_core.sizes(1)};
  return partial_decompress_core(start_indices, end_indices, left_core, right_core);
}

/**
 * \brief
 * Unrolls two compatible cores into a larger core, new_core{i, [l p], j} = sum_k this{i, s+l, k} x other_{k, t+p, j}
 * Compatible means that the ranks match, such as with adjacent cores in a train
 * {s, t} form the start indices, allowing you to unroll subtrains.
 */

template <execution_space space, typename _data_t>
Tensor<3, space, _data_t> partial_decompress_core(
  Array<size_t, 2>& start_indices,
  Array<size_t, 2>& end_indices,
  const Tensor<3, space, _data_t>& left_core,
  const Tensor<3, space, _data_t>& right_core)
{
  BOBA_CALI_MARK

  // Requires resolving Eigen stride issue
  // https://lc.llnl.gov/gitlab/boba/boba/-/issues/238
  Array<size_t, 2> new_sizes{left_core.sizes(1), right_core.sizes(1)};
  boba_always_assert_equal(end_indices - start_indices, new_sizes, "Unroll subtensors not currently supported, see issue 238");

  auto tensor_4 = tensor_contraction<1>(
    {"l1", "i", "r1"}, left_core, {"r1", "j", "r2"}, right_core, {"l1", "i", "j", "r2"});

  auto new_size = product(new_sizes);
  auto new_rank_left = left_core.sizes(0);
  auto new_rank_right = right_core.sizes(2);
  Tensor<3, space, _data_t> new_core({new_rank_left, new_size, new_rank_right});

  new_core.reshape(tensor_4);
  return new_core;
}

template <execution_space space, typename _data_t>
void add_subcore(
  Tensor<3, space, _data_t>& this_core,
  Tensor<3, space, _data_t> const& subcore,
  size_t core_dim,
  size_t train_dimension,
  const size_t initial_index)
{
  BOBA_CALI_MARK

  /*

  For tt cores that aren't the first or last, this image represents addition

    current    added
     ranks     ranks
  _____________________
  |         |         |
  |  core   |    0    |  current ranks
  |_________|_________|
  |         |         |
  |    0    | subcore |  added ranks
  |_________|_________|

  For the first core, the rank addition looks something like this:

    current    added
     ranks     ranks
  _____________________
  |         |         |
  |  core   | subcore |  ranks = 1
  |_________|_________|

  For the last core, the rank addition looks something like this:

   ranks = 1
  ___________
  |         |
  |  core   | current ranks
  |_________|
  |         |
  | subcore | added ranks
  |_________|
  */

  // For each block
  /*
  _____
  |   | <- 0
  | 0 |
  |___| <- initial_index
  |   |
  | X |
  |___| <- initial_index + sub_size = end_nonzero_size
  |   |
  | 0 |
  |___| <- size

  */

  if (train_dimension == 1)
  {
    auto size = subcore.sizes(1);
    boba_assert_equal(subcore.sizes(0), 1_z, "Unexpected ranks for core");
    boba_assert_equal(subcore.sizes(2), 1_z, "Unexpected ranks for core");
    auto subtrain_view = subcore.const_view();
    auto this_view = this_core.view();
    ::boba::loop<space, 1>(size,
                           [=] __boba_host_device__(size_t i)
    {
      this_view(i + initial_index) += subtrain_view(i);
    });
    return;
  }

  const size_t added_ranks_left = subcore.sizes(0);
  const size_t added_ranks_right = subcore.sizes(2);

  const size_t sub_size = subcore.sizes(1);

  const size_t this_ranks_left = this_core.sizes(0);
  const size_t this_ranks_right = this_core.sizes(2);

  size_t new_ranks_left = this_ranks_left + added_ranks_left;
  size_t new_ranks_right = this_ranks_right + added_ranks_right;

  size_t this_ranks_left_offset = this_ranks_left;
  size_t this_ranks_right_offset = this_ranks_right;

  if (core_dim == 0)
  {
    new_ranks_left = this_ranks_left;
    this_ranks_left_offset = 0;
  }
  if (core_dim == train_dimension - 1)
  {
    new_ranks_right = this_ranks_right;
    this_ranks_right_offset = 0;
  }

  const size_t size = this_core.sizes(1);

  const size_t end_nonzero_size = initial_index + sub_size;

  boba_assert(size >= (initial_index + sub_size), "Subtrain size exceeds parent");

  checkpoint();
  this_core.resize({new_ranks_left, size, new_ranks_right});

  checkpoint();
  auto this_view = this_core.view();
  auto subcore_view = subcore.const_view();

  checkpoint();

  ::boba::loop<space, 3>({new_ranks_left, new_ranks_right, size},
                         [=] __boba_host_device__(Array<size_t, 3> ijk)
  {
    auto [rank_left, rank_right, index] = ijk;
    auto x = ::boba::PotentiallyComplex<_data_t>::value(0.0);

    // are the rank-indices on the support of the core?
    bool is_core_ranks_left = rank_left < this_ranks_left;
    bool is_core_ranks_right = rank_right < this_ranks_right;
    bool is_core_ranks = is_core_ranks_right && is_core_ranks_left;

    if (is_core_ranks)
    {
      x = this_view({rank_left, index, rank_right});
    }

    // are the indices on the support of the subcore?
    bool index_upper_bound = index < end_nonzero_size;
    bool index_lower_bound = index >= initial_index;
    bool is_subcore_index = index_upper_bound && index_lower_bound;

    // are the rank-indices on the support of the subcore?
    bool is_subcore_ranks_left_lower_bound = rank_left >= this_ranks_left_offset;
    bool is_subcore_ranks_left_upper_bound = rank_left < new_ranks_left;
    bool is_subcore_ranks_left = is_subcore_ranks_left_lower_bound && is_subcore_ranks_left_upper_bound;

    bool is_subcore_ranks_right_lower_bound = rank_right >= this_ranks_right_offset;
    bool is_subcore_ranks_right_upper_bound = rank_right < new_ranks_right;
    bool is_subcore_ranks_right = is_subcore_ranks_right_lower_bound && is_subcore_ranks_right_upper_bound;

    bool is_subcore_ranks = is_subcore_ranks_right && is_subcore_ranks_left;

    bool is_subcore = is_subcore_ranks && is_subcore_index;

    if (is_subcore)
    {
      const size_t sub_index = index - initial_index;
      const size_t sub_rank_left = rank_left - this_ranks_left_offset;
      const size_t sub_rank_right = rank_right - this_ranks_right_offset;

      x = subcore_view({sub_rank_left, sub_index, sub_rank_right});
    }

    this_view({rank_left, index, rank_right}) = x;
  });
}

// -------------------------------------------------------------------------------------
// I/O
// -------------------------------------------------------------------------------------

/**
 * \brief
 * Write the tt to file in a way consistent with Tensor::write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_file(const TensorTrain<dimension, space, _data_t>& tt, std::string_view filename = "")
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
  for (size_t d = 0; d < dimension; d++)
  {
    boba::write_to_file(tt.cores[d], print_filename + "_core_" + std::to_string(d));
  }
}

/**
 * \brief
 * Read from a file generated from write_to_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_file(TensorTrain<dimension, space, _data_t>& tt, std::string_view filename = "")
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
  for (size_t d = 0; d < dimension; d++)
  {
    boba::read_from_file(tt.cores[d], print_filename + "_core_" + std::to_string(d));
  }
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

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_mat_file(const TensorTrain<dimension, space, _data_t>& tt, std::string_view filename = "")
{
  std::string print_filename = "";

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
}

/**
 * \brief
 * Read from a file generated by write_to_mat_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_mat_file(TensorTrain<dimension, space, _data_t>& tt, std::string_view filename = "")
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
  matlab_file.read_cell_array(print_filename, tt.cores);
}

/**
 * \brief
 * Write the tt to a HDF5 file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void write_to_hdf5_file(const TensorTrain<dimension, space, _data_t>& tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = tt.name();
  }

  detail::HDF5File h5_file(filename, "w");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.write_array(object_name + "_core_" + std::to_string(d), tt.cores[d]);
  }
}

/**
 * \brief
 * Read from a file generated by write_to_hdf5_file
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void read_from_hdf5_file(TensorTrain<dimension, space, _data_t>& tt, std::string_view filename, std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = tt.name();
  }

  detail::HDF5File h5_file(filename, "r");

  for (size_t d = 0; d < dimension; d++)
  {
    h5_file.read_array(object_name + "_core_" + std::to_string(d), tt.cores[d]);
  }
}

} // namespace boba
