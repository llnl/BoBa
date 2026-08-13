// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Canonical Polyadic Decomposition
 */

template <size_t dimension, ::boba::execution_space space, typename data_t>
struct CanonicalPolyadicDecomposition
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using real_data_t = real_type_t<data_t>;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  std::string m_name = "CanonicalPolyadicDecomposition";

  real_data_t ALS_tolerance_relative = 1.0e-12;
  real_data_t ALS_tolerance_absolute = 1.0e-12;
  real_data_t ALS_iters = 201;
  size_t ALS_restarts = 3;

  Array<Matrix<space, data_t>, dimension> m_cores;
  Vector<space, data_t> m_weights;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief Constructor that resizes the CPD to the given extents and initializes it to zero.
   * \param[in] input_sizes tensor extents
   */

  CanonicalPolyadicDecomposition(::boba::Array<index_t, dimension> input_sizes)
  {
    this->resize(input_sizes);
  }

  /// \brief default constructor
  CanonicalPolyadicDecomposition() = default;

  /// \brief copy constructor
  CanonicalPolyadicDecomposition(CanonicalPolyadicDecomposition const&) = default;

  /// \brief move constructor
  CanonicalPolyadicDecomposition(CanonicalPolyadicDecomposition&&) = default;

  /// \brief copy assignment operator
  CanonicalPolyadicDecomposition& operator=(CanonicalPolyadicDecomposition const&) = default;

  /// \brief move assignment operator
  CanonicalPolyadicDecomposition& operator=(CanonicalPolyadicDecomposition&&) = default;

  /**
   * \brief Copy constructor for a different execution space.
   * \param[in] rhs CPD from a different execution space.
   */

  template <execution_space rhs_space>
  CanonicalPolyadicDecomposition(CanonicalPolyadicDecomposition<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        ALS_tolerance_relative(rhs.ALS_tolerance_relative),
        ALS_tolerance_absolute(rhs.ALS_tolerance_absolute),
        ALS_restarts(rhs.ALS_restarts),
        m_cores(rhs.m_cores),
        m_weights(rhs.m_weights)
  {
  }

  /**
   * \brief Copy assignment operator for a different execution space.
   * \param[in] rhs CPD from a different execution space.
   */

  template <execution_space rhs_space>
  CanonicalPolyadicDecomposition& operator=(CanonicalPolyadicDecomposition<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    ALS_tolerance_absolute = rhs.ALS_tolerance_absolute;
    ALS_tolerance_relative = rhs.ALS_tolerance_relative;
    ALS_restarts = rhs.ALS_restarts;
    m_cores = rhs.m_cores;
    m_weights = rhs.m_weights;
    return *this;
  }

  ~CanonicalPolyadicDecomposition() = default;

  /**
   * \brief Resizes the CPD extents and reinitializes the factors to zero.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   * \param[in] input_sizes New tensor extents.
   */

  void resize(::boba::Array<index_t, dimension> input_sizes)
  {
    checkpoint();
    boba_assert(input_sizes > 0, "invalid size");
    m_weights.resize({1});
    for (size_t d = 0; d < dimension; d++)
    {
      m_cores[d].resize({input_sizes[d], 1});
      m_cores[d].fill_with_zeros();
    }
    checkpoint();
  }

  /**
   * \brief Renames this CPD.
   * \param[in] new_name New name.
   */

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < dimension; d++)
    {
      this->m_cores[d].rename(m_name + "_core_" + std::to_string(d));
    }
  }

  // -------------------------------------------------------------------------------------
  // Printing
  // -------------------------------------------------------------------------------------

  /**
   * \brief Prints the CPD to cout, useful for debugging.
   * \param[in] indent Indentation depth.
   */

  void print(
    size_t indent = 1) const
  {
    std::cout << write_indent(indent) << m_name << std::endl
              << write_indent(indent + 1) << "rank = " << rank() << std::endl
              << write_indent(indent + 1) << "sizes = " << sizes_string() << std::endl;
    std::cout << std::endl;
    Vector<host_space, data_t> host_weights(m_weights);
    for (size_t r = 0; r < rank(); r++)
    {
      std::cout << write_indent(indent + 1) << "component " << r << std::endl
                << write_indent(indent + 2) << "weight = " << host_weights({r}) << std::endl;

      for (size_t d = 0; d < dimension; d++)
      {
        auto factor = boba::flatten(m_cores[d].extract_columns(r));
        std::cout << write_indent(indent + 2) << "factor vector " << d << std::endl;
        factor.print(indent + 3);
      }
    }
  }

  /**
   * \return A string describing the ranks.
   */

  [[nodiscard]]
  std::string ranks_string() const
  {
    return (" ( " + std::to_string(rank()) + " ) ");
  }

  /**
   * \return A string describing the tensor extents.
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
   * \return This object's name.
   */

  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * \return The number of cores.
   */

  static constexpr size_t get_number_cores()
  {
    return get_dimension();
  }

  /**
   * \return The dimension of the CPD.
   */

  static constexpr size_t get_dimension()
  {
    return dimension;
  }

  /**
   * \return The CPD rank.
   */

  [[nodiscard]]
  size_t get_rank() const noexcept
  {
    return this->m_weights.size();
  }

  /**
   * \return The total number of scalars used to define this CPD.
   */

  [[nodiscard]]
  size_t get_number_elements() const
  {
    size_t sum_elements = this->m_weights.size();
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += this->m_cores[d].get_number_elements();
    }
    return sum_elements;
  }

  /**
   * \return The product of the extents, which is the size of the uncompressed tensor this represents.
   */

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
   * \return The ratio of the full size over the compressed size, truncated to two digits.
   */

  [[nodiscard]]
  float compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) / static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  /**
   * \return The extents of the tensor that this CPD approximates.
   */

  [[nodiscard]]
  Array<size_t, dimension> sizes() const noexcept
  {
    ::boba::Array<size_t, dimension> this_sizes{0};
    for (size_t d = 0; d < dimension; d++)
    {
      this_sizes[d] = sizes(d);
    }
    return this_sizes;
  }

  /**
   * \return The rank of the CPD.
   */

  [[nodiscard]]
  size_t rank() const noexcept
  {
    return this->m_weights.size();
  }

  /**
   * \return The size of the `d`th mode.
   */

  index_t sizes(index_t d) const noexcept
  {
    return this->m_cores[d].sizes(0);
  }

  /**
   * \return The CPD weights.
   */

  [[nodiscard]]
  Vector<space, data_t>& weights() noexcept
  {
    return this->m_weights;
  }

  /**
   * \return The CPD weights.
   */

  [[nodiscard]]
  Vector<space, data_t> const& weights() const noexcept
  {
    return this->m_weights;
  }

  /**
   * \brief Normalizes each rank-one factor column and folds the removed scale into the weights.
   *
   * After this call, each nonzero column of every factor matrix has unit Frobenius norm and
   * the corresponding scale is stored in `m_weights`.
   */

  void normalize_factors_into_weights()
  {
    BOBA_CALI_MARK

    Vector<host_space, data_t> host_weights(m_weights);

    for (index_t r = 0; r < rank(); r++)
    {
      auto weight = host_weights({r});

      for (size_t d = 0; d < dimension; d++)
      {
        auto column = m_cores[d].extract_columns(r);
        auto column_norm = ::boba::norm_frobenius(column);

        if (column_norm <= boba::epsilon<real_data_t>())
        {
          weight = ::boba::PotentiallyComplex<data_t>::value(0.0);
          break;
        }

        auto inverse_column_norm =
          ::boba::PotentiallyComplex<data_t>::value(real_data_t(1.0) / column_norm);
        column *= inverse_column_norm;

        m_cores[d].replace_submatrix(
          {0_z, m_cores[d].rows()},
          {r, r + 1},
          column);

        weight *= ::boba::PotentiallyComplex<data_t>::value(column_norm);
      }

      host_weights({r}) = weight;
    }

    m_weights = host_weights;
  }

  // -------------------------------------------------------------------------------------
  // Section: Views
  // -------------------------------------------------------------------------------------

  using core_view_type = ::boba::TensorView<::boba::DefaultAccessor<data_t>, 2>;

  /**
   * \return An array of views of the CPD cores.
   * Don't forget to grab the weights if you need those.
   */

  ::boba::Array<core_view_type, dimension> get_core_views() const
  {
    ::boba::Array<core_view_type, dimension> views;
    for (size_t d = 0; d < dimension; d++)
    {
      views[d] = this->m_cores[d].view();
    }
    return views;
  }

  using core_const_view_type = ::boba::TensorView<::boba::DefaultAccessor<data_t const>, 2>;

  /**
   * \return An array of const views of the CPD cores.
   * Don't forget to grab the weights if you need those.
   */

  ::boba::Array<core_const_view_type, dimension> get_core_const_views() const
  {
    ::boba::Array<core_const_view_type, dimension> views;
    for (size_t d = 0; d < dimension; d++)
    {
      views[d] = this->m_cores[d].const_view();
    }
    return views;
  }

  // -------------------------------------------------------------------------------------
  // Section: Write
  // -------------------------------------------------------------------------------------

  /**
   * \brief Sets this into a rank-one CPD filled with `x`, equivalent to a tensor filled with `x`.
   */

  void fill_with(data_t x)
  {
    BOBA_CALI_MARK
    for (size_t d = 0; d < dimension; d++)
    {
      auto core_size = this->m_cores[d].sizes(0);
      this->m_cores[d].resize({core_size, 1});
      this->m_cores[d].fill_with(1.0);
    }
    this->m_weights.resize({1});
    this->m_weights.fill_with(x);
  }

  /**
   * \brief Sets this into a rank-one CPD filled with a real scalar.
   * This overload is used when the stored data type is complex.
   */

  template <typename _real_data_t>
    requires IsRealDataType<_real_data_t, data_t>
  void fill_with(_real_data_t x)
  {
    BOBA_CALI_MARK
    data_t complex_x{x, static_cast<_real_data_t>(0)};
    fill_with(complex_x);
  }

  /**
   * \brief Sets this into a rank-one CPD filled with zero, equivalent to a tensor filled with zero.
   */

  void fill_with_zeros()
  {
    BOBA_CALI_MARK
    for (size_t d = 0; d < dimension; d++)
    {
      auto core_size = this->m_cores[d].sizes(0);
      this->m_cores[d].resize({core_size, 1});
      this->m_cores[d].fill_with_zeros();
    }
    this->m_weights.resize({1});
    this->m_weights.fill_with_zeros();
  }

  // -------------------------------------------------------------------------------------
  // Section: Unroll
  // -------------------------------------------------------------------------------------

  /**
   * \param[in] mid Multiindex.
   * \return This CPD evaluated at `mid`.
   */

  data_t unroll_value(
    const Array<size_t, dimension> mid) const
  {
    detail::ignore(mid);
    boba_error("Not yet implemented.");
    return data_t{};
  }

  /**
   * \brief Generates the full subtensor inscribed by the extents described by `start` and `end`.
   * \param[in] start Starting multiindex.
   * \param[in] end Ending multiindex, exclusive.
   * \return A full tensor of size `product(end - start)` generated from this decomposition.
   */

  Tensor<dimension, space, data_t> unroll_subtensor(
    const Array<size_t, dimension>& start,
    const Array<size_t, dimension>& end) const
  {
    BOBA_CALI_MARK
    checkpoint();

    boba::Tensor<dimension, space, data_t> output(end - start);
    auto output_view = output.view();

    auto weights_view = m_weights.const_view();
    auto core_views = this->get_core_const_views();
    auto local_rank = rank();
    auto local_dimension = get_dimension();

    // To compute each entry of the output tensor:
    loop<space, 1>(output_view.size(),
                   [=] __boba_host_device__(size_t id)
    {
      auto mid = output_view.multiindex(id);
      data_t entry{};
      for (size_t r = 0; r < local_rank; r++)
      {
        auto weight = weights_view(r);
        for (size_t d = 0; d < local_dimension; d++)
        {
          auto core_view = core_views[d];
          weight *= core_view({start[d] + mid[d], r});
        }
        entry += weight;
      }
      output_view(mid) = entry;
    });
    return output;
  }

  /**
   * \return The full tensor that this CPD represents.
   */

  boba::Tensor<dimension, space, data_t> decompress() const
  {
    BOBA_CALI_MARK
    return unroll_subtensor(::boba::filled_array<dimension>(0_z), sizes());
  }

  // -------------------------------------------------------------------------------------
  // Section: Insert dimension
  // -------------------------------------------------------------------------------------

  /**
   * \brief Maps a CPD to a higher-dimensional CPD.
   * Example: say you have a CPD of cores A B D. Calling this function with
   * `dimension_to_insert = 2` corresponds to inserting before D.
   * The new CPD will be A B C D, where C = 1 in the long dimension.
   * This implies the new CPD's corresponding tensor is equal to the old one modulo a reshape.
   * \return The higher-dimensional CPD.
   */

  CanonicalPolyadicDecomposition<dimension + 1, space, data_t> insert_dimension(
    const size_t dimension_to_insert,
    const size_t new_size = 1) const
  {
    detail::ignore(dimension_to_insert);
    detail::ignore(new_size);
    BOBA_CALI_MARK
    checkpoint();
    boba_error("Not yet implemented");
    return CanonicalPolyadicDecomposition<dimension + 1, space, data_t>{};
  }

  // -------------------------------------------------------------------------------------
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief Adds the input CPD to this one.
   */

  void cpd_add(CanonicalPolyadicDecomposition const& input)
  {
    BOBA_CALI_MARK
    detail::ignore(input);
    checkpoint();
    boba_error("Not yet implemented");
    // PTG - I think we just concatenate weight vectors and each of the core columns
  }

  /// \brief this += rhs

  CanonicalPolyadicDecomposition& operator+=(CanonicalPolyadicDecomposition const& rhs)
  {
    BOBA_CALI_MARK
    this->cpd_add(rhs);
    return *this;
  }

  /// \return this + rhs

  CanonicalPolyadicDecomposition operator+(CanonicalPolyadicDecomposition const& rhs) const
  {
    BOBA_CALI_MARK
    CanonicalPolyadicDecomposition output{*this};
    output += rhs;
    return output;
  }

  /// \brief this -= rhs

  CanonicalPolyadicDecomposition& operator-=(CanonicalPolyadicDecomposition const& rhs)
  {
    BOBA_CALI_MARK
    CanonicalPolyadicDecomposition rhs_minus{rhs};
    rhs_minus *= -1.0;
    this->cpd_add(rhs_minus);
    return *this;
  }

  /// \return this - rhs

  CanonicalPolyadicDecomposition operator-(CanonicalPolyadicDecomposition const& rhs) const
  {
    BOBA_CALI_MARK
    CanonicalPolyadicDecomposition output{*this};
    output -= rhs;
    return output;
  }

  /// \return -this

  CanonicalPolyadicDecomposition operator-() const
  {
    BOBA_CALI_MARK
    CanonicalPolyadicDecomposition output{*this};
    output *= -1.0;
    return output;
  }

  /// \brief this *= scalar

  CanonicalPolyadicDecomposition& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    multiply_scalar(scalar);
    return *this;
  }

  /// \return this * scalar

  CanonicalPolyadicDecomposition operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    CanonicalPolyadicDecomposition output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  /**
   * \brief Computes an update that uses a Khatri-Rao-like product, skipping one mode.
   * This is useful for finding the solution of an ALS step in CP (Canonical Polyadic) decomposition explicitly.
   * Original reference:
   * Khatri, C.G., & Rao, C.R. (1968). "Solutions to some functional equations and their applications
   * to characterization of probability distributions." Sankhya: The Indian Journal of Statistics, Series A, 30(2), 167–180.
   *
   * \param[in] skip Mode to skip.
   * \return Khatri-Rao product result.
   */

  Matrix<space, data_t> CPD_KhatriRao(size_t skip)
  {
    // Build mats, skipping the 'skip' index
    std::vector<::boba::Matrix<space, data_t>> mats;

    for (size_t i = 0; i < dimension; ++i)
    {
      if (i != skip)
      {
        mats.push_back(m_cores[i]);
      }
    }
    if (mats.size() == 0)
    {
      return ::boba::Matrix<space, data_t>();
    }

    auto output = mats[0];

    for (size_t i = 1; i < mats.size(); i++)
    {
      auto temp = output;
      output = boba::mode_n_tensor_product(temp, mats[i], 0);
    }

    auto output_view = output.view();
    auto weights_view = m_weights.const_view();
    ::boba::loop<space, 2>(output.sizes(),
                           [=] __boba_host_device__(Array<index_t, 2> ij)
    {
      output_view(ij) *= weights_view(ij[1]);
    });

    return output;
  }

  /**
   * \brief Performs the ALS_step_direct update.
   * An alternative least-squared (ALS) optimization step by fixing factor matrices
   * other than "factors[skip]" and reducing the Frobenius norm difference between
   * CP decomposition and original tensor. Using Khatri-Rao product for direct solution
   * References:
   * Khatri, C. G., & Rao, C. R. (1968). Solutions to Some Functional Equations and Their Applications to
   * Characterization of Probability Distributions.Proceedings of the Cambridge Philosophical Society, 64(3), 507–516.
   *
   * PARAFAC: Parallel factor analysis, R.A. Harshman, 1970, UCLA Technical Report
   *
   * Tensor decompositions and applications, Tamara G. Kolda, Brett W. Bader, 2009, SIAM Review (DOI: 10.1137/07070111X)
   *
   * \param[in] input Tensor to approximate.
   * \param[in] skip Current update direction.
   */

  void ALS_step_direct(const Tensor<dimension, space, data_t>& input, size_t skip)
  {
    checkpoint();

    // 1st step. unfold the tensor along the skipped mode
    auto tensor_unfold_right = boba::unfold(input, skip);
    checkpoint();

    // 2nd step. compute the Khatri-Rao product of the factors except the one corresponding to "skip"
    ::boba::Matrix<space, data_t> KR = CPD_KhatriRao(skip);
    checkpoint();

    // 3rd step. compute (KR^T).(KR)
    auto gram = tensor_contraction<1>({"r", "c"}, KR, {"r", "p"}, KR, {"c", "p"});
    checkpoint();

    // R is the desired CP rank
    auto R = gram.sizes(0);
    boba_always_assert_equal(gram.sizes(0), gram.sizes(1), "gram is a square matrix");

    // define an identity matrix of size R x R
    ::boba::Matrix<space, data_t> eye({R, R});
    eye.set_to_identity_matrix();

    // A small diagonal damping term helps avoid singular or nearly singular
    // normal equations during ALS updates.
    data_t lambda = static_cast<data_t>(1.0e-8);
    auto gram_plus = reshape_to_matrix(gram + lambda * eye, {R, R});
    auto gram_inv = ::boba::backsolve(gram_plus, eye);
    checkpoint();

    // replace factors[skip] with the update
    auto tensor_KR = tensor_contraction<1>({"r", "c"}, tensor_unfold_right, {"c", "p"}, KR, {"r", "p"});
    auto factor_update = tensor_contraction<1>({"r", "p"}, tensor_KR, {"p", "c"}, gram_inv, {"r", "c"});
    checkpoint();
    m_cores[skip] = reshape_to_matrix(factor_update, factor_update.sizes());
  }

  /**
   * \brief Performs ALS_step_direct repeatedly to generate this CPD approximation at a fixed rank.
   * Since this algorithm uses random initialization, it is not deterministic.
   * If one ALS run does not satisfy the convergence tolerances within `ALS_iters`,
   * the decomposition is reinitialized and retried up to `ALS_restarts` times.
   *
   * \param[in] input Tensor to approximate.
   * \param[in] rank Rank to which this CPD will approximate `input`.
   * \return The iteration count of the converged ALS run, or `ALS_iters` if all restarts fail.
   */

  size_t compress(const boba::Tensor<dimension, space, data_t>& input,
                  size_t rank)
  {
    BOBA_CALI_MARK
    checkpoint();

    if constexpr (dimension == 1_z)
    {
      m_cores[0].resize({sizes(0), 1_z});
      m_cores[0].reshape(input);
      m_weights.resize({1_z});
      m_weights.fill_with(1.0);
      return 0;
    }

    auto input_norm = ::boba::norm_frobenius(input);
    auto initialize_random_factors = [&]()
    {
      checkpoint();
      for (size_t d = 0; d < dimension; d++)
      {
        m_cores[d].resize({sizes(d), rank});
        m_cores[d].fill_with_random();
      }
      m_weights.resize({rank});
      m_weights.fill_with(1.0);
      normalize_factors_into_weights();
    };

    CanonicalPolyadicDecomposition<dimension, space, data_t> best_cpd(input.sizes());
    auto best_l2_err = boba::highest_value<real_data_t>();
    auto best_l2_err_relative = boba::highest_value<real_data_t>();
    bool have_best = false;

    checkpoint();
    for (size_t restart = 0; restart <= ALS_restarts; restart++)
    {
      initialize_random_factors();

      for (size_t iter = 0; iter < ALS_iters; iter++)
      {
        for (size_t skip = 0; skip < dimension; skip++)
        {
          ALS_step_direct(input, skip);
        }

        normalize_factors_into_weights();

        auto l2_err = norm_difference_frobenius(input, this->decompress());
        auto l2_err_relative = l2_err / input_norm;

        if (!have_best || l2_err_relative < best_l2_err_relative)
        {
          best_cpd = *this;
          best_l2_err = l2_err;
          best_l2_err_relative = l2_err_relative;
          have_best = true;
        }

        if (l2_err < ALS_tolerance_absolute)
        {
          return iter;
        }
        if (l2_err_relative < ALS_tolerance_relative)
        {
          return iter;
        }
      }
    }

    if (have_best)
    {
      *this = best_cpd;
    }

    return ALS_iters;
  }

  /**
   * \brief Rounds this CPD.
   */

  void round()
  {
    BOBA_CALI_MARK
    checkpoint();
    boba_error("Not yet implemented.");
    if constexpr (dimension == 1)
    {
      // ...
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Multiply scalar
  // -------------------------------------------------------------------------------------

  /**
   * \brief Multiplies the CPD by `x`.
   */

  void multiply_scalar(data_t x)
  {
    checkpoint();
    this->m_weights *= x;
  }

  // -------------------------------------------------------------------------------------
  // Section: Reduction
  // -------------------------------------------------------------------------------------

  /**
   * \return Inner product of this and the input CPD.
   */

  data_t inner_product(const CanonicalPolyadicDecomposition<dimension, space, data_t>& input) const
  {
    BOBA_CALI_MARK
    checkpoint();

    // Example logic, given sets of vectors A_i, B_i, C_i, and scalar weights w_i
    // < \sum\limits_i w_i A_i x B_i x C_i , \sum\limits_j w_j A_j x B_j x C_j >
    // \sum\limits_j\sum\limits_i w_i * w_j * < A_i x B_i x C_i , A_j x B_j x C_j >
    // \sum\limits_j\sum\limits_i w_i * w_j * (A_i^T x B_i^T x C_i^T) (A_j x B_j x C_j)
    // \sum\limits_j\sum\limits_i w_i * w_j * (A_i^T A_j) x (B_i^T B_j) x (C_i^T C_j)
    // Note (A_i^T A_j) is a scalar for a given i, j
    // \sum\limits_j\sum\limits_i w_i * w_j * (A_i^T A_j)(B_i^T B_j)(C_i^T C_j)
    // w_i (A_i^T A_j) . (B_i^T B_j) . (C_i^T C_j) w_j
    // where . is the Hadamard product

    auto rankwise_inner_products = this->m_cores[0].transpose() * input.m_cores[0];
    for (size_t d = 1; d < dimension; d++)
    {
      auto temp = elementwise_product(rankwise_inner_products, this->m_cores[d].transpose() * input.m_cores[d]);
      rankwise_inner_products = temp;
    }

    return inner_product(this->m_weights, rankwise_inner_products * input.m_weights);
  }

  // -------------------------------------------------------------------------------------
  // Section: Extract rank one terms
  // -------------------------------------------------------------------------------------
  /**
   * \param[in] rank Rank to extract.
   * \return The rank-one CPD corresponding to the given rank.
   */

  [[nodiscard]]
  CanonicalPolyadicDecomposition<dimension, space, data_t> extract_rank_one_cpd(
    const index_t rank)
  {
    BOBA_CALI_MARK

    CanonicalPolyadicDecomposition<dimension, space, data_t> rank_one_cpd(sizes());

    Vector<host_space, data_t> host_weights(m_weights);

    auto weight = host_weights({rank});

    for (size_t d = 0; d < dimension; d++)
    {
      rank_one_cpd.m_cores[d] = m_cores[d].extract_columns(rank);
    }
    rank_one_cpd.m_weights.resize({1});
    rank_one_cpd.m_weights.fill_with(weight);

    return rank_one_cpd;
  }
};

} // namespace boba
