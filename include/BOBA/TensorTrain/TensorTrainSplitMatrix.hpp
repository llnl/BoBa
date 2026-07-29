// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tensor train matrix, split row-col storage.
 * This is a more complex tensor train object, it is recommended that you fully understand tensor trains and tensor train matrices before reading about tensor-train split matrices.
\verbatim
 Example for dimension = 3

    (1, r_0)       (r_0, r_1)       (r_1, r_2)       (r_2, r_3)
 ___  ____     __                 ___  ____     ___
 \    |  |     \    ________      \    |  |     \    ________
  >   |  | (x)  >   |_______| (x)  >   |  | (x)  >   |_______| (x)  ....
 /__  |__|     /__                /__  |__|     /__
 r_0  rows[0]  r_1   cols[0]      r_2  rows[1]  r_3   cols[1]     rows[2]...
\endverbatim
 */

template <size_t true_dimension, ::boba::execution_space space, typename _data_t>
struct TensorTrainSplitMatrix
{
  // -------------------------------------------------------------------------------------
  // Typedefs
  // -------------------------------------------------------------------------------------

  using data_t = _data_t;

  // -------------------------------------------------------------------------------------
  // Member objects
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Split ttmatrices are twice as long as the corresponding ttmatrix
   */

  constexpr static size_t dimension = 2 * true_dimension;
  std::string m_name = "TensorTrainSplitMatrix";

  /**
   * \brief
   * Number of rows in each corresponding 1d matrix
   */

  Array<size_t, true_dimension> m_rows = filled_array<true_dimension, size_t>(0);
  Array<size_t, true_dimension> m_cols = filled_array<true_dimension, size_t>(0);
  ::boba::Array<size_t, dimension + 1> max_ranks = ::boba::filled_array<dimension + 1>(highest_value<size_t>());

  data_t svd_tolerance_relative = 1.0e-07;
  data_t svd_tolerance_absolute = 1.0e-12;

  Array<Tensor<3, space, data_t>, dimension> cores;

  // -------------------------------------------------------------------------------------
  // Constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Construct a split ttmatrix from a conventional ttm. See from_ttm.
   */

  TensorTrainSplitMatrix(
    TensorTrainMatrix<true_dimension, space, data_t>& tt_matrix)
  {
    from_ttm(tt_matrix);
  }

  /**
   * \brief
   * Construct a split ttmatrix from a conventional ttm.
   * This is accomplished by
   * (1) Computing SVDs of every submatrix in the ttm.
   * (2) Treat each SVD as a ttmatrix where the first size is nx1 and the second size is 1xm.
   * (3) Add this mini ttmatrix to the split ttm.
   */

  void from_ttm(
    TensorTrainMatrix<true_dimension, space, data_t>& tt_matrix)
  {
    checkpoint();
    // TODO<optimization> clean up and optimize this function
    m_rows = tt_matrix.core_rows();
    m_cols = tt_matrix.core_cols();
    for (size_t d = 0; d < true_dimension; d++)
    {
      cores[2 * d].resize({1, m_rows[d], 1});
      cores[2 * d + 1].resize({1, m_cols[d], 1});
      cores[2 * d].fill_with_zeros();
      cores[2 * d + 1].fill_with_zeros();
    }

    SVD<space, data_t> core_svd;
    for (size_t d = 0; d < true_dimension; d++)
    {
      size_t ranks_left = tt_matrix.get_ranks_left(d);
      size_t ranks_right = tt_matrix.get_ranks_right(d);
      checkpoint();
      size_t new_common = boba::min(m_rows[d], m_cols[d]);
      size_t new_common_ranks = new_common * ranks_left * ranks_right;

      auto common_sizes_view = ::boba::Multiindexer<3>({ranks_right, new_common, ranks_left});
      auto left_sizes_view = ::boba::Multiindexer<1>({ranks_left});
      auto right_sizes_view = ::boba::Multiindexer<1>({ranks_right});

      cores[2 * d].resize({left_sizes_view.size(), m_rows[d], new_common_ranks});
      cores[2 * d + 1].resize({new_common_ranks, m_cols[d], right_sizes_view.size()});
      cores[2 * d].fill_with_zeros();
      cores[2 * d + 1].fill_with_zeros();

      for (size_t rank_left = 0; rank_left < ranks_left; rank_left++)
      {
        for (size_t rank_right = 0; rank_right < ranks_right; rank_right++)
        {
          auto matrix_1d = tt_matrix.get_matrix(d, rank_left, rank_right);
          matrix_1d.rename("matrix_1d");
          checkpoint();
          core_svd(matrix_1d);
          apply_as_diagonal_right_in_place(core_svd.S, core_svd.U);
          checkpoint();
          size_t significant_singular_values = core_svd.significant_singular_values;

          boba_always_assert_le(
            significant_singular_values, new_common, "significant_singular_values should be less than the minimum of rows and cols");

          auto row_core_view = cores[2 * d].view();
          auto col_core_view = cores[2 * d + 1].view();
          auto U_view = core_svd.U.view();
          auto V_view = core_svd.V.view();

          checkpoint();
          ::boba::loop<space, 2>({m_rows[d], new_common},
                                 [=] __boba_host_device__(Array<size_t, 2> ir)
          {
            size_t i = ir[0];
            size_t r = ir[1];
            auto value = data_t(0.0);
            size_t index_common = common_sizes_view.index({rank_right, r, rank_left});
            size_t index_left = left_sizes_view.index({rank_left});

            if (r < significant_singular_values)
            {
              value = U_view({i, r});
            }
            row_core_view({index_left, i, index_common}) = value;
          });
          checkpoint();
          ::boba::loop<space, 2>({m_cols[d], new_common},
                                 [=] __boba_host_device__(Array<size_t, 2> ir)
          {
            size_t i = ir[0];
            size_t r = ir[1];
            auto value = data_t(0.0);
            size_t index_common = common_sizes_view.index({rank_right, r, rank_left});
            size_t index_right = right_sizes_view.index({rank_right});
            if (r < significant_singular_values)
            {
              value = V_view({i, r});
            }
            col_core_view({index_common, i, index_right}) = value;
          });
          checkpoint();
        }
      }
    }

    this->round();
  }

  /**
   * \brief
   * Construct from array of sizes and max ranks by calling resize. Instantiated object is the zero tensor train.
   */

  TensorTrainSplitMatrix(
    const Array<size_t, true_dimension> input_rows,
    const Array<size_t, true_dimension> input_cols)
  {
    this->resize(input_rows, input_cols);
  }

  /**
   * \brief
   * Resizes the split tensor train matrix metadata.
   * Unlike Tensor::resize, this does not preserve overlapping entries.
   */

  void resize(
    const Array<size_t, true_dimension> input_rows,
    const Array<size_t, true_dimension> input_cols)
  {
    for (size_t d = 0; d < true_dimension; d++)
    {
      boba_assert(input_rows[d] > 0, "invalid size");
      boba_assert(input_cols[d] > 0, "invalid size");
      m_rows[d] = input_rows[d];
      m_cols[d] = input_cols[d];
    }
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
    for (size_t d = 0; d < true_dimension; d++)
    {
      this->cores[2 * d].rename(m_name + "_row_core_" + std::to_string(d));
      this->cores[2 * d + 1].rename(m_name + "_col_core_" + std::to_string(d));
    }
  }

  // Default constructor
  TensorTrainSplitMatrix() = default;

  // Copy constructor
  TensorTrainSplitMatrix(TensorTrainSplitMatrix const&) = default;

  // Move constructor
  TensorTrainSplitMatrix(TensorTrainSplitMatrix&&) = default;

  // Copy assignment
  TensorTrainSplitMatrix& operator=(TensorTrainSplitMatrix const&) = default;

  // Move assignment
  TensorTrainSplitMatrix& operator=(TensorTrainSplitMatrix&&) = default;

  // -------------------------------------------------------------------------------------
  // Section: Printing
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Returns an array of the ranks
   */

  Array<size_t, dimension + 1> ranks() const
  {
    Array<size_t, dimension + 1> rarr;
    for (size_t d = 0; d < get_number_cores(); d++)
    {
      rarr[d] = get_ranks_left(d);
    }
    rarr[dimension] = get_ranks_right(dimension - 1);
    return rarr;
  }

  /**
   * \brief
   * A string describing the tensor train ranks
   */

  std::string ranks_string() const
  {
    return (" ( " + make_delimited_string(ranks()) + " ) ");
  }

  void print(
    size_t indent = 1,
    data_t tolerance = 0.0) const
  {
    std::cout << write_indent(indent) << m_name << std::endl;
    for (size_t d = 0; d < dimension; d++)
    {
      std::cout << write_indent(indent + 1) << "core " << d << std::endl;
      cores[d].print(indent + 2, tolerance);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Decompress
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Maps this to a ttmatrix. This does to by fusing corresponding rows and cols.
   */

  TensorTrainMatrix<true_dimension, space, data_t> decompress_to_ttm() const
  {
    TensorTrainMatrix<true_dimension, space, data_t> ttm(core_rows(), core_cols());
    boba_print(get_ranks_left());
    boba_print(get_ranks_right());

    for (size_t d = 0; d < true_dimension; d++)
    {
      checkpoint();
      boba_always_assert_equal(get_ranks_right(2 * d), get_ranks_left(2 * d + 1), "Incompatible ranks");
      checkpoint();
      ttm.cores[d] = tensor_contraction_single_index(
        {"row_l", "row", "common"}, cores[2 * d], {"common", "col", "col_r"}, cores[2 * d + 1], {"row_l", "row", "col", "col_r"});
    }
    return ttm;
  }

  /**
   * \brief
   * Maps this to a matrix.
   */

  Matrix<space, data_t> decompress() const
  {
    checkpoint();
    auto ttm = decompress_to_ttm();
    ttm.rename("ttm_from_split");
    checkpoint();
    return ttm.decompress();
  }

  // -------------------------------------------------------------------------------------
  // Section: Getters
  // -------------------------------------------------------------------------------------

  size_t core_rows(size_t i) const
  {
    boba_always_assert_nonnegative(i, "invalid core number");
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[2 * i].sizes(1);
  }

  size_t core_cols(size_t i) const
  {
    boba_always_assert_nonnegative(i, "invalid core number");
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[2 * i + 1].sizes(1);
  }

  Array<size_t, true_dimension> core_rows() const
  {
    return m_rows;
  }

  Array<size_t, true_dimension> core_cols() const
  {
    return m_cols;
  }

  /**
   * \brief
   * Rows of the corresponding full matrix.
   */

  size_t rows() const
  {
    return product(m_rows);
  }

  /**
   * \brief
   * Cols of the corresponding full matrix.
   */

  size_t cols() const
  {
    return product(m_cols);
  }

  /**
   * \brief Returns number of cores
   */

  static constexpr size_t get_number_cores()
  {
    return get_dimension();
  }

  /**
   * \brief Returns number of cores
   */

  static constexpr size_t get_dimension()
  {
    return dimension;
  }

  size_t get_ranks_left(size_t i) const
  {
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(0);
  }

  size_t get_ranks_right(size_t i) const
  {
    boba_always_assert(i < dimension, "invalid core number");
    return this->cores[i].sizes(2);
  }

  auto get_ranks_left() const
  {
    Array<size_t, dimension + 1> ranks;
    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] = get_ranks_left(d);
    }
    return ranks;
  }

  auto get_ranks_right() const
  {
    Array<size_t, dimension + 1> ranks;
    for (size_t d = 0; d < dimension; d++)
    {
      ranks[d] = get_ranks_right(d);
    }
    return ranks;
  }

  size_t get_number_elements() const
  {
    size_t sum_elements = 0;
    for (size_t d = 0; d < dimension; d++)
    {
      sum_elements += this->cores[d].get_number_elements();
    }
    return sum_elements;
  }

  size_t get_sizes(size_t d) const
  {
    boba_always_assert_nonnegative(d, "invalid core number");
    boba_always_assert_lt(d, dimension, "invalid core number");
    return ((d % 2) == 0) ? m_rows[d / 2] : m_cols[d / 2];
  }

  /**
   * \brief
   * Cols of the corresponding full matrix.
   */

  size_t ranks_bound()
  {
    size_t bound = 0;
    for (size_t d = 0; d < dimension; d++)
    {
      bound = boba::max(bound, get_ranks_left(d));
      bound = boba::max(bound, get_ranks_right(d));
    }
    return bound;
  }

  /**
   * \brief
   * Number of elements of the corresponding full matrix.
   */

  size_t get_full_size() const
  {
    size_t full_size = 1;
    for (size_t d = 0; d < true_dimension; d++)
    {
      full_size *= m_rows[d];
      full_size *= m_cols[d];
    }
    return full_size;
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
    set_to_rank_one_scalar_core(this->cores[0], x);
    for (size_t d = 1; d < dimension; d++)
    {
      set_to_rank_one_scalar_core(this->cores[d], 1.0);
    }
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
  // Section: Addition
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tensor train addition of this and subtrain, where subtrain is smaller than this.
   * The subtrain must have an offset and size for each dimension such that offset+size fits into this.
   * Zeros are padded to subtrain so that addition is valid.
   */

  void add_subtrain(
    TensorTrainSplitMatrix const& subtrain,
    const ::boba::Array<size_t, true_dimension>& initial_row,
    const ::boba::Array<size_t, true_dimension>& initial_col)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < true_dimension; d++)
    {
      boba_assert(core_rows()[d] >= (initial_row[d] + subtrain.core_rows()[d]), "Subtrain size exceeds parent");
      boba_assert(core_cols()[d] >= (initial_col[d] + subtrain.core_cols()[d]), "Subtrain size exceeds parent");
    }

    // Add middle row cores
    checkpoint();
    for (size_t di = 0; di < dimension; di++)
    {
      bool is_row = ((di % 2) == 0);
      auto row_or_col_id = di / 2;
      auto first_index = is_row ? initial_row[row_or_col_id] : initial_col[row_or_col_id];
      add_subcore(this->cores.at(di), subtrain.cores.at(di), di, dimension, first_index);
    }
  }

  /**
   * \brief
   * Tensor train addition of this and input.
   */

  void TensorTrainSplitMatrix_add(TensorTrainSplitMatrix const& input)
  {
    BOBA_CALI_MARK
    checkpoint();
    for (size_t d = 0; d < true_dimension; d++)
    {
      boba_assert_equal(input.core_rows()[d], this->core_rows()[d], "incompatible sizes");
      boba_assert_equal(input.core_cols()[d], this->core_cols()[d], "incompatible sizes");
    }
    checkpoint();

    auto initial_row = ::boba::filled_array<true_dimension>(0_z);
    auto initial_col = ::boba::filled_array<true_dimension>(0_z);

    checkpoint();
    this->add_subtrain(input, initial_row, initial_col);
    checkpoint();
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrainSplitMatrix& operator+=(TensorTrainSplitMatrix const& rhs)
  {
    BOBA_CALI_MARK
    this->TensorTrainSplitMatrix_add(rhs);
    return *this;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TensorTrainSplitMatrix operator+(TensorTrainSplitMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix output{*this};
    output += rhs;
    return output;
  }

  /**
   * \brief
   * this += rhs
   */

  TensorTrainSplitMatrix& operator-=(TensorTrainSplitMatrix const& rhs)
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix rhs_minus{rhs};
    rhs_minus *= -1.0;
    this->TensorTrainSplitMatrix_add(rhs_minus);
    return *this;
  }

  /**
   * \brief
   * negation operator
   */

  TensorTrainSplitMatrix operator-()
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix output{*this};
    output *= -1.0;
    return output;
  }

  /**
   * \brief
   * output = this + rhs
   */

  TensorTrainSplitMatrix operator-(TensorTrainSplitMatrix const& rhs) const
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix output{*this};
    output -= rhs;
    return output;
  }

  /**
   * \brief
   * this *= input, using default methods
   */

  TensorTrainSplitMatrix operator*=(TensorTrainSplitMatrix const& input)
  {
    BOBA_CALI_MARK
    this->TensorTrainSplitMatrix_right_multiply(input, false);
    return *this;
  }

  /**
   * \brief
   * output = this * input
   */

  TensorTrainSplitMatrix operator*(TensorTrainSplitMatrix const& input) const
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix output{*this};
    output *= input;
    return output;
  }

  /**
   * \brief
   * output = this * input
   */

  TensorTrain<true_dimension, space, data_t> operator*(TensorTrain<true_dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    return this->TensorTrainMatrix_vector_multiply(input);
  }

  /**
   * \brief
   * this *= scalar
   */

  TensorTrainSplitMatrix& operator*=(data_t const scalar)
  {
    BOBA_CALI_MARK
    this->multiply_scalar(scalar);
    return *this;
  }

  /**
   * \brief
   * output = this * scalar
   */

  TensorTrainSplitMatrix operator*(data_t const scalar) const
  {
    BOBA_CALI_MARK
    TensorTrainSplitMatrix output{*this};
    output *= scalar;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matrix-Matrix Multiply
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Tensor train Matrix-Matrix Multiply for split ttmatrices.
   */

  void TensorTrainSplitMatrix_right_multiply(
    const TensorTrainSplitMatrix& input,
    bool use_inner_product_svd)
  {
    BOBA_CALI_MARK
    checkpoint();

    Array<boba::QR<space, data_t>, dimension> qr;
    boba::Matrix<space, data_t> unfold_right;

    boba_always_assert(not(use_inner_product_svd), "use_inner_product_svd is deprecated");

    for (size_t true_d = 0; true_d < true_dimension; true_d++)
    {
      const size_t row_id = 2 * true_d;
      const size_t col_id = 2 * true_d + 1;

      //
      BOBA_CALI_BEGIN("copy_input_data");
      //
      checkpoint();
      size_t common_dimension = input.core_rows(true_d);

      checkpoint();
      size_t input_rows_ranks_left = input.get_ranks_left(row_id);
      size_t input_common_ranks = input.get_ranks_right(row_id);
      size_t input_cols_ranks_right = input.get_ranks_right(col_id);
      boba_always_assert_equal(
        input_common_ranks, input.get_ranks_left(col_id), "ranks mismatch");

      //
      BOBA_CALI_SWITCH("copy_input_data", "copy_old_data");
      //
      size_t old_rows_ranks_left = get_ranks_left(row_id);
      size_t old_common_ranks = get_ranks_right(row_id);
      size_t old_cols_ranks_right = get_ranks_right(col_id);
      boba_always_assert_equal(
        old_common_ranks,
        get_ranks_left(col_id),
        "ranks mismatch");

      checkpoint();
      auto old_row_core = this->cores[row_id];
      auto old_col_core = this->cores[col_id];

      //
      BOBA_CALI_SWITCH("copy_old_data", "compute_new_dimensions");
      // Compute new dimensions
      //
      size_t new_rows = this->core_rows(true_d);
      size_t new_cols = input.core_cols(true_d);

      auto left_row_view = ::boba::Multiindexer<2>({input_rows_ranks_left, old_rows_ranks_left});
      size_t new_rows_ranks_left = left_row_view.size();

      auto right_col_view = ::boba::Multiindexer<2>({input_cols_ranks_right, old_cols_ranks_right});
      size_t new_cols_ranks_right = right_col_view.size();

      auto new_common_view = ::boba::Multiindexer<2>({input_common_ranks, old_cols_ranks_right});
      size_t new_common_size = new_common_view.size();

      checkpoint();
      if ((new_rows_ranks_left == 0_z) || (new_cols_ranks_right == 0_z) || (new_common_size == 0_z))
      {
        BOBA_CALI_END("compute_new_dimensions");
        checkpoint();
        this->fill_with_zeros();
        return;
      }

      auto old_row_core_view = old_row_core.const_view();
      auto old_col_core_view = old_col_core.const_view();

      auto input_row_core_view = input.cores[row_id].const_view();
      auto input_col_core_view = input.cores[col_id].const_view();

      BOBA_CALI_SWITCH("compute_new_dimensions", "setup_inner_products");

      auto inner_product_rows_view = ::boba::Multiindexer<2>({old_common_ranks, input_rows_ranks_left});
      auto inner_product_cols_view = ::boba::Multiindexer<2>({input_common_ranks, old_cols_ranks_right});

      size_t inner_product_rows_size = inner_product_rows_view.size();
      size_t inner_product_cols_size = inner_product_cols_view.size();

      boba::Matrix<space, data_t> inner_products_matrix({inner_product_rows_size, inner_product_cols_size});

      auto inner_products_matrix_view = inner_products_matrix.view();

      BOBA_CALI_SWITCH("setup_inner_products", "compute_inner_products");

      checkpoint();
      ::boba::loop<space, 2>(inner_products_matrix.sizes(),
                             [=] __boba_host_device__(Array<size_t, 2> rc)
      {
        auto [inner_product_row, inner_product_col] = rc;

        // Decompose left index
        const auto [old_common, input_rows_rank_left] = inner_product_rows_view.multiindex(inner_product_row);

        // Decompose right index
        const auto [input_common, old_cols_rank_right] = inner_product_cols_view.multiindex(inner_product_col);

        auto ip = data_t(0.0);
        for (size_t ip_index = 0; ip_index < common_dimension; ip_index++)
        {
          auto value_input_row = input_row_core_view({input_rows_rank_left, ip_index, input_common});
          auto value_old_col = old_col_core_view({old_common, ip_index, old_cols_rank_right});
          ip += value_old_col * value_input_row;
        }
        inner_products_matrix_view({inner_product_row, inner_product_col}) = ip;
      });

      BOBA_CALI_SWITCH("compute_inner_products", "compute_inner_products_svd");

      ::boba::SVD<space, data_t> inner_product_svd;
      size_t inner_product_significant_values = 0;
      if (use_inner_product_svd)
      {
        inner_product_svd(inner_products_matrix);
        apply_as_diagonal_right_in_place(inner_product_svd.S, inner_product_svd.U);
        inner_product_significant_values = inner_product_svd.significant_singular_values;
        if (inner_product_significant_values == 0_z)
        {
          BOBA_CALI_END("compute_inner_products_svd");
          checkpoint();
          this->fill_with_zeros();
          return;
        }
      }
      auto inner_product_svd_US_view = inner_product_svd.U.const_view();
      auto inner_product_svd_V_view = inner_product_svd.V.const_view();

      BOBA_CALI_SWITCH("compute_inner_products_svd", "resize_new_row_core");

      size_t left_index = 0;

      checkpoint();
      Array<size_t, 3> row_resizes{
        new_rows_ranks_left,
        new_rows,
        new_common_size};

      boba_always_assert_positive(new_rows_ranks_left, "nonpositive ranks");
      boba_always_assert_positive(new_common_size, "nonpositive ranks");

      if (use_inner_product_svd)
      {
        row_resizes[2] = inner_product_significant_values;
      }

      checkpoint();
      this->cores[row_id].resize(row_resizes);

      auto this_row_core_view = this->cores[row_id].view();
      auto R_row_view = qr[left_index].R.view();

      BOBA_CALI_SWITCH("resize_new_row_core", "compute_new_row_core");

      //
      // Expand row core to new dimensions
      //
      if (use_inner_product_svd)
      {
        checkpoint();
        ::boba::loop<space, 3>(row_resizes,
                               [=] __boba_host_device__(Array<size_t, 3> irr)
        {
          auto [rank_left, i, rank_right] = irr;

          auto [input_rows_rank_left, old_rows_rank_left] = left_row_view.multiindex(rank_left);

          checkpoint();
          auto sum = data_t(0.0);
          for (size_t old_common = 0; old_common < old_common_ranks; old_common++)
          {
            size_t inner_product_row = inner_product_rows_view.index({old_common, input_rows_rank_left});
            auto value_US = inner_product_svd_US_view({inner_product_row, rank_right});
            auto value_old_row = old_row_core_view({old_rows_rank_left, i, old_common});
            sum += value_US * value_old_row;
          }
          this_row_core_view({rank_left, i, rank_right}) = sum;
        });
      }
      else
      {
        checkpoint();
        ::boba::loop<space, 3>(row_resizes,
                               [=] __boba_host_device__(Array<size_t, 3> irr)
        {
          auto [rank_left, i, rank_right] = irr;

          auto left_indices = left_row_view.multiindex(rank_left);
          const size_t input_rows_rank_left = left_indices[0];
          const size_t old_rows_rank_left = left_indices[1];

          auto sum = data_t(0.0);
          for (size_t old_common = 0; old_common < old_common_ranks; old_common++)
          {
            size_t inner_product_row = inner_product_rows_view.index({old_common, input_rows_rank_left});
            auto value_inner_product_matrix = inner_products_matrix_view({inner_product_row, rank_right});
            auto value_old_row = old_row_core_view({old_rows_rank_left, i, old_common});
            sum += value_inner_product_matrix * value_old_row;
          }
          this_row_core_view({rank_left, i, rank_right}) = sum;
        });
      }

      BOBA_CALI_SWITCH("compute_new_row_core", "resize_new_col_core");

      //
      //
      size_t new_cols_ranks_left = row_resizes[2];
      Array<size_t, 3> col_resizes{
        new_cols_ranks_left,
        new_cols,
        new_cols_ranks_right};

      boba_always_assert_positive(new_cols_ranks_left, "nonpositive ranks");
      boba_always_assert_positive(new_cols_ranks_right, "nonpositive ranks");

      checkpoint();
      left_index = row_id;

      this->cores[col_id].resize(col_resizes);

      auto this_col_core_view = this->cores[col_id].view();
      auto R_col_view = qr[left_index].R.view();

      BOBA_CALI_SWITCH("resize_new_col_core", "compute_new_col_core");

      //
      // Compute new col core
      //
      if (use_inner_product_svd)
      {
        checkpoint();
        ::boba::loop<space, 3>(col_resizes,
                               [=] __boba_host_device__(Array<size_t, 3> irr)
        {
          auto [rank_left, i, rank_right] = irr;

          auto indices_right = right_col_view.multiindex(rank_right);
          const size_t input_cols_rank_right = indices_right[0];
          const size_t old_cols_rank_right = indices_right[1];

          auto sum = data_t(0.0);
          for (size_t input_common = 0; input_common < input_common_ranks; input_common++)
          {
            size_t inner_product_col = inner_product_cols_view.index({input_common, old_cols_rank_right});

            auto value_V = inner_product_svd_V_view({inner_product_col, rank_left});
            auto value_old_col = input_col_core_view({input_common, i, input_cols_rank_right});
            sum += value_old_col * value_V;
          }
          this_col_core_view({rank_left, i, rank_right}) = sum;
        });
      }
      else
      {
        checkpoint();
        ::boba::loop<space, 3>(col_resizes,
                               [=] __boba_host_device__(Array<size_t, 3> irr)
        {
          auto [rank_left, i, rank_right] = irr;

          auto indices_right = right_col_view.multiindex(rank_right);
          const size_t input_cols_rank_right = indices_right[0];
          const size_t old_cols_rank_right = indices_right[1];

          auto inner_product_col_indices = inner_product_cols_view.multiindex(rank_left);
          const size_t input_common_rank = inner_product_col_indices[0];
          const size_t old_cols_rank_right_long = inner_product_col_indices[1];

          auto sum = data_t(0.0);
          if (old_cols_rank_right == old_cols_rank_right_long)
          {
            sum = input_col_core_view({input_common_rank, i, input_cols_rank_right});
          }
          this_col_core_view({rank_left, i, rank_right}) = sum;
        });
      }

      BOBA_CALI_END("compute_new_col_core");
    } // end loop over true_dimension
  }

  // -------------------------------------------------------------------------------------
  // Section: Tensor round
  // -------------------------------------------------------------------------------------

  void compress(boba::Tensor<dimension, space, data_t>& input)
  {
    BOBA_CALI_MARK

    checkpoint();
    detail::ignore(input);

    boba_error("Not yet implemented");
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
    boba::Matrix<space, data_t> unfold_right;
    Array<boba::QR<space, data_t>, dimension> qr;
    checkpoint();
    round_qr_step(qr, unfold_right);
    checkpoint();
    round_svd_step(qr, unfold_right);
  }

  void round_qr_step(
    Array<QR<space, data_t>, dimension>& qr,
    boba::Matrix<space, data_t>& unfold_right)
  {
    BOBA_CALI_MARK
    checkpoint();

    auto unfold_left = compute_unfold_left(this->cores[0]);
    set_to_rank_one_scalar_core(this->cores[0], 0.0);
    for (size_t d = 0; d < dimension - 1; d++)
    {
      checkpoint();
      unfold_right = compute_unfold_right(this->cores[d + 1]);
      set_to_rank_one_scalar_core(this->cores[d + 1], 0.0);
      checkpoint();
      qr[d](unfold_left);
      checkpoint();
      qr[d].apply_R_left_in_place(unfold_right);
      qr[d].R.resize({1, 1});
      checkpoint();
      if (d < dimension - 2)
      {
        unfold_left = compute_unfold_left_from_unfold_right(unfold_right, this->cores[d + 1].sizes(1));
      }
      checkpoint();
    }
    checkpoint();
  }

  void round_svd_step(
    Array<QR<space, data_t>, dimension>& qr,
    boba::Matrix<space, data_t>& unfold_right)
  {
    Array<boba::SVD<space, data_t>, dimension - 1> svd;
    round_svd_step(qr, svd, unfold_right);
  }

  void round_svd_step(
    Array<QR<space, data_t>, dimension>& qr,
    Array<boba::SVD<space, data_t>, dimension - 1>& svd,
    boba::Matrix<space, data_t>& unfold_right)
  {
    BOBA_CALI_MARK

    for (size_t d = 0; d < dimension - 1; d++)
    {
      svd[d].tolerance_relative = svd_tolerance_relative;
      svd[d].tolerance_absolute = svd_tolerance_absolute;
      svd[d].max_kept_singular_values = max_ranks[1 + d];
    }

    for (size_t d = dimension - 1; d > 0; d--)
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
      svd[d - 1].V.resize({1, 1});
      this->cores[d] = write_to_core_from_right_fold(unfold_right, get_sizes(d));
      checkpoint();
      boba::Matrix<space, data_t> unfold_left = qr[d - 1].apply_householder_left(svd[d - 1].U);
      qr[d - 1].Q.resize({1, 1});
      svd[d - 1].U.resize({1, 1});
      checkpoint();
      if (d > 1)
      {
        unfold_right = compute_unfold_right_from_unfold_left(unfold_left, this->cores[d - 1].sizes(1));
      }
      else if (d == 1)
      {
        this->cores[0] = write_to_core_from_left_fold(unfold_left, this->cores[0].sizes(1));
      }
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
    this->cores[0] *= x;
  }

  /**
   * \brief
   * Multiplies each core of the tensor train by its own scalar, such that A = product(scalars)*A
   */

  void multiply_scalars(::boba::Array<data_t, dimension>& scalars)
  {
    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      this->cores[d].multiply_scalar(scalars[d]);
    }
  }

  // -------------------------------------------------------------------------------------
  // Section: Inverse
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Takes an initial guess inverse and computes the tensor train matrix inverse of this.
   */

  bool compute_inverse(
    TensorTrainSplitMatrix& inverse,
    data_t& residual_final,
    data_t tolerance,
    data_t maximum_iterations,
    bool ip_svd,
    bool verbose)
  {
    BOBA_CALI_MARK
    bool success = false;
    checkpoint();
    BOBA_CALI_BEGIN("setup");
    for (size_t d = 0; d < true_dimension; d++)
    {
      boba_always_assert_equal(core_rows()[d], core_cols()[d], "nonsquare inverse not yet implemented");
    }
    this->round();
    inverse.round();
    checkpoint();
    TensorTrainSplitMatrix inverse_temp(core_rows(), core_cols());
    inverse_temp.fill_with_zeros();
    inverse_temp.rename("inverse_temp");
    checkpoint();

    TensorTrainMatrix<true_dimension, space, data_t> identity_ttm(core_rows(), core_cols());
    identity_ttm.set_to_identity_train();
    TensorTrainSplitMatrix identity(identity_ttm);
    identity.round();

    auto initial_convergence = data_t(0.0);
    {
      TensorTrainSplitMatrix convergence_check(core_rows(), core_cols());
      convergence_check.fill_with_zeros();
      convergence_check.rename("convergence_check");
      convergence_check = inverse;
      convergence_check.TensorTrainSplitMatrix_right_multiply(*this, ip_svd);
      checkpoint();
      initial_convergence = boba::norm_difference_frobenius(convergence_check.decompress(), identity.decompress());
    }
    checkpoint();
    if (boba::is_env_nonempty("BOBA_VERBOSE"))
    {
      std::cout << "initial guess" << std::endl;
      inverse.decompress().print();
    }
    if (verbose)
    {
      std::cout << write_indent(1)
                << "% iterations,||R|| residual metric, time, compression rate, ranks"
                << std::endl;
    }

    BOBA_CALI_SWITCH("setup", "iterating");

    size_t iterations_used = 0;
    bool iterating = true;
    auto convergence = data_t(-1.0);
    while (iterating)
    {
      boba::TicToc<boba::tictoc_units::milliseconds> iterate;
      // scheme is Y_new = I - 0.5*A*X
      //           round Y_new
      //           X_new = X*2*Y_new
      checkpoint();
      inverse_temp = *this;
      inverse_temp.TensorTrainSplitMatrix_right_multiply(inverse, ip_svd);

      inverse_temp.round();
      inverse_temp *= -0.5;
      inverse_temp.TensorTrainSplitMatrix_add(identity);

      inverse_temp.round();
      inverse_temp *= 2.0;

      inverse.TensorTrainSplitMatrix_right_multiply(inverse_temp, ip_svd);
      inverse.round();

      {
        TensorTrainSplitMatrix convergence_check(core_rows(), core_cols());
        convergence_check.fill_with_zeros();
        convergence_check.rename("convergence_check");
        convergence_check = inverse;
        convergence_check.TensorTrainSplitMatrix_right_multiply(*this, ip_svd);
        checkpoint();
        convergence = boba::norm_difference_frobenius(convergence_check.decompress(), identity.decompress());
        convergence /= initial_convergence;
      }
      checkpoint();
      size_t iteration_time = iterate.timing();
      residual_final = convergence;
      checkpoint();
      if (verbose)
      {
        std::cout << write_indent(1)
                  << "% " << iterations_used
                  << "  ||R|| " << convergence
                  << "  " << iterate.units_string << " " << iteration_time
                  << "  CR " << inverse.compression_rate() << "x  "
                  << inverse.ranks_string()
                  << std::endl;
      }
      iterations_used++;
      size_t iteration_timeout = iterate.convert<boba::tictoc_units::seconds>(3600);
      if constexpr (boba::is_boba_debug_mode())
      {
        iteration_timeout *= 100;
      }
      if (iteration_time > iteration_timeout)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << " iteration timeout failure, "
                    << iteration_time << " > " << iteration_timeout
                    << " " << iterate.units_string << std::endl;
        }
        iterating = false;
        residual_final = 1;
        success = true;
      }
      if (convergence < tolerance)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << "% convergence achieved = " << convergence << std::endl;
        }
        iterating = false;
        success = true;
      }
      if (iterations_used < maximum_iterations)
      {
        iterating = true;
      }
      if (iterations_used >= maximum_iterations)
      {
        if (verbose)
        {
          std::cout << boba::write_indent(1) << "% maximum iterations reached " << std::endl;
        }

        if (!boba::isnan(residual_final))
        {
          success = true;
        }
        iterating = false;
      }
      checkpoint();
    }
    BOBA_CALI_END("iterating");
    checkpoint();
    return success;
  }

  // -------------------------------------------------------------------------------------
  // Section: Matrix-Vector Multiply
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Takes a tensor-train matrix times a tensor-train vector
   * this  : ( ... sum_j Rows_{i,:,j} x sum_k Cols_{j,:,k} ...) \n
   * input : ( ... sum_p Vector_{l,:,p} ...) \n
   * output: ( ... sum_j sum_k sum_p Rows_{i,:,j}*(Cols_{j,:,k}*Vector_{l,:,p}) ... ) \n
   */

  TensorTrain<true_dimension, space, data_t> TensorTrainMatrix_vector_multiply(
    TensorTrain<true_dimension, space, data_t> const& input) const
  {
    BOBA_CALI_MARK
    checkpoint();
    TensorTrain<true_dimension, space, data_t> output(input.sizes());

    for (size_t d = 0; d < true_dimension; d++)
    {
      size_t row_id = 2 * d;
      size_t col_id = 2 * d + 1;

      BOBA_CALI_BEGIN("initialize_sizes");

      checkpoint();
      size_t common_dimension = input.sizes(d);
      boba_always_assert_equal(
        common_dimension,
        this->core_cols()[d],
        "mismatch of common sizes");

      checkpoint();
      size_t input_ranks_left = input.get_ranks_left(d);
      size_t input_ranks_right = input.get_ranks_right(d);

      size_t old_rows_ranks_left = this->get_ranks_left(row_id);
      size_t old_common_ranks = this->get_ranks_right(row_id);
      size_t old_cols_ranks_right = this->get_ranks_right(col_id);
      boba_always_assert_equal(
        old_common_ranks,
        get_ranks_left(col_id),
        "ranks mismatch");

      size_t new_rows = this->core_rows()[d];

      auto left_row_view = ::boba::Multiindexer<2>({input_ranks_left, old_rows_ranks_left});
      size_t new_ranks_left = left_row_view.size();

      auto right_col_view = ::boba::Multiindexer<2>({input_ranks_right, old_cols_ranks_right});
      size_t new_ranks_right = right_col_view.size();

      auto old_row_core_view = this->cores[row_id].const_view();
      auto old_col_core_view = this->cores[col_id].const_view();

      auto input_core_view = input.cores[d].const_view();

      auto inner_product_rows_view = ::boba::Multiindexer<2>({old_common_ranks, input_ranks_left});
      auto inner_product_cols_view = ::boba::Multiindexer<2>({input_ranks_right, old_cols_ranks_right});

      size_t inner_product_rows_size = inner_product_rows_view.size();
      size_t inner_product_cols_size = inner_product_cols_view.size();

      boba::Matrix<space, data_t> inner_products_matrix({inner_product_rows_size, inner_product_cols_size});

      auto inner_products_matrix_view = inner_products_matrix.view();

      BOBA_CALI_SWITCH("initialize_sizes", "compute_inner_products");

      checkpoint();
      ::boba::loop<space, 2>(inner_products_matrix.sizes(),
                             [=] __boba_host_device__(Array<size_t, 2> rc)
      {
        size_t inner_product_row = rc[0];
        size_t inner_product_col = rc[1];

        // Decompose left index
        auto inner_product_left_indices = inner_product_rows_view.multiindex(inner_product_row);

        const size_t old_common = inner_product_left_indices[0];
        const size_t input_rows_rank_left = inner_product_left_indices[1];

        // Decompose right index
        auto inner_product_right_indices = inner_product_cols_view.multiindex(inner_product_col);

        const size_t input_common = inner_product_right_indices[0];
        const size_t old_cols_rank_right = inner_product_right_indices[1];

        auto ip = data_t(0.0);
        for (size_t ip_index = 0; ip_index < common_dimension; ip_index++)
        {
          auto value_input = input_core_view({input_rows_rank_left, ip_index, input_common});
          auto value_old_col = old_col_core_view({old_common, ip_index, old_cols_rank_right});
          ip += value_old_col * value_input;
        }
        inner_products_matrix_view({inner_product_row, inner_product_col}) = ip;
      });

      checkpoint();
      Array<size_t, 3> row_resizes{new_ranks_left, new_rows, new_ranks_right};

      checkpoint();
      output.cores[d].resize(row_resizes);
      auto output_core_view = output.cores[d].view();

      //
      // Expand row core to new dimensions
      //

      checkpoint();
      ::boba::loop<space, 3>(row_resizes,
                             [=] __boba_host_device__(Array<size_t, 3> irr)
      {
        auto [rank_left, i, rank_right] = irr;

        // Decompose rank left
        auto left_indices = left_row_view.multiindex(rank_left);
        const size_t input_rows_rank_left = left_indices[0];
        const size_t old_rows_rank_left = left_indices[1];

        auto sum = data_t(0.0);
        for (size_t old_common = 0; old_common < old_common_ranks; old_common++)
        {
          Array<size_t, 2> inner_product_rows_indices{old_common, input_rows_rank_left};
          size_t inner_product_row = inner_product_rows_view.index(inner_product_rows_indices);

          auto value_inner_product_matrix = inner_products_matrix_view({inner_product_row, rank_right});
          auto value_old_row = old_row_core_view({old_rows_rank_left, i, old_common});
          sum += value_inner_product_matrix * value_old_row;
        }
        output_core_view({rank_left, i, rank_right}) = sum;
      });
    } // end loop over true_dimension

    return output;
  }
};

// -------------------------------------------------------------------------------------
// Section: operator*
// -------------------------------------------------------------------------------------

template <size_t dimension, ::boba::execution_space space, typename _data_t>
void nan_check(TensorTrainSplitMatrix<dimension, space, _data_t> const& sttm)
{
  for (size_t d = 0; d < dimension; d++)
  {
    ::boba::nan_check(sttm.cores[d]);
  }
}

template <size_t dimension, ::boba::execution_space space, typename _data_t>
TensorTrainSplitMatrix<dimension, space, _data_t> operator*(_data_t scalar, TensorTrainSplitMatrix<dimension, space, _data_t> const& input)
{
  BOBA_CALI_MARK
  TensorTrainSplitMatrix<dimension, space, _data_t> temp{input};
  temp *= scalar;
  return temp;
}

} // namespace boba
