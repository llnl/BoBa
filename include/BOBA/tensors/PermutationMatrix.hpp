// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace boba
{

/**
 * \brief
 * Permutation matrix implemented as a list of indices (integers)
 */

template <execution_space space, typename _data_t>
struct PermutationMatrix : Tensor<1, space, _data_t>
{
  using base = Tensor<1, space, _data_t>;

  using typename base::data_t;
  using typename base::index_array;

  static constexpr std::string_view object_type_name = "PermutationMatrix";

  // constructor
  explicit PermutationMatrix(
    index_array sizes = ::boba::filled_array<1>(static_cast<index_t>(0)),
    std::string_view name = object_type_name)
      : base(std::move(sizes), name)
  {
  }

  PermutationMatrix(PermutationMatrix const&) = default;
  PermutationMatrix(PermutationMatrix&&) = default;
  PermutationMatrix& operator=(PermutationMatrix const&) = default;
  PermutationMatrix& operator=(PermutationMatrix&&) = default;

  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  PermutationMatrix(PermutationMatrix<rhs_space, data_t> const& rhs)
      : base(rhs)
  {
  }

  template <::boba::execution_space rhs_space>
    requires(space != rhs_space)
  PermutationMatrix& operator=(PermutationMatrix<rhs_space, data_t> const& rhs)
  {
    return static_cast<base&>(*this) = rhs;
  }

  // -------------------------------------------------------------------------------------
  // Section: Special cases
  // -------------------------------------------------------------------------------------

  size_t rows() const
  {
    return this->sizes(0);
  }

  size_t cols() const
  {
    return this->sizes(0);
  }

  // -------------------------------------------------------------------------------------
  // Section: Read/write
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Overwrite this with the identity PermutationMatrix
   */

  void set_to_identity_matrix()
  {
    BOBA_CALI_OBJECT_BEGIN("permutation_set_to_identity");
    auto* this_data = this->data();
    ::boba::loop<space, 1>({this->size()},
                           [=] __boba_host_device__(size_t i)
    {
      this_data[i] = i;
    });
    BOBA_CALI_OBJECT_END("permutation_set_to_identity");
  }

  // -------------------------------------------------------------------------------------
  // Section: Transpose
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * output = this^T
   */

  PermutationMatrix transpose() const
  {
    BOBA_CALI_OBJECT_BEGIN("permutation_transpose");
    PermutationMatrix transpose(*this);
    transpose.rename(this->name() + "_transpose");
    auto transpose_view = transpose.view();
    auto this_view = this->const_view();
    auto length = this->size();
    ::boba::loop<space, 1>({length},
                           [=] __boba_host_device__(index_t i)
    {
      auto find_id = data_t(0);
      bool id_found = false;
      for (size_t j = 0; j < length; j++)
      {
        data_t match = this_view({j});
        if (match == static_cast<data_t>(i))
        {
          find_id = static_cast<data_t>(j);
          id_found = true;
        }
      }
      boba_assert(id_found, "Invalid permutation");
      transpose_view({i}) = find_id;
    });
    BOBA_CALI_OBJECT_END("permutation_transpose");
    return transpose;
  }

  // -------------------------------------------------------------------------------------
  // Section: Norm difference
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Returns the number of mismatched entries
   */

  template <execution_space rhs_space>
  size_t norm_difference(PermutationMatrix<rhs_space, data_t>& rhs)
  {
    BOBA_CALI_OBJECT_MARK
    if (not(this->sizes() == rhs.sizes()))
      throw std::length_error("matrix shapes must match.");

    auto rhs_data = rhs.data();
    auto this_data = this->data();

    data_t error_norm = 0.0;

    ::boba::sum_reduce<boba::default_execution_space>(error_norm, index_t(0), rows(), [=] __boba_host_device__(index_t i, sum_reducer_operator<data_t> & _error_norm)
    {
      if (rhs_data[i] != this_data[i])
      {
        _error_norm += 1;
      }
    });

    return error_norm;
  }
};

// -------------------------------------------------------------------------------------
// Section: Permute
// -------------------------------------------------------------------------------------

/**
 * \brief
 * output = permutation*input
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> permute_from_left(
  PermutationMatrix<space, index_t> const& permutation,
  Matrix<space, data_t> const& input)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  auto output_rows = permutation.rows();
  auto output_cols = input.cols();

  Matrix<space, data_t> output({output_rows, output_cols});

  auto permutation_view = permutation.const_view();
  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 2>({output_rows, output_cols},
                         [=] __boba_host_device__(Array<size_t, 2> rc)
  {
    auto [row, col] = rc;
    auto permuted_row = static_cast<index_t>(permutation_view({row}));
    output_view({permuted_row, col}) = input_view({row, col});
  });
  return output;
}

/**
 * \brief
 * output = permutation*input
 */

template <execution_space space, typename data_t>
Vector<space, data_t> permute_from_left(
  PermutationMatrix<space, index_t> const& permutation,
  Vector<space, data_t> const& input)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  auto output_rows = permutation.rows();
  auto output_cols = input.size();

  Vector<space, data_t> output({output_rows});

  auto permutation_view = permutation.const_view();
  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 1>({output_rows},
                         [=] __boba_host_device__(index_t row)
  {
    auto permuted_row = static_cast<index_t>(permutation_view({row}));
    output_view(permuted_row) = input_view(row);
  });
  return output;
}

/**
 * \brief
 * output = permutation*input
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> permute_from_right(
  PermutationMatrix<space, index_t> const& permutation,
  Matrix<space, data_t> const& input)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  auto output_rows = input.rows();
  auto output_cols = permutation.cols();

  Matrix<space, data_t> output({output_rows, output_cols});

  auto permutation_view = permutation.const_view();
  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 2>({output_rows, output_cols},
                         [=] __boba_host_device__(Array<size_t, 2> rc)
  {
    auto [row, col] = rc;
    auto permuted_col = static_cast<index_t>(permutation_view({col}));
    output_view({row, col}) = input_view({row, permuted_col});
  });
  return output;
}

/**
 * \brief
 * output = permutation*input
 */

template <execution_space space, typename data_t>
Vector<space, data_t> permute_from_right(
  PermutationMatrix<space, index_t> const& permutation,
  Vector<space, data_t> const& input)
{
  BOBA_CALI_OBJECT_MARK
  checkpoint_objects();

  auto output_rows = input.rows();
  auto output_cols = permutation.cols();

  Vector<space, data_t> output({output_rows});

  auto permutation_view = permutation.const_view();
  auto input_view = input.const_view();
  auto output_view = output.view();

  ::boba::loop<space, 1>({output_rows},
                         [=] __boba_host_device__(index_t row)
  {
    auto permuted_row = static_cast<index_t>(permutation_view({row}));
    output_view(permuted_row) = input_view(row);
  });
  return output;
}

/**
 * \brief
 * output = permutations * vec
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(
  const PermutationMatrix<space, index_t>& permutations,
  const Vector<space, data_t>& vec)
{
  return permute_from_left(permutations, vec);
}

/**
 * \brief
 * output = vec * permutations
 */

template <execution_space space, typename data_t>
Vector<space, data_t> operator*(
  const Vector<space, data_t>& vec,
  const PermutationMatrix<space, index_t>& permutations)
{
  return permute_from_right(permutations, vec);
}

/**
 * \brief
 * output = permutations * vec
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator*(
  const PermutationMatrix<space, index_t>& permutations,
  const Matrix<space, data_t>& matrix)
{
  return permute_from_left(permutations, matrix);
}

/**
 * \brief
 * output = vec * permutations
 */

template <execution_space space, typename data_t>
Matrix<space, data_t> operator*(
  const Matrix<space, data_t>& matrix,
  const PermutationMatrix<space, index_t>& permutations)
{
  return permute_from_right(permutations, matrix);
}
} // namespace boba
