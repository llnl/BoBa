// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <execution_space space, typename data_t>
void nan_check(QuantizedTensorTrainMatrix<space, data_t> const& qttm)
{
  for (size_t d = 0; d < qttm.exponent; d++)
  {
    ::boba::nan_check(qttm.cores[d]);
  }
}

// -------------------------------------------------------------------------------------
// Section: Norms
// -------------------------------------------------------------------------------------

/**
 * \brief Computes the Frobenius norm of a quantized tensor train matrix.
 */

template <execution_space space, typename data_t>
typename QuantizedTensorTrainMatrix<space, data_t>::real_data_t
norm_frobenius(QuantizedTensorTrainMatrix<space, data_t> const& qttm)
{
  BOBA_CALI_MARK

  checkpoint();

  boba::Vector<space, data_t> in;
  in.rename("in");
  boba::Vector<space, data_t> out;
  out.rename("out");

  const size_t number_cores = qttm.exponent;

  for (size_t d = number_cores; d > 0; d--)
  {
    checkpoint();
    auto this_core_view = qttm.cores[d - 1].const_view();
    const size_t this_ranks_left = qttm.get_ranks_left(d - 1);
    const size_t this_ranks_right = qttm.get_ranks_right(d - 1);
    const size_t this_rows = qttm.core_rows(d - 1);
    const size_t this_cols = qttm.core_cols(d - 1);

    const size_t new_ranks_left = ::boba::pow(this_ranks_left, 2);
    const size_t new_ranks_right = ::boba::pow(this_ranks_right, 2);
    checkpoint();
    if (d == number_cores)
    {
      boba_assert_equal(new_ranks_right, 1_z, "Invalid rank size for the last core. The last rank should be 1.");
      in.resize({1});
      in.fill_with(1.0);
    }
    else
    {
      in = out;
    }
    out.resize({new_ranks_left});
    out.fill_with_zeros();

    auto rank_left_view = ::boba::Multiindexer<2>({this_ranks_left, this_ranks_left});
    auto rank_right_view = ::boba::Multiindexer<2>({this_ranks_right, this_ranks_right});

    boba::Matrix<space, data_t> temp({new_ranks_left, new_ranks_right});
    temp.fill_with_zeros();
    auto temp_atomic_view = temp.atomic_view();

    auto loop_indexer = ::boba::Multiindexer<4>({new_ranks_left, this_rows, this_cols, new_ranks_right});
    ::boba::loop<space, 1>(loop_indexer.size(),
                           [=] __boba_host_device__(size_t I)
    {
      auto lijk = loop_indexer.multiindex(I);
      auto [l, i, j, k] = lijk;
      auto rank_left_mid = rank_left_view.multiindex(l);
      auto rank_right_mid = rank_right_view.multiindex(k);
      auto value_this_1 = this_core_view({rank_left_mid[0], i, j, rank_right_mid[0]});
      auto value_this_2 = this_core_view({rank_left_mid[1], i, j, rank_right_mid[1]});
      temp_atomic_view({l, k}) += value_this_1 * value_this_2;
    });

    out = temp * in;
  }
  checkpoint();
  boba_assert_equal(out.size(), 1_z, "end result should be a scalar");
  auto inner_product = ::boba::sqrt(::boba::abs(out.sum_reduce()));
  checkpoint();
  return inner_product;
}

/**
 * \brief Frobenius norm difference of two quantized tensor train matrices.
 */

template <execution_space space, typename data_t>
auto norm_difference_frobenius(
  QuantizedTensorTrainMatrix<space, data_t> const& qttm_A,
  QuantizedTensorTrainMatrix<space, data_t> const& qttm_B)
{
  BOBA_CALI_MARK
  checkpoint();
  QuantizedTensorTrainMatrix<space, data_t> temp = qttm_A - qttm_B;
  checkpoint();
  temp.rename("norm_difference_temp");
  auto error_l2 = ::boba::norm_frobenius(temp);
  return error_l2;
}

} // namespace boba
