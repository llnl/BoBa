// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cassert>
#include <vector>

#ifdef BOBA_ENABLE_EIGEN
#include "Eigen/Dense"

#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/FFT>
#endif

namespace boba
{

#ifdef BOBA_ENABLE_EIGEN

/**
 * @brief Applies a one-dimensional Eigen FFT along the leading tensor mode.
 * @tparam dimension Tensor rank.
 * @tparam data_t Real scalar type underlying the complex data.
 * @tparam index_t Index type.
 * @param input Input tensor.
 * @param output Output tensor.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension, typename data_t>
void eigen_fft(
  Tensor<dimension, execution_space::CPU, complex<data_t>>& input,
  Tensor<dimension, execution_space::CPU, complex<data_t>>& output,
  bool is_inverse)
{
  using in_t = Eigen::Matrix<boba::complex<data_t>, Eigen::Dynamic, 1>;
  using out_t = Eigen::Matrix<boba::complex<data_t>, Eigen::Dynamic, 1>;

  size_t mode = 0;
  auto mode_size = static_cast<Eigen::Index>(input.sizes(mode));

  Eigen::FFT<data_t> fft;

  if constexpr (dimension == 1)
  {
    Eigen::Map<in_t> in_map(input.data(), static_cast<Eigen::Index>(mode_size));
    Eigen::Map<out_t> out_map(output.data(), static_cast<Eigen::Index>(mode_size));

    if (is_inverse)
    {
      fft.inv(out_map, in_map);
    }
    else
    {
      fft.fwd(out_map, in_map);
    }

    return;
  }

  //
  // We need to loop over all non-transformed modes and apply
  //
  auto untransformed_modes = boba::delete_element(input.sizes(), mode);
  ::boba::Multiindexer<dimension - 1> outer_mider(untransformed_modes);

  auto input_view = input.view(); // eigen errors if this is const_view
  auto output_view = output.view();

  for (size_t id = 0; id < outer_mider.size(); id++)
  {
    auto mid = outer_mider.multiindex(id);
    auto begin_mid = ::boba::concatenate(::boba::filled_array<1, index_t>(0_z), mid);

    // Offset pointer to location of begin_mid
    auto* in_begin = &input_view(begin_mid);
    auto* out_begin = &output_view(begin_mid);

    Eigen::Map<in_t> in_map(in_begin, static_cast<Eigen::Index>(mode_size));
    Eigen::Map<out_t> out_map(out_begin, static_cast<Eigen::Index>(mode_size));

    if (is_inverse)
    {
      fft.inv(out_map, in_map);
    }
    else
    {
      fft.fwd(out_map, in_map);
    }
  }
}

#else

/**
 * @brief Reports that Eigen FFT support is unavailable.
 * @tparam dimension Tensor rank.
 * @tparam data_t Real scalar type underlying the complex data.
 * @tparam index_t Index type.
 * @param input Input tensor.
 * @param output Output tensor.
 * @param is_inverse Selects inverse instead of forward transform.
 */
template <size_t dimension, typename data_t>
void eigen_fft(
  Tensor<dimension, execution_space::CPU, complex<data_t>>& input,
  Tensor<dimension, execution_space::CPU, complex<data_t>>& output,
  bool is_inverse)
{
  detail::ignore(input);
  detail::ignore(output);
  detail::ignore(is_inverse);
  boba_error("Eigen is not enabled");
}

#endif

} // namespace boba
