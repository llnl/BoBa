// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

namespace boba
{

// -------------------------------------------------------------------------------------
// Section: NaN checks
// -------------------------------------------------------------------------------------

template <execution_space space, typename data_t>
void nan_check(QuantizedTensorTrain<space, data_t> const& x)
{
  for (size_t d = 0; d < x.exponent; d++)
  {
    ::boba::nan_check(x.cores.at(d));
  }
}

// -------------------------------------------------------------------------------------
// Section: Norms
// -------------------------------------------------------------------------------------

/**
 * \brief Little l2 norm of the quantized tensor train in the sense of vectors.
 */

template <execution_space space, typename data_t>
auto norm_frobenius(QuantizedTensorTrain<space, data_t> const& x)
{
  BOBA_CALI_MARK
  const auto product = x.inner_product(x);
  const auto abs_product = boba::abs(product);
  return boba::sqrt(abs_product);
}

/**
 * \brief Compute the inner product of two quantized tensor trains.
 *
 * \param[in] A first quantized tensor train
 * \param[in] B second quantized tensor train
 * \return inner product of `A` and `B`
 */

template <execution_space space, typename data_t>
data_t inner_product(const QuantizedTensorTrain<space, data_t>& A, const QuantizedTensorTrain<space, data_t>& B)
{
  return A.inner_product(B);
}

/**
 * \brief Compute the FFT of a quantized tensor train.
 *
 * This function takes a quantized tensor train and computes its FFT,
 * returning a new quantized tensor train with complex data type.See
 * "Superfast Fourier transform using QTT approximation" by Doglov,
 *  Khoromskij, Savostyanov (2012).
 *
 * \param[in] x input quantized tensor train, which must have base 2
 * \param[in] tol tolerance used in the FFT computation
 * \return FFT of the input tensor train
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
fft(QuantizedTensorTrain<space, data_t> x, data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for FFT");
  return fft(to_complex<space, data_t>(x), tol);
}

template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>> fft(QuantizedTensorTrain<space, boba::complex<data_t>> x, data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for FFT");

  const boba::complex<data_t> img{static_cast<data_t>(0), static_cast<data_t>(1)};

  index_t dimension = x.exponent;

  std::vector<index_t> ranks(dimension + 1);
  for (index_t i = 0; i < dimension; i++)
  {
    ranks[i] = x.get_ranks_left(i);
  }
  ranks[dimension] = x.get_ranks_right(dimension - 1);

  for (index_t i = dimension - 1; i > 0; i--)
  {
    index_t ri1 = x.cores[i].sizes(0);
    index_t ri2 = x.cores[i].sizes(2);

    boba::Tensor<3, space, boba::complex<data_t>> crd2({ri1, 2, ri2});
    {
      auto y_i_view = x.cores[i].const_view();
      auto crd2_view = crd2.view();
      ::boba::loop<space, 2>(boba::Array<size_t, 2>({ri1, ri2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
      {
        auto [ii, jj] = mid;
        boba::Array<index_t, 3> mid1({ii, 0, jj});
        boba::Array<index_t, 3> mid2({ii, 1, jj});
        crd2_view(mid1) = (y_i_view(mid1) + y_i_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
        crd2_view(mid2) = (y_i_view(mid1) - y_i_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
      });
    }

    {
      x.cores[i].resize({2 * ri1, 2, ri2});
      x.cores[i].fill_with_zeros();
      auto y_i_view = x.cores[i].view();
      auto crd2_view = crd2.const_view();
      ::boba::loop<space, 2>(boba::Array<size_t, 2>({ri1, ri2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
      {
        auto [ii, jj] = mid;
        boba::Array<index_t, 3> mid1({ii, 0, jj});
        boba::Array<index_t, 3> mid2({ii, 1, jj});
        boba::Array<index_t, 3> mid3({ii + ri1, 1, jj});
        y_i_view(mid1) = crd2_view(mid1);
        y_i_view(mid3) = crd2_view(mid2);
      });
    }

    boba::Matrix<space, boba::complex<data_t>> rv({1, 1});
    rv.fill_with(1.0);

    for (index_t j = 0; j <= i - 1; j++)
    {
      auto cr = x.cores[j];
      index_t rj1 = cr.sizes(0);
      index_t rj2 = cr.sizes(2);

      data_t theta = -2 * boba::pi / boba::pow(2, i - j + 1);
      boba::complex<data_t> omega = boba::cos(theta) + boba::sin(theta) * img;

      if (j == 0)
      {
        ranks[j] = rj1;
        ranks[j + 1] = 2 * rj2;

        x.cores[j].resize({ranks[j], 2, ranks[j + 1]});
        x.cores[j].fill_with_zeros();

        auto y_j_view = x.cores[j].view();
        auto cr_view = cr.const_view();
        ::boba::loop<space, 2>(boba::Array<size_t, 2>({rj1, rj2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
        {
          auto [i2, j2] = mid;
          boba::Array<index_t, 3> mid1({i2, 0, j2});
          boba::Array<index_t, 3> mid3({i2, 0, j2 + rj2});
          boba::Array<index_t, 3> mid2({i2, 1, j2});
          boba::Array<index_t, 3> mid4({i2, 1, j2 + rj2});
          y_j_view(mid1) = cr_view(mid1);
          y_j_view(mid2) = cr_view(mid2);
          y_j_view(mid3) = cr_view(mid1);
          y_j_view(mid4) = omega * y_j_view(mid2);
        });
      }
      else
      {
        ranks[j] = 2 * rj1;
        ranks[j + 1] = 2 * rj2;

        x.cores[j].resize({ranks[j], 2, ranks[j + 1]});
        x.cores[j].fill_with_zeros();

        auto y_j_view = x.cores[j].view();
        auto cr_view = cr.const_view();
        ::boba::loop<space, 2>(boba::Array<size_t, 2>({rj1, rj2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
        {
          auto [i2, j2] = mid;
          boba::Array<index_t, 3> mid1({i2, 0, j2});
          boba::Array<index_t, 3> mid3({i2 + rj1, 0, j2 + rj2});
          boba::Array<index_t, 3> mid2({i2, 1, j2});
          boba::Array<index_t, 3> mid4({i2 + rj1, 1, j2 + rj2});
          y_j_view(mid1) = cr_view(mid1);
          y_j_view(mid2) = cr_view(mid2);
          y_j_view(mid3) = cr_view(mid1);
          y_j_view(mid4) = omega * y_j_view(mid2);
        });
      }

      // Reshape
      boba::Matrix<space, boba::complex<data_t>> y_j_reshaped({ranks[j], 2 * ranks[j + 1]});
      y_j_reshaped.reshape(x.cores[j]);
      // Matrix multiply
      auto y_j_reshaped_2 = rv * y_j_reshaped;
      // Update Rank
      ranks[j] = y_j_reshaped_2.rows();
      // Reshape Again
      y_j_reshaped.resize({2 * ranks[j], ranks[j + 1]});
      y_j_reshaped.reshape(y_j_reshaped_2);
      // QR
      ::boba::QR<space, boba::complex<data_t>> qr;
      qr(y_j_reshaped);
      rv = qr.R;
      // Final reshape
      x.cores[j].resize({ranks[j], 2, rv.rows()});
      x.cores[j].reshape(qr.Q);
    }
    boba::Matrix<space, boba::complex<data_t>> y_i_reshaped({ranks[i], 2 * ranks[i + 1]});
    y_i_reshaped.reshape(x.cores[i]);

    ranks[i] = rv.rows();

    auto y_j_reshaped = rv * y_i_reshaped;

    for (index_t j = i; j > 0; j--)
    {
      ::boba::SVD<space, boba::complex<data_t>> svd;
      svd.tolerance_relative = tol / boba::sqrt(static_cast<data_t>(i));
      svd(y_j_reshaped);
      auto rnew = svd.S.sizes(0);

      auto u = svd.U;
      apply_as_diagonal_right_in_place(svd.S, u);

      x.cores[j].resize({rnew, 2, ranks[j + 1]});
      x.cores[j].reshape(svd.V.conjugate_transpose());

      boba::Matrix<space, boba::complex<data_t>> y_jm1_reshaped_2({2 * ranks[j - 1], ranks[j]});
      y_jm1_reshaped_2.reshape(x.cores[j - 1]);
      y_jm1_reshaped_2 = y_jm1_reshaped_2 * u;

      ranks[j] = rnew;

      y_j_reshaped.resize({ranks[j - 1], 2 * ranks[j]});
      y_j_reshaped.reshape(y_jm1_reshaped_2);
    }
    x.cores[0].resize({ranks[0], 2, ranks[1]});
    x.cores[0].reshape(y_j_reshaped);
  }

  boba::Tensor<3, space, boba::complex<data_t>> y_0_2({ranks[0], 2, ranks[1]});
  {
    auto y_0_view = x.cores[0].const_view();
    auto y_0_view_2 = y_0_2.view();
    ::boba::loop<space, 2>(boba::Array<size_t, 2>({ranks[0], ranks[1]}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      boba::Array<index_t, 3> mid1({i, 0, j});
      boba::Array<index_t, 3> mid2({i, 1, j});
      y_0_view_2(mid1) = (y_0_view(mid1) + y_0_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
      y_0_view_2(mid2) = (y_0_view(mid1) - y_0_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
    });
  }
  x.cores[0] = y_0_2;

  // Reverse the train
  QuantizedTensorTrain<space, boba::complex<data_t>> y_out(2, dimension);
  for (index_t i = 0; i < dimension; i++)
  {
    y_out.cores[dimension - i - 1] = x.cores[i];
    boba::permute(y_out.cores[dimension - i - 1], {2, 1, 0});
  }

  return y_out;
}

/**
 * @brief Computes the Inverse Fast Fourier Transform (IFFT) of a quantized tensor train.
 *
 * This function takes a quantized tensor train and computes its FFT,
 * returning a new quantized tensor train with complex data type.See
 * "Superfast Fourier transform using QTT approximation" by Doglov,
 *  Khoromskij, Savostyanov (2012). We made slight modification to compute the inverse
 *
 * @param x The input quantized tensor train to be transformed.
 *          It must have a base of 2 for the FFT to be valid.
 * @param tol A tolerance value used in the FFT computation.
 *
 * @return A quantized tensor train containing the FFT of the input tensor train.
 *
 * @note The function asserts that the base of the input tensor train is 2.
 *       Ensure that the input tensor train is properly initialized before calling this function.
 *
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
ifft(QuantizedTensorTrain<space, data_t> x, data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for FFT");
  return ifft(to_complex<space, data_t>(x), tol);
}

template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>> ifft(QuantizedTensorTrain<space, boba::complex<data_t>> x, data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for FFT");

  const boba::complex<data_t> img{static_cast<data_t>(0), static_cast<data_t>(1)};

  index_t dimension = x.exponent;

  std::vector<index_t> ranks(dimension + 1);
  for (index_t i = 0; i < dimension; i++)
  {
    ranks[i] = x.get_ranks_left(i);
  }
  ranks[dimension] = x.get_ranks_right(dimension - 1);

  for (index_t i = dimension - 1; i > 0; i--)
  {
    index_t ri1 = x.cores[i].sizes(0);
    index_t ri2 = x.cores[i].sizes(2);

    boba::Tensor<3, space, boba::complex<data_t>> crd2({ri1, 2, ri2});
    {
      auto y_i_view = x.cores[i].const_view();
      auto crd2_view = crd2.view();
      ::boba::loop<space, 2>(boba::Array<size_t, 2>({ri1, ri2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
      {
        auto [ii, jj] = mid;
        boba::Array<index_t, 3> mid1({ii, 0, jj});
        boba::Array<index_t, 3> mid2({ii, 1, jj});
        crd2_view(mid1) = (y_i_view(mid1) + y_i_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
        crd2_view(mid2) = (y_i_view(mid1) - y_i_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
      });
    }

    {
      x.cores[i].resize({2 * ri1, 2, ri2});
      x.cores[i].fill_with_zeros();
      auto y_i_view = x.cores[i].view();
      auto crd2_view = crd2.const_view();
      ::boba::loop<space, 2>(boba::Array<size_t, 2>({ri1, ri2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
      {
        auto [ii, jj] = mid;
        boba::Array<index_t, 3> mid1({ii, 0, jj});
        boba::Array<index_t, 3> mid2({ii, 1, jj});
        boba::Array<index_t, 3> mid3({ii + ri1, 1, jj});
        y_i_view(mid1) = crd2_view(mid1);
        y_i_view(mid3) = crd2_view(mid2);
      });
    }

    boba::Matrix<space, boba::complex<data_t>> rv({1, 1});
    rv.fill_with(1.0);

    for (index_t j = 0; j <= i - 1; j++)
    {
      auto cr = x.cores[j];
      index_t rj1 = cr.sizes(0);
      index_t rj2 = cr.sizes(2);

      data_t theta = 2 * boba::pi / boba::pow(2, i - j + 1);
      boba::complex<data_t> omega = boba::cos(theta) + boba::sin(theta) * img;

      if (j == 0)
      {
        ranks[j] = rj1;
        ranks[j + 1] = 2 * rj2;

        x.cores[j].resize({ranks[j], 2, ranks[j + 1]});
        x.cores[j].fill_with_zeros();

        auto y_j_view = x.cores[j].view();
        auto cr_view = cr.const_view();
        ::boba::loop<space, 2>(boba::Array<size_t, 2>({rj1, rj2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
        {
          auto [i2, j2] = mid;
          boba::Array<index_t, 3> mid1({i2, 0, j2});
          boba::Array<index_t, 3> mid3({i2, 0, j2 + rj2});
          boba::Array<index_t, 3> mid2({i2, 1, j2});
          boba::Array<index_t, 3> mid4({i2, 1, j2 + rj2});
          y_j_view(mid1) = cr_view(mid1);
          y_j_view(mid2) = cr_view(mid2);
          y_j_view(mid3) = cr_view(mid1);
          y_j_view(mid4) = omega * y_j_view(mid2);
        });
      }
      else
      {
        ranks[j] = 2 * rj1;
        ranks[j + 1] = 2 * rj2;

        x.cores[j].resize({ranks[j], 2, ranks[j + 1]});
        x.cores[j].fill_with_zeros();

        auto y_j_view = x.cores[j].view();
        auto cr_view = cr.const_view();
        ::boba::loop<space, 2>(boba::Array<size_t, 2>({rj1, rj2}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
        {
          auto [i2, j2] = mid;
          boba::Array<index_t, 3> mid1({i2, 0, j2});
          boba::Array<index_t, 3> mid3({i2 + rj1, 0, j2 + rj2});
          boba::Array<index_t, 3> mid2({i2, 1, j2});
          boba::Array<index_t, 3> mid4({i2 + rj1, 1, j2 + rj2});
          y_j_view(mid1) = cr_view(mid1);
          y_j_view(mid2) = cr_view(mid2);
          y_j_view(mid3) = cr_view(mid1);
          y_j_view(mid4) = omega * y_j_view(mid2);
        });
      }

      // Reshape
      boba::Matrix<space, boba::complex<data_t>> y_j_reshaped({ranks[j], 2 * ranks[j + 1]});
      y_j_reshaped.reshape(x.cores[j]);
      // Matrix multiply
      auto y_j_reshaped_2 = rv * y_j_reshaped;
      // Update Rank
      ranks[j] = y_j_reshaped_2.rows();
      // Reshape Again
      y_j_reshaped.resize({2 * ranks[j], ranks[j + 1]});
      y_j_reshaped.reshape(y_j_reshaped_2);
      // QR
      ::boba::QR<space, boba::complex<data_t>> qr;
      qr(y_j_reshaped);
      rv = qr.R;
      // Final reshape
      x.cores[j].resize({ranks[j], 2, rv.rows()});
      x.cores[j].reshape(qr.Q);
    }

    boba::Matrix<space, boba::complex<data_t>> y_i_reshaped({ranks[i], 2 * ranks[i + 1]});
    y_i_reshaped.reshape(x.cores[i]);

    ranks[i] = rv.rows();

    auto y_j_reshaped = rv * y_i_reshaped;

    for (index_t j = i; j > 0; j--)
    {
      ::boba::SVD<space, boba::complex<data_t>> svd;
      svd.tolerance_relative = tol / boba::sqrt(static_cast<data_t>(i));
      svd(y_j_reshaped);
      auto rnew = svd.S.sizes(0);

      auto u = svd.U;
      apply_as_diagonal_right_in_place(svd.S, u);

      x.cores[j].resize({rnew, 2, ranks[j + 1]});
      x.cores[j].reshape(svd.V.conjugate_transpose());

      boba::Matrix<space, boba::complex<data_t>> y_jm1_reshaped_2({2 * ranks[j - 1], ranks[j]});
      y_jm1_reshaped_2.reshape(x.cores[j - 1]);
      y_jm1_reshaped_2 = y_jm1_reshaped_2 * u;

      ranks[j] = rnew;

      y_j_reshaped.resize({ranks[j - 1], 2 * ranks[j]});
      y_j_reshaped.reshape(y_jm1_reshaped_2);
    }

    x.cores[0].resize({ranks[0], 2, ranks[1]});
    x.cores[0].reshape(y_j_reshaped);
  }

  boba::Tensor<3, space, boba::complex<data_t>> y_0_2({ranks[0], 2, ranks[1]});
  {
    auto y_0_view = x.cores[0].const_view();
    auto y_0_view_2 = y_0_2.view();
    ::boba::loop<space, 2>(boba::Array<size_t, 2>({ranks[0], ranks[1]}), [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      boba::Array<index_t, 3> mid1({i, 0, j});
      boba::Array<index_t, 3> mid2({i, 1, j});
      y_0_view_2(mid1) = (y_0_view(mid1) + y_0_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
      y_0_view_2(mid2) = (y_0_view(mid1) - y_0_view(mid2)) / boba::sqrt(static_cast<data_t>(2));
    });
  }
  x.cores[0] = y_0_2;

  // Reverse the train
  QuantizedTensorTrain<space, boba::complex<data_t>> y_out(2, dimension);
  for (index_t i = 0; i < dimension; i++)
  {
    y_out.cores[dimension - i - 1] = x.cores[i];
    boba::permute(y_out.cores[dimension - i - 1], {2, 1, 0});
  }
  return y_out;
}

/**
 * @brief Function overload where we convert the qtt x to complex and call the complex version of qtt_dft_kron_id.
 */

template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_dft_kron_id(QuantizedTensorTrain<space, data_t> x,
                index_t d1,
                index_t d2,
                data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for FFT");

  // Convert real QTT -> complex QTT with imag=0, then call the complex overload.
  return qtt_dft_kron_id(to_complex<space, data_t>(x), d1, d2, tol);
}
/**
 * @brief Computes the (V x I)f where f is a quantized tensor train.
 *        V represents the FFT from
 * "Superfast Fourier transform using QTT approximation" by Doglov,
 *  Khoromskij, Savostyanov (2012).
 *        I is an identity operator
 * the DFT will be applied to the last d2 cores of f. Denote
 * f(i1,...,id1,j1,...,jd2) = sum_k f1(i1,...,id1,k) f2(k,j1,...,jd2)
 * We form the TT cores of f2(k,:,...,:) for each individual k, and apply
 * the superfast QTT Fourier transform to obtain y2(:,...,:,k). Then we
 * put together the final result:
 * y(i1,...,id1,j1,...,jd2) = sum_k f1(i1,...,id1,k) y2(k,j1,...,jd2)

 *
 * @param x The input quantized tensor train to be transformed.
 *          It must have a base of 2 for the FFT to be valid.
 * @param tol A tolerance value used in the FFT computation.
 *
 * @return A quantized tensor train containing the FFT of the input tensor train.
 *
 * @note The function asserts that the base of the input tensor train is 2.
 *       Ensure that the input tensor train is properly initialized before calling this function.
 *
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_dft_kron_id(QuantizedTensorTrain<space, boba::complex<data_t>> f,
                index_t d1,
                index_t d2,
                data_t tol)
{
  // dimension should be d1 + d2
  QuantizedTensorTrain<space, boba::complex<data_t>> y(2, d1 + d2);
  QuantizedTensorTrain<space, boba::complex<data_t>> y_sums(2, d1 + d2);
  for (index_t i = 0; i < d1 + d2; ++i)
  {
    y_sums.cores[i].fill_with_zeros();
  }

  // boundary is between cores (d1-1) and d1
  const index_t Rb_left = f.cores[d1 - 1].sizes(2); // right rank of core d1-1
  const index_t boundary_rank = Rb_left;

  // Build f2 as TT of length d2 (these are cores d1..d1+d2-1 in f)
  QuantizedTensorTrain<space, boba::complex<data_t>> f2(2, d2);

  // Copy the unchanged tail of f2: cores (d1+1 .. d1+d2-1) into f2.cores[1 .. d2-1]
  for (index_t i = d1 + 1; i < d1 + d2; ++i)
  {                                          // includes last core d1+d2-1
    f2.cores[i - (d1 + 1) + 1] = f.cores[i]; // map i=d1+1 -> f2[1]
  }

  // Copy the first (d1-1) cores of y (unchanged part)
  for (index_t i = 0; i < d1 - 1; ++i)
  {
    y.cores[i] = f.cores[i];
  }

  // Views for slicing
  auto core_view = f.cores[d1].view(); // boundary core for the f2-side
  auto idx1 = core_view.sizes(1);      // should be 2
  auto idx2 = core_view.sizes(2);      // right rank of core d1

  // f2.cores[0] will be a slice: (1, idx1, idx2)
  f2.cores[0].resize({1, idx1, idx2});
  auto f2_core0_view = f2.cores[0].view();

  // y.cores[d1-1] will be a slice: (idx1Y, idx2Y, 1)
  auto idx1Y = f.cores[d1 - 1].sizes(0);
  auto idx2Y = f.cores[d1 - 1].sizes(1);
  y.cores[d1 - 1].resize({idx1Y, idx2Y, 1});
  auto core_viewY = y.cores[d1 - 1].view();

  auto f_core_d1m1_view = f.cores[d1 - 1].view(); // (idx1Y, idx2Y, boundary_rank)

  // Loop over the boundary rank
  for (index_t k = 0; k < boundary_rank; ++k)
  {
    // Slice f.cores[d1] on LEFT rank: fcores{d1+1}(k,:,:) in MATLAB
    // i.e. set f2.cores[0](0,:,:) = f.cores[d1](k,:,:)
    boba::Array<index_t, 2> loop_upper_bounds{{idx1, idx2}};
    ::boba::loop<space, 2>(loop_upper_bounds,
                           [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      f2_core0_view({0, i, j}) = core_view({k, i, j});
    });

    // Apply FFT to the d2-block
    auto y2 = fft(f2, tol);

    // Copy over the last d2 cores of y:
    // y.cores[d1] is the "first core of f2 slice" output, and y.cores[d1+1..] are the unchanged tail
    y.cores[d1] = y2.cores[0];

    // Mapping for tail cores (start at f2 index 1)
    for (index_t i = d1 + 1; i < d1 + d2; ++i)
    {
      y.cores[i] = y2.cores[i - d1]; // i=d1+1 -> y2[1]
    }

    // Slice f.cores[d1-1] on RIGHT rank: fcores{d1}(:,:,k) in MATLAB
    boba::Array<index_t, 2> loop_upper_bounds2{{idx1Y, idx2Y}};
    ::boba::loop<space, 2>(loop_upper_bounds2,
                           [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      core_viewY({i, j, 0}) = f_core_d1m1_view({i, j, k});
    });

    // Sum into the result (consider rounding here in practice)
    y_sums = y_sums + y;
  }

  return y_sums;
}

/**
 * @brief Function overload where we convert the qtt x to complex and call the complex version of qtt_idft_kron_id.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_idft_kron_id(QuantizedTensorTrain<space, data_t> x,
                 index_t d1,
                 index_t d2,
                 data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2 for iFFT");

  // Convert real QTT -> complex QTT with imag=0, then call the complex overload.
  return qtt_idft_kron_id(to_complex<space, data_t>(x), d1, d2, tol);
}
/**
 * @brief Computes the (V^{-1} x I)f where f is a quantized tensor train.
 *        V represents the FFT from
 * "Superfast Fourier transform using QTT approximation" by Doglov,
 *  Khoromskij, Savostyanov (2012).
 *        I is an identity operator
 * the DFT will be applied to the last d2 cores of f. Denote
 * f(i1,...,id1,j1,...,jd2) = sum_k f1(i1,...,id1,k) f2(k,j1,...,jd2)
 * We form the TT cores of f2(k,:,...,:) for each individual k, and apply
 * the superfast QTT Fourier transform to obtain y2(:,...,:,k). Then we
 * put together the final result:
 * y(i1,...,id1,j1,...,jd2) = sum_k f1(i1,...,id1,k) y2(k,j1,...,jd2)

 *
 * @param x The input quantized tensor train to be transformed.
 *          It must have a base of 2 for the FFT to be valid.
 * @param tol A tolerance value used in the FFT computation.
 *
 * @return A quantized tensor train containing the FFT of the input tensor train.
 *
 * @note The function asserts that the base of the input tensor train is 2.
 *       Ensure that the input tensor train is properly initialized before calling this function.
 *
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>> qtt_idft_kron_id(QuantizedTensorTrain<space, boba::complex<data_t>> f, index_t d1, index_t d2, data_t tol)
{
  // dimension should be d1 + d2
  QuantizedTensorTrain<space, boba::complex<data_t>> y(2, d1 + d2);
  QuantizedTensorTrain<space, boba::complex<data_t>> y_sums(2, d1 + d2);
  for (index_t i = 0; i < d1 + d2; ++i)
  {
    y_sums.cores[i].fill_with_zeros();
  }

  // boundary is between cores (d1-1) and d1
  const index_t Rb_left = f.cores[d1 - 1].sizes(2); // right rank of core d1-1
  const index_t boundary_rank = Rb_left;

  // Build f2 as TT of length d2 (these are cores d1..d1+d2-1 in f)
  QuantizedTensorTrain<space, boba::complex<data_t>> f2(2, d2);

  // Copy the unchanged tail of f2: cores (d1+1 .. d1+d2-1) into f2.cores[1 .. d2-1]
  for (index_t i = d1 + 1; i < d1 + d2; ++i)
  {                                          // includes last core d1+d2-1
    f2.cores[i - (d1 + 1) + 1] = f.cores[i]; // map i=d1+1 -> f2[1]
  }

  // Copy the first (d1-1) cores of y (unchanged part)
  for (index_t i = 0; i < d1 - 1; ++i)
  {
    y.cores[i] = f.cores[i];
  }

  // Views for slicing
  auto core_view = f.cores[d1].view(); // boundary core for the f2-side
  auto idx1 = core_view.sizes(1);      // should be 2
  auto idx2 = core_view.sizes(2);      // right rank of core d1

  // f2.cores[0] will be a slice: (1, idx1, idx2)
  f2.cores[0].resize({1, idx1, idx2});
  auto f2_core0_view = f2.cores[0].view();

  // y.cores[d1-1] will be a slice: (idx1Y, idx2Y, 1)
  auto idx1Y = f.cores[d1 - 1].sizes(0);
  auto idx2Y = f.cores[d1 - 1].sizes(1);
  y.cores[d1 - 1].resize({idx1Y, idx2Y, 1});
  auto core_viewY = y.cores[d1 - 1].view();

  auto f_core_d1m1_view = f.cores[d1 - 1].view(); // (idx1Y, idx2Y, boundary_rank)

  // Loop over the boundary rank
  for (index_t k = 0; k < boundary_rank; ++k)
  {
    // Slice f.cores[d1] on LEFT rank: fcores{d1+1}(k,:,:) in MATLAB
    // i.e. set f2.cores[0](0,:,:) = f.cores[d1](k,:,:)
    boba::Array<index_t, 2> loop_upper_bounds{{idx1, idx2}};
    ::boba::loop<space, 2>(loop_upper_bounds,
                           [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      f2_core0_view({0, i, j}) = core_view({k, i, j});
    });

    // Apply FFT to the d2-block
    auto y2 = ifft(f2, tol);

    // Copy over the last d2 cores of y:
    // y.cores[d1] is the "first core of f2 slice" output, and y.cores[d1+1..] are the unchanged tail
    y.cores[d1] = y2.cores[0];

    // Mapping for tail cores (start at f2 index 1)
    for (index_t i = d1 + 1; i < d1 + d2; ++i)
    {
      y.cores[i] = y2.cores[i - d1]; // i=d1+1 -> y2[1]
    }

    // Slice f.cores[d1-1] on RIGHT rank: fcores{d1}(:,:,k) in MATLAB
    boba::Array<index_t, 2> loop_upper_bounds2{{idx1Y, idx2Y}};
    ::boba::loop<space, 2>(loop_upper_bounds2,
                           [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [i, j] = mid;
      core_viewY({i, j, 0}) = f_core_d1m1_view({i, j, k});
    });

    // Sum into the result (consider rounding here in practice)
    y_sums = y_sums + y;
  }

  return y_sums;
}

/**
 * @brief Function overload where we convert the qtt x to complex and call the complex version of qtt_column_extract.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_column_extract(QuantizedTensorTrain<space, data_t> x,
                   index_t d1,
                   index_t d2,
                   index_t j,
                   data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2");

  // Convert real QTT -> complex QTT with imag=0, then call the complex overload.
  return qtt_column_extract(to_complex<space, data_t>(x), d1, d2, j, tol);
}
/**
 * @brief TT_COLUMN_EXTRACT Extract column of TT tensor that's viewed as a matrix
 *  v = tt_column_extract(A, d1, d2, j) treats the first d1 dimensions
 *  of the (d1 + d2)-dimensional TT tensor A as rows and the last d2
 *   dimensions as columns, and extracts the TT tensor that represents the
 *   j-th column in this view.
 *
 *
 * @param A The input quantized tensor train to be transformed.
 *          It must have a base of 2 for the FFT to be valid.
 * @param tol A tolerance value used in the FFT computation.
 *
 * @return A quantized tensor train containing the FFT of the input tensor train.
 *
 * @note The function asserts that the base of the input tensor train is 2.
 *       Ensure that the input tensor train is properly initialized before calling this function.
 *
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>> qtt_column_extract(const QuantizedTensorTrain<space, boba::complex<data_t>>& A, index_t d1, index_t d2, index_t j, data_t tol)
{
  detail::ignore(tol);

  // m = A.n(1:d1), n = A.n(d1+1:d1+d2)
  std::vector<index_t> m(d1), n(d2);
  for (index_t i = 0; i < d1; ++i)
  {
    m[i] = A.sizes(i);
  }
  for (index_t i = 0; i < d2; ++i)
  {
    n[i] = A.sizes(d1 + i);
  }

  // r = A.r  (ranks length d+1)
  index_t d = d1 + d2; // should match A.exponent
  std::vector<index_t> ranks(d + 1);
  ranks[0] = A.get_ranks_left(0);
  for (index_t k = 0; k < d; ++k)
  {
    ranks[k + 1] = A.get_ranks_right(k);
  }

  // determine multi-index corresponding to j (0-based)
  std::vector<index_t> jmulti(d2);
  index_t tmp = j; // j is 0-based, so no "-1"
  for (index_t k = 0; k < d2; ++k)
  {
    jmulti[k] = tmp % n[k]; // 0 .. n[k]-1
    tmp /= n[k];
  }

  // collapse the last d2 dimensions using the multi-index
  // tempMat starts as slice of last core: A.cores[d-1](:, jmulti[d2-1], :)
  const auto& last_core = A.cores[d - 1];
  auto last_view = last_core.view();
  index_t rL = last_core.sizes(0);
  index_t rR = last_core.sizes(2);
  index_t jlast = jmulti[d2 - 1];

  boba::Matrix<space, boba::complex<data_t>> tempMat({rL, rR});
  auto temp_view = tempMat.view();

  ::boba::Array<index_t, 2> upperIDX{{rL, rR}};
  ::boba::loop<space, 2>(upperIDX,
                         [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
  {
    auto [a, b] = mid;
    // tempMat(a,b) = last_core(:, jlast, :)
    temp_view({a, b}) = last_view({a, jlast, b});
  });

  // for k = d-2 down to d1: tempMat = core(:, jmulti[k-d1], :) * tempMat
  for (index_t k = d - 2; k >= d1; --k)
  {
    const auto& core = A.cores[k];
    auto core_view = A.cores[k].view();
    index_t rk = core.sizes(0);
    index_t rkp1 = core.sizes(2);

    index_t jk = jmulti[k - d1]; // valid for k in [d1, d-2]

    boba::Matrix<space, boba::complex<data_t>> Mk({rk, rkp1});
    auto Mk_view = Mk.view();
    ::boba::Array<index_t, 2> upperMk{{rk, rkp1}};
    ::boba::loop<space, 2>(upperMk,
                           [=] __boba_host_device__(::boba::Array<index_t, 2> mid)
    {
      auto [a, b] = mid;
      Mk_view({a, b}) = core_view({a, jk, b});
    });

    tempMat = Mk * tempMat;
  }
  QuantizedTensorTrain<space, boba::complex<data_t>> v(2, d1);

  // Copy cores 0...d1-2
  for (index_t w = 0; w < d1 - 1; ++w)
  {
    auto src = A.cores[w].view();
    auto [r0, nk, r1] = src.sizes();

    v.cores[w].resize({r0, nk, r1});
    auto dst = v.cores[w].view();

    ::boba::loop<space, 3>(::boba::Array<index_t, 3>{r0, nk, r1},
                           [=] __boba_host_device__(::boba::Array<index_t, 3> ijk)
    {
      auto [ri, rj, rk] = ijk;
      dst({ri, rj, rk}) = src({ri, rj, rk});
    });
  }
  // last row-mode core index
  index_t krow = d1 - 1;

  auto A_view = A.cores[krow].view();
  index_t rLastL = A_view.sizes(0); // r_{d1-1}
  index_t mk = A_view.sizes(1);     // m(d1)
  index_t rLastR = A_view.sizes(2); // r_{d1}
  // v.core[krow] should be (rLastL, mk, 1)
  v.cores[krow].resize({rLastL, mk, 1_z});
  v.cores[krow].fill_with_zeros();
  auto Vcore = v.cores[krow].view();

  // Vcore(a,i,0) = sum_{s=0}^{rR-1} A_view(a,i,s) * tempMat(s,0)
  ::boba::Array<index_t, 2> upper{rLastL, mk};

  auto tempMat_view = tempMat.const_view();

  ::boba::loop<space, 2>(upper,
                         [=] __boba_host_device__(::boba::Array<index_t, 2> ai)
  {
    auto [a, i] = ai;

    boba::complex<data_t> sum = boba::complex<data_t>(0);
    for (index_t s = 0; s < rLastR; ++s)
    {
      sum += A_view({a, i, s}) * tempMat_view({s, 0});
    }

    Vcore({a, i, 0}) = sum;
  });
  // For a true TT/QTT tensor, the last TT-rank should be 1.
  // If it's not 1, then tempMat is not a vector and tempMat(s,0) would be wrong.
  boba_always_assert_equal(
    tempMat.sizes(1),
    1_z,
    "qtt_column_extract: expected final TT rank r_d = 1 so that the contracted object is a vector. "
    "Got tempMat with sizes (r_{d1}, r_d) where r_d != 1; using tempMat(s,0) would be incorrect.");

  // Now because we changed the vcore I think that should have changed v as well
  return v;
}

/**
 * @brief Function overload where we convert the qtt x to complex and call the complex version of qtt_column_extract.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_column_addinto(
  const QuantizedTensorTrain<space, data_t>& x,
  index_t d1,
  index_t d2,
  index_t j,
  const QuantizedTensorTrain<space, data_t>& v,
  data_t tol)
{
  boba_always_assert(x.base == 2, "Base must be 2");
  boba_always_assert_equal(x.exponent, d1 + d2, "qtt_column_addinto: x.exponent must equal d1+d2.");
  boba_always_assert_equal(v.exponent, d1, "qtt_column_addinto: v.exponent must equal d1.");

  // Convert real -> complex (imag=0) and dispatch to the complex overload.
  return qtt_column_addinto(
    to_complex<space, data_t>(x),
    d1,
    d2,
    j,
    to_complex<space, data_t>(v),
    tol);
}
/**
 * @brief Add a complex QTT vector into a single column of a complex QTT Tensor  (conceptually), returning the update term.
 * We think of the QTT tensor as A(i0,i1,...id1,id1+1,..id1+id2), so we think tof that as a matrix of row size d1 and col d2
 * this code will insert overwrite a col in that matrix, and return it
 */

template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, boba::complex<data_t>>
qtt_column_addinto(
  const QuantizedTensorTrain<space, boba::complex<data_t>>& A,
  index_t d1,
  index_t d2,
  index_t j,
  const QuantizedTensorTrain<space, boba::complex<data_t>>& v,
  data_t tol)
{
  detail::ignore(tol);
  index_t d = d1 + d2;

  // Sizes
  std::vector<index_t> m(d1), n(d2);
  for (index_t i = 0; i < d1; ++i)
  {
    m[i] = A.sizes(i);
  }
  for (index_t i = 0; i < d2; ++i)
  {
    n[i] = A.sizes(d1 + i);
  }

  // Check v row-dims match m
  for (index_t i = 0; i < d1; ++i)
  {
    boba_always_assert_equal(v.sizes(i), m[i], "qtt_column_addinto: v sizes do not match A row sizes.");
  }
  // Compute j multiindex (0-based)
  std::vector<index_t> jmulti(d2);
  index_t tmp = j;
  for (index_t k = 0; k < d2; ++k)
  {
    jmulti[k] = tmp % n[k];
    tmp /= n[k];
  }

  // Build padded TT V: first d1 cores = v.cores[0..d1-1], last d2 cores are one-hot
  QuantizedTensorTrain<space, boba::complex<data_t>> V(2, d);

  // Copy first d1 cores from v into V
  for (index_t k = 0; k < d1; ++k)
  {
    V.cores[k] = v.cores[k];
  }

  // Append d2 selector cores (1, n(k), 1) with a 1 at jmulti[k]
  for (index_t t = 0; t < d2; ++t)
  {
    index_t k = d1 + t; // core index in V
    index_t nk = n[t];
    index_t jt = jmulti[t];
    V.cores[k].resize({1_z, nk, 1_z});
    V.cores[k].fill_with_zeros();
    auto cv = V.cores[k].view();

    // set cv(0, jt, 0) = 1
    ::boba::loop<space, 1>(1_z,
                           [=] __boba_host_device__(index_t null)
    {
      boba::detail::ignore(null);
      cv({0_z, jt, 0_z}) = boba::complex<data_t>(1, 0);
    });
  }

  // Add and round
  auto Ap = A + V;
  Ap.round();

  return Ap;
}
/**
 * @brief Convert a real quantized tensor-train (QTT) to a complex QTT with zero imaginary part.
 *
 * This routine constructs a complex-valued QTT \(y\) from a real-valued QTT \(x\) by copying
 * every TT core entry into the real part and setting the imaginary part to zero:
 * \f[
 *   y = x + i\,0.
 * \f]
 *
 * The conversion preserves:
 * - the QTT base (must be 2),
 * - the exponent / number of cores,
 * - all TT ranks,
 * - the per-core dimensions (each core has mode size 2 in a base-2 QTT).
 *
 * The core data are copied in the requested execution space via @c ::boba::loop, and thus the
 * operation is suitable for host/device execution depending on @p space.

 * @return Complex-valued QTT with the same structure as @p x, where each element is
 * @pre @c x.base == 2. This is enforced with @c boba_always_assert.
 *
 * @note The output cores are allocated with extents \c {r_k, 2, r_{k+1}} for each core \c k,
 *       where the ranks \c r_k are read from @p x using @c get_ranks_left and @c get_ranks_right.
 * @note This is a value-returning conversion; it does not modify @p x.
 */
template <::boba::execution_space space, typename data_t>
boba::QuantizedTensorTrain<space, boba::complex<data_t>>
to_complex(const boba::QuantizedTensorTrain<space, data_t>& x)
{
  boba_always_assert(x.base == 2, "to_complex: Base must be 2");

  index_t d = x.exponent;

  // ranks
  std::vector<index_t> r(d + 1);
  for (index_t k = 0; k < d; ++k)
  {
    r[k] = x.get_ranks_left(k);
  }
  r[d] = x.get_ranks_right(d - 1);

  // allocate complex TT
  boba::QuantizedTensorTrain<space, boba::complex<data_t>> y(2, d);

  // copy cores with imag=0
  for (index_t k = 0; k < d; ++k)
  {
    index_t r0 = r[k];
    index_t r1 = r[k + 1];

    y.cores[k] = boba::Tensor<3, space, boba::complex<data_t>>({r0, 2_z, r1});

    auto xv = x.cores[k].const_view();
    auto yv = y.cores[k].view();

    ::boba::Array<index_t, 3> upper{{r0, 2_z, r1}};
    ::boba::loop<space, 3>(upper,
                           [=] __boba_host_device__(::boba::Array<index_t, 3> mid)
    {
      yv(mid) = boba::complex<data_t>{xv(mid), static_cast<data_t>(0)};
    });
  }

  return y;
}

/**
 * @brief Returns the tensor (aka kronecker) product of two qttm tensors
 * This is the equivalent to tensor product of the two matrices the qttm are estimating
 *
 * @param[in] left qttm
 * @param[in] right qttm
 * @return tensor product of left x right
 */

template <::boba::execution_space space, typename data_t>
boba::QuantizedTensorTrainMatrix<space, data_t>
tensor_product(
  const boba::QuantizedTensorTrainMatrix<space, data_t>& A,
  const boba::QuantizedTensorTrainMatrix<space, data_t>& B)
{
  boba_always_assert_equal(A.rows_base, B.rows_base, "qtt_kron: rows_base mismatch");
  boba_always_assert_equal(A.cols_base, B.cols_base, "qtt_kron: cols_base mismatch");

  const size_t d1 = A.exponent;
  const size_t d2 = B.exponent;

  boba::QuantizedTensorTrainMatrix<space, data_t> K(A.rows_base, A.cols_base, d1 + d2);

  for (size_t i = 0; i < d1; ++i)
  {
    K.cores[i] = A.cores[i];
  }
  for (size_t j = 0; j < d2; ++j)
  {
    K.cores[d1 + j] = B.cores[j];
  }

  return K;
}

// -------------------------------------------------------------------------
// sum and round
// -------------------------------------------------------------------------

/**
 * @brief Computes the sum of a sequence of quantized tensor trains,
 * rounding after each addition.
 *
 * Each tensor in @p sequence is added to an accumulator, followed by a rounding
 * operation to maintain a low-rank representation.
 *
 * @tparam space      Execution space (e.g., host or device).
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::span of QuantizedTensorTrain objects to be summed and rounded.
 * @return A new QuantizedTensorTrain representing the accumulated and rounded result.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, data_t>
sum_and_round(std::span<const QuantizedTensorTrain<space, data_t>> sequence)
{
  using qtt_t = QuantizedTensorTrain<space, data_t>;

  boba_always_assert(!sequence.empty(), "Cannot perform sum_and_round on an empty sequence.");

  // Use the first tensor to infer shape
  const auto& first = sequence.front();
  qtt_t output(first.base, first.exponent);

  for (const auto& item : sequence)
  {
    output += item;
    output.round();
  }

  return output;
}

/**
 * @brief Lightweight adapter for converting std::vector into std::span for
 * addition and rounding.
 *
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::vector of QuantizedTensorTrain objects to be summed and rounded.
 * @return A new QuantizedTensorTrain representing the accumulated and rounded result.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, data_t>
sum_and_round(const std::vector<QuantizedTensorTrain<space, data_t>>& sequence)
{
  using qtt_t = QuantizedTensorTrain<space, data_t>;
  return sum_and_round(std::span<const qtt_t>{sequence});
}

/**
 * @brief Lightweight adapter for converting std::initializer_list into
 * std::span for addition and rounding.
 *
 * @note This overload is convenient, but it may copy the input tensors into the
 * initializer-list backing array. Prefer the std::span or std::vector overloads
 * for performance-sensitive code.
 *
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence Initializer list of QuantizedTensorTrain objects to be summed and rounded.
 * @return A new QuantizedTensorTrain representing the accumulated and rounded result.
 */
template <::boba::execution_space space, typename data_t>
QuantizedTensorTrain<space, data_t>
sum_and_round(const std::initializer_list<QuantizedTensorTrain<space, data_t>> sequence)
{
  using qtt_t = QuantizedTensorTrain<space, data_t>;
  return sum_and_round(std::span<const qtt_t>{sequence.begin(), sequence.size()});
}

} // namespace boba
