// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <vector>

#include "BOBA/boba.hpp"

namespace boba
{

/**
 * \brief
 * Tensor train whose number of cores is chosen at run time.
 *
 * Same representation and algorithms as TensorTrain<dimension, space, data_t>, but the cores are held in a
 * std::vector, so the dimension can come from an input file. Every operation loops over the cores on the host
 * and calls the same per-core kernels as TensorTrain (unfold/fold, QR, SVD, tensor_contraction, add_subcore),
 * so the core work runs in the given execution space. Operations that need a full multi-index of compile-time
 * length (decompression, static views) are available by converting to TensorTrain<dimension> with to_static().
 *
 * Core d has shape (r_d, n_d, r_{d+1}) with r_0 = r_D = 1.
 */

template <::boba::execution_space space, typename _data_t>
struct RuntimeTensorTrain
{
  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;
  using core_t = Tensor<3, space, data_t>;

  std::string m_name = "RuntimeTensorTrain";
  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);
  std::vector<size_t> max_ranks;   ///< per interface (dimension + 1 entries); empty: unbounded
  std::vector<core_t> cores;

  RuntimeTensorTrain() = default;

  /**
   * \brief
   * Zero tensor train (rank one) with the given mode sizes.
   */

  explicit RuntimeTensorTrain(std::vector<index_t> const& sizes)
  {
    resize(sizes);
  }

  /**
   * \brief
   * Copy the cores of a compile-time-dimension tensor train.
   */

  template <size_t dimension>
  explicit RuntimeTensorTrain(TensorTrain<dimension, space, data_t> const& train)
      : m_name(train.m_name),
        svd_tolerance_relative(train.svd_tolerance_relative),
        svd_tolerance_absolute(train.svd_tolerance_absolute)
  {
    cores.reserve(dimension);
    for (size_t d = 0; d < dimension; d++)
    {
      cores.push_back(train.cores[d]);
    }
  }

  /**
   * \brief
   * Convert to a compile-time-dimension tensor train (dimension must equal get_dimension()).
   */

  template <size_t dimension>
  TensorTrain<dimension, space, data_t> to_static() const
  {
    boba_always_assert_equal(dimension, get_dimension(), "to_static: dimension mismatch");
    TensorTrain<dimension, space, data_t> train;
    train.m_name = m_name;
    train.svd_tolerance_relative = svd_tolerance_relative;
    train.svd_tolerance_absolute = svd_tolerance_absolute;
    for (size_t d = 0; d < dimension; d++)
    {
      train.cores[d] = cores[d];
    }
    return train;
  }

  void resize(std::vector<index_t> const& sizes)
  {
    cores.resize(sizes.size());
    for (size_t d = 0; d < sizes.size(); d++)
    {
      cores[d].resize({1_z, sizes[d], 1_z});
    }
    fill_with_zeros();
  }

  void rename(std::string_view new_name)
  {
    m_name = new_name;
  }

  // -------------------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------------------

  size_t get_dimension() const noexcept
  {
    return cores.size();
  }

  index_t sizes(size_t d) const
  {
    return cores[d].sizes(1);
  }

  std::vector<index_t> sizes() const
  {
    std::vector<index_t> s(get_dimension());
    for (size_t d = 0; d < get_dimension(); d++)
    {
      s[d] = sizes(d);
    }
    return s;
  }

  size_t get_ranks_left(size_t d) const
  {
    return cores[d].sizes(0);
  }

  size_t get_ranks_right(size_t d) const
  {
    return cores[d].sizes(2);
  }

  /**
   * \brief
   * Interface ranks r_0 .. r_D.
   */

  std::vector<size_t> ranks() const
  {
    std::vector<size_t> r(get_dimension() + 1, 1_z);
    for (size_t d = 0; d < get_dimension(); d++)
    {
      r[d] = get_ranks_left(d);
    }
    return r;
  }

  size_t max_rank() const
  {
    size_t r = 1;
    for (size_t d = 0; d < get_dimension(); d++)
    {
      r = std::max(r, get_ranks_right(d));
    }
    return r;
  }

  /**
   * \brief
   * Number of stored entries over all cores.
   */

  size_t get_number_elements() const
  {
    size_t n = 0;
    for (auto const& core : cores)
    {
      n += core.size();
    }
    return n;
  }

  // -------------------------------------------------------------------------------------
  // Filling
  // -------------------------------------------------------------------------------------

  void fill_with(data_t x)
  {
    if (cores.empty())
    {
      return;
    }
    set_to_rank_one_scalar_core(cores[0], x);
    for (size_t d = 1; d < get_dimension(); d++)
    {
      set_to_rank_one_scalar_core(cores[d], 1.0);
    }
  }

  void fill_with_zeros()
  {
    for (auto& core : cores)
    {
      set_to_rank_one_scalar_core(core, 0.0);
    }
  }

  // -------------------------------------------------------------------------------------
  // Arithmetic (exact; ranks add / multiply, use round() to recompress)
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * this += rhs (block-diagonal core concatenation; interface ranks add).
   */

  RuntimeTensorTrain& operator+=(RuntimeTensorTrain const& rhs)
  {
    BOBA_CALI_MARK
    boba_always_assert_equal(get_dimension(), rhs.get_dimension(), "incompatible dimensions");
    for (size_t d = 0; d < get_dimension(); d++)
    {
      boba_always_assert_equal(sizes(d), rhs.sizes(d), "incompatible sizes");
    }
    for (size_t d = 0; d < get_dimension(); d++)
    {
      add_subcore(cores[d], rhs.cores[d], d, get_dimension(), 0_z);
    }
    return *this;
  }

  RuntimeTensorTrain operator+(RuntimeTensorTrain const& rhs) const
  {
    RuntimeTensorTrain output{*this};
    output += rhs;
    return output;
  }

  RuntimeTensorTrain& operator*=(data_t const scalar)
  {
    if (!cores.empty())
    {
      cores[0] *= scalar;
    }
    return *this;
  }

  RuntimeTensorTrain operator*(data_t const scalar) const
  {
    RuntimeTensorTrain output{*this};
    output *= scalar;
    return output;
  }

  RuntimeTensorTrain& operator-=(RuntimeTensorTrain const& rhs)
  {
    RuntimeTensorTrain minus_rhs{rhs};
    minus_rhs *= static_cast<data_t>(-1);
    return *this += minus_rhs;
  }

  RuntimeTensorTrain operator-(RuntimeTensorTrain const& rhs) const
  {
    RuntimeTensorTrain output{*this};
    output -= rhs;
    return output;
  }

  // -------------------------------------------------------------------------------------
  // Rounding
  // -------------------------------------------------------------------------------------

  /**
   * \brief
   * Left-orthogonalize by a left-to-right QR sweep (cores 0..D-2 left-orthogonal).
   */

  void orthogonalize()
  {
    BOBA_CALI_MARK
    const size_t dimension = get_dimension();
    if (dimension < 2)
    {
      return;
    }
    boba::QR<space, data_t> qr;
    for (size_t d = 0; d + 1 < dimension; d++)
    {
      {
        auto unfold_left = compute_unfold_left(cores[d]);
        qr(unfold_left);
      }
      cores[d] = write_to_core_from_left_fold(qr.Q, sizes(d));
      cores[d + 1] = boba::tensor_contraction<1>(
        {"k", "l"}, qr.R, {"l", "i", "r"}, cores[d + 1], {"k", "i", "r"});
    }
  }

  /**
   * \brief
   * Round: QR left-orthogonalization, then a right-to-left truncated SVD sweep with svd_tolerance_relative,
   * svd_tolerance_absolute and max_ranks (the algorithm of TensorTrain::round). The first core is the
   * orthogonality center afterwards.
   */

  void round()
  {
    BOBA_CALI_MARK
    const size_t dimension = get_dimension();
    if (dimension < 2)
    {
      return;
    }
    boba::SVD<space, data_t> svd;
    svd.tolerance_relative = svd_tolerance_relative;
    svd.tolerance_absolute = svd_tolerance_absolute;
    orthogonalize();
    for (size_t d = dimension - 1; d > 0; d--)
    {
      svd.max_kept_singular_values = max_ranks.empty() ? highest_value<size_t>() : max_ranks[d];
      {
        auto unfold_right = compute_unfold_right(cores[d]);
        svd(unfold_right);
      }
      if (svd.significant_singular_values == 0_z)
      {
        fill_with_zeros();
        return;
      }
      {
        auto unfold_right = svd.V.transpose();
        cores[d] = write_to_core_from_right_fold(unfold_right, sizes(d));
      }
      apply_as_diagonal_right_in_place(svd.S, svd.U);
      cores[d - 1] = boba::tensor_contraction<1>(
        {"l", "i", "k"}, cores[d - 1], {"k", "r"}, svd.U, {"l", "i", "r"});
    }
  }

  /**
   * \brief
   * Round with a given relative tolerance (sets svd_tolerance_relative, absolute tolerance zero).
   */

  void round(real_data_t relative_tolerance)
  {
    svd_tolerance_relative = relative_tolerance;
    svd_tolerance_absolute = static_cast<real_data_t>(0);
    round();
  }

  /**
   * \brief
   * Round to relative Frobenius accuracy: ||A - round(A)||_F <= relative_tolerance * ||A||_F (TT-SVD bound).
   *
   * Same QR + right-to-left SVD sweep as round(), but each split keeps the smallest rank whose discarded
   * singular-value tail has energy <= delta^2, delta = relative_tolerance * ||A||_F / sqrt(D - 1) (the TT-Toolbox
   * my_chop2 rule), instead of thresholding each singular value against the largest. max_ranks still applies.
   * The singular values are copied to the host (their number is the rank) to choose the cut.
   */

  void round_frobenius(real_data_t relative_tolerance)
  {
    BOBA_CALI_MARK
    const size_t dimension = get_dimension();
    if (dimension < 2)
    {
      return;
    }
    orthogonalize();
    // After left-orthogonalization ||A||_F is the Frobenius norm of the last core.
    const real_data_t norm = boba::norm_frobenius(cores[dimension - 1]);
    const real_data_t delta = relative_tolerance * norm / std::sqrt(static_cast<real_data_t>(dimension - 1));

    boba::SVD<space, data_t> svd;
    svd.tolerance_relative = static_cast<real_data_t>(0);
    svd.tolerance_absolute = static_cast<real_data_t>(0);
    for (size_t d = dimension - 1; d > 0; d--)
    {
      {
        auto unfold_right = compute_unfold_right(cores[d]);
        svd(unfold_right);
      }
      if (svd.significant_singular_values == 0_z)
      {
        fill_with_zeros();
        return;
      }
      const boba::Vector<host_space, real_data_t> sigma(svd.S);
      auto sigma_view = sigma.const_view();
      size_t keep = sigma.size();
      real_data_t tail = static_cast<real_data_t>(0);
      while (keep > 1)
      {
        const real_data_t s = sigma_view(keep - 1);
        if (tail + s * s > delta * delta)
        {
          break;
        }
        tail += s * s;
        keep--;
      }
      if (not max_ranks.empty())
      {
        keep = boba::min(keep, max_ranks[d]);
      }
      if (keep < sigma.size())
      {
        svd.S.resize({keep});
        svd.U.resize({svd.U.rows(), keep});
        svd.V.resize({svd.V.rows(), keep});
      }
      {
        auto unfold_right = svd.V.transpose();
        cores[d] = write_to_core_from_right_fold(unfold_right, sizes(d));
      }
      apply_as_diagonal_right_in_place(svd.S, svd.U);
      cores[d - 1] = boba::tensor_contraction<1>(
        {"l", "i", "k"}, cores[d - 1], {"k", "r"}, svd.U, {"l", "i", "r"});
    }
  }
};

// -------------------------------------------------------------------------------------
// Free functions
// -------------------------------------------------------------------------------------

template <::boba::execution_space space, typename data_t>
RuntimeTensorTrain<space, data_t> operator*(data_t scalar, RuntimeTensorTrain<space, data_t> const& train)
{
  return train * scalar;
}

/**
 * \brief
 * Elementwise (Hadamard) product; interface ranks multiply.
 */

template <::boba::execution_space space, typename data_t>
RuntimeTensorTrain<space, data_t> elementwise_product(
  RuntimeTensorTrain<space, data_t> const& train_A,
  RuntimeTensorTrain<space, data_t> const& train_B)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(train_A.get_dimension(), train_B.get_dimension(), "incompatible dimensions");
  RuntimeTensorTrain<space, data_t> output;
  output.cores.resize(train_A.get_dimension());
  for (size_t d = 0; d < train_A.get_dimension(); d++)
  {
    boba_always_assert_equal(train_A.sizes(d), train_B.sizes(d), "sizes must match");
    auto rank_left_indexer = ::boba::Multiindexer<2>({train_A.get_ranks_left(d), train_B.get_ranks_left(d)});
    auto rank_right_indexer = ::boba::Multiindexer<2>({train_A.get_ranks_right(d), train_B.get_ranks_right(d)});
    auto A_core_view = train_A.cores[d].const_view();
    auto B_core_view = train_B.cores[d].const_view();
    output.cores[d].resize({rank_left_indexer.size(), train_A.sizes(d), rank_right_indexer.size()});
    auto output_view = output.cores[d].view();
    ::boba::loop<space, 3>(output_view.sizes(),
                           [=] __boba_host_device__(::boba::Array<size_t, 3> ijk)
    {
      auto [rank_left, index, rank_right] = ijk;
      auto [rank_A_left, rank_B_left] = rank_left_indexer.multiindex(rank_left);
      auto [rank_A_right, rank_B_right] = rank_right_indexer.multiindex(rank_right);
      output_view({rank_left, index, rank_right}) =
        A_core_view({rank_A_left, index, rank_A_right}) * B_core_view({rank_B_left, index, rank_B_right});
    });
  }
  return output;
}

/**
 * \brief
 * Sum of all entries weighted by one vector per core: sum_{i_0..i_{D-1}} prod_d w_d[i_d] T[i_0, ..., i_{D-1}].
 * The weights live on the host; contraction is done core by core.
 */

template <::boba::execution_space space, typename data_t>
data_t contract_with_vectors(
  RuntimeTensorTrain<space, data_t> const& train,
  std::vector<std::vector<data_t>> const& weights)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(weights.size(), train.get_dimension(), "one weight vector per core");
  boba::Vector<space, data_t> carry({1_z});
  carry.fill_with(1.0);
  for (size_t d = 0; d < train.get_dimension(); d++)
  {
    boba_always_assert_equal(weights[d].size(), train.sizes(d), "weight vector size");
    boba::Vector<host_space, data_t> w_host({weights[d].size()});
    auto w_host_view = w_host.view();
    for (size_t i = 0; i < weights[d].size(); i++)
    {
      w_host_view(i) = weights[d][i];
    }
    boba::Vector<space, data_t> w(w_host);
    // M_d[l, r] = sum_i w[i] core[l, i, r]
    const size_t rl = train.get_ranks_left(d), n = train.sizes(d), rr = train.get_ranks_right(d);
    boba::Matrix<space, data_t> Md({rl, rr});
    auto Md_view = Md.view();
    auto core_view = train.cores[d].const_view();
    auto w_view = w.const_view();
    ::boba::loop<space, 2>({rl, rr},
                           [=] __boba_host_device__(::boba::Array<size_t, 2> lr)
    {
      auto value = ::boba::PotentiallyComplex<data_t>::value(0.0);
      for (size_t i = 0; i < n; i++)
      {
        value += w_view(i) * core_view({lr[0], i, lr[1]});
      }
      Md_view(lr) = value;
    });
    boba::Matrix<space, data_t> carry_row({1_z, rl});
    carry_row.reshape(carry);
    auto next = carry_row * Md;
    carry.resize({rr});
    carry.reshape(next);
  }
  boba_always_assert_equal(carry.size(), 1_z, "end result should be a scalar");
  return carry.sum_reduce();
}

}  // namespace boba
