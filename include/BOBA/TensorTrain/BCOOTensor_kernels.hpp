// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/tensors/BCOOTensor_functions.hpp"

#include <vector>

#ifdef BOBA_ENABLE_EIGEN
#include <Eigen/Dense>
#endif

/*
 * Tensor-train kernels for operator cores in block coordinate format, BCOOTensor<4> A(ra1, n, m, ra2): one dense
 * ra1 x ra2 block per stored (n, m) pair.  Overloads of the dense/COO kernels in TensorTrainSolver_utilities.hpp with
 * the same results and index orders; the block structure turns every contraction with A into dense matrix products
 * over the stored blocks.  Host-only.
 */

namespace boba
{

/**
 * \brief Environment update (amen_solve2's compute_next_Phi) for a block coordinate operator core.
 *
 * Same result and index order as the dense version:
 *   left to right:  Phi_out(rx2, ry2, ra2) = sum x(rx1, n, rx2) Phi_prev(rx1, ry1, ra1) A(ra1, n, m, ra2) y(ry1, m, ry2)
 *   right to left:  Phi_out(ry1, ra1, rx1) = sum y(ry1, m, ry2) Phi_prev(ry2, ra2, rx2) A(ra1, n, m, ra2) x(rx1, n, rx2)
 * then normalized as in compute_next_Phi.
 *
 * One sequential pass over the blocks (stored column-major in (n, m)).  Left to right, the blocks of one column share
 * y(:, m, :): their contributions are summed into one buffer that meets y_m in a single product.  The row factors
 * x_n^T Phi_prev (left to right) and the per-row accumulators (right to left) are formed on first use and released
 * after the last block of their row, so for operators local in a cell index only one cell's rows are alive at a time.
 */
template <size_t dimension, execution_space space, typename data_t>
std::pair<Tensor<3, space, data_t>, data_t> compute_next_Phi(
  const Tensor<3, space, data_t>& Phi_prev,
  const Tensor<3, space, data_t>& x,
  const BCOOTensor<4, space, data_t>& A,
  const Tensor<3, space, data_t>& y,
  bool sweep_left_right,
  data_t external_norm)
{
  BOBA_CALI_MARK
  static_assert(space == host_space, "BCOOTensor tensor-train kernels are host-only");
#ifndef BOBA_ENABLE_EIGEN
  boba_error("BCOOTensor tensor-train kernels require BOBA_ENABLE_EIGEN");
#else
  using MatX = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Stride = Eigen::OuterStride<>;
  using EIndex = Eigen::Index;
  const auto RA1 = static_cast<EIndex>(A.sizes(0));
  const auto N = static_cast<EIndex>(A.sizes(1));
  const auto M = static_cast<EIndex>(A.sizes(2));
  const auto RA2 = static_cast<EIndex>(A.sizes(3));
  const auto RX1 = static_cast<EIndex>(x.sizes(0));
  const auto RX2 = static_cast<EIndex>(x.sizes(2));
  const auto RY1 = static_cast<EIndex>(y.sizes(0));
  const auto RY2 = static_cast<EIndex>(y.sizes(2));
  boba_always_assert(x.sizes(1) == A.sizes(1) && y.sizes(1) == A.sizes(2),
                     "compute_next_Phi (BCOO): mode sizes do not match the core");
  const index_t blocks = A.nonzero_blocks();
  const auto block_size = static_cast<size_t>(RA1 * RA2);
  const data_t* block_data = A.m_values.data();
  const auto indices_view = A.m_indices.const_view();
  std::vector<index_t> last_use(static_cast<size_t>(N), 0);
  for (index_t q = 0; q < blocks; ++q)
  {
    last_use[static_cast<size_t>(indices_view({q, 0}))] = q;
  }

  Tensor<3, space, data_t> Phi_out;
  if (sweep_left_right)
  {
    boba_always_assert(static_cast<EIndex>(Phi_prev.sizes(0)) == RX1 && static_cast<EIndex>(Phi_prev.sizes(1)) == RY1 &&
                         static_cast<EIndex>(Phi_prev.sizes(2)) == RA1,
                       "compute_next_Phi (BCOO): left environment size mismatch");
    // G_n(ra1, (rx2, ry1)) = sum_rx1 Phi_prev(rx1, ry1, ra1) x(rx1, n, rx2)
    std::vector<MatX> G(static_cast<size_t>(N));
    auto row_factor = [&](EIndex n) -> const MatX&
    {
      MatX& g = G[static_cast<size_t>(n)];
      if (g.size() == 0)
      {
        g.resize(RA1, RX2 * RY1);
        Eigen::Map<const MatX, 0, Stride> Xn(x.data() + RX1 * n, RX1, RX2, Stride(RX1 * N));
        for (EIndex ry1 = 0; ry1 < RY1; ++ry1)
        {
          Eigen::Map<const MatX, 0, Stride> P(Phi_prev.data() + RX1 * ry1, RX1, RA1, Stride(RX1 * RY1));
          g.middleCols(RX2 * ry1, RX2).noalias() = P.transpose() * Xn;
        }
      }
      return g;
    };
    MatX Phi_t = MatX::Zero(RA2 * RX2, RY2);   // ((ra2, rx2), ry2), permuted to (rx2, ry2, ra2) at the end
    MatX U(RA2 * RX2, RY1);
    for (index_t q = 0; q < blocks;)
    {
      const index_t m = indices_view({q, 1});
      U.setZero();
      Eigen::Map<MatX> U_rows(U.data(), RA2, RX2 * RY1);
      for (; q < blocks && indices_view({q, 1}) == m; ++q)
      {
        const index_t n = indices_view({q, 0});
        Eigen::Map<const MatX> Anm(block_data + static_cast<size_t>(q) * block_size, RA1, RA2);
        U_rows.noalias() += Anm.transpose() * row_factor(static_cast<EIndex>(n));
        if (last_use[static_cast<size_t>(n)] == q)
        {
          MatX().swap(G[static_cast<size_t>(n)]);
        }
      }
      Eigen::Map<const MatX, 0, Stride> Ym(y.data() + RY1 * static_cast<EIndex>(m), RY1, RY2, Stride(RY1 * M));
      Phi_t.noalias() += U * Ym;
    }
    Phi_out = Tensor<3, space, data_t>({static_cast<index_t>(RX2), static_cast<index_t>(RY2), static_cast<index_t>(RA2)});
    data_t* out = Phi_out.data();
    for (EIndex ra2 = 0; ra2 < RA2; ++ra2)
    {
      for (EIndex ry2 = 0; ry2 < RY2; ++ry2)
      {
        for (EIndex rx2 = 0; rx2 < RX2; ++rx2)
        {
          out[rx2 + RX2 * (ry2 + RY2 * ra2)] = Phi_t(ra2 + RA2 * rx2, ry2);
        }
      }
    }
  }
  else
  {
    boba_always_assert(static_cast<EIndex>(Phi_prev.sizes(0)) == RY2 && static_cast<EIndex>(Phi_prev.sizes(1)) == RA2 &&
                         static_cast<EIndex>(Phi_prev.sizes(2)) == RX2,
                       "compute_next_Phi (BCOO): right environment size mismatch");
    // Phi_prev permuted once to (ry2, (rx2, ra2)) so that H_m = y_m Phi' has rows ry1 and columns (rx2, ra2)
    MatX P(RY2, RX2 * RA2);
    {
      const data_t* p = Phi_prev.data();
      for (EIndex rx2 = 0; rx2 < RX2; ++rx2)
      {
        for (EIndex ra2 = 0; ra2 < RA2; ++ra2)
        {
          for (EIndex ry2 = 0; ry2 < RY2; ++ry2)
          {
            P(ry2, rx2 + RX2 * ra2) = p[ry2 + RY2 * (ra2 + RA2 * rx2)];
          }
        }
      }
    }
    std::vector<MatX> V(static_cast<size_t>(N));   // V_n((ry1, rx2), ra1), per row until its last block
    Phi_out = Tensor<3, space, data_t>({static_cast<index_t>(RY1), static_cast<index_t>(RA1), static_cast<index_t>(RX1)});
    Phi_out.fill_with_zeros();
    data_t* out = Phi_out.data();
    MatX H;
    for (index_t q = 0; q < blocks;)
    {
      const index_t m = indices_view({q, 1});
      Eigen::Map<const MatX, 0, Stride> Ym(y.data() + RY1 * static_cast<EIndex>(m), RY1, RY2, Stride(RY1 * M));
      H.noalias() = Ym * P;
      Eigen::Map<const MatX> H_rows(H.data(), RY1 * RX2, RA2);
      for (; q < blocks && indices_view({q, 1}) == m; ++q)
      {
        const index_t n = indices_view({q, 0});
        Eigen::Map<const MatX> Anm(block_data + static_cast<size_t>(q) * block_size, RA1, RA2);
        MatX& v = V[static_cast<size_t>(n)];
        if (v.size() == 0)
        {
          v = MatX::Zero(RY1 * RX2, RA1);
        }
        v.noalias() += H_rows * Anm.transpose();
        if (last_use[static_cast<size_t>(n)] == q)
        {
          // Phi_out(ry1, ra1, rx1) += sum_rx2 V_n(ry1, rx2, ra1) x(rx1, n, rx2)
          Eigen::Map<const MatX, 0, Stride> Xn(x.data() + RX1 * static_cast<EIndex>(n), RX1, RX2, Stride(RX1 * N));
          for (EIndex ra1 = 0; ra1 < RA1; ++ra1)
          {
            Eigen::Map<const MatX> V_ra1(v.data() + RY1 * RX2 * ra1, RY1, RX2);
            Eigen::Map<MatX, 0, Stride> O(out + RY1 * ra1, RY1, RX1, Stride(RY1 * RA1));
            O.noalias() += V_ra1 * Xn.transpose();
          }
          MatX().swap(v);
        }
      }
    }
  }

  data_t phi_norm = 0.0;
  if (external_norm > 0.0)
  {
    Phi_out *= 1.0 / external_norm;
  }
  else
  {
    phi_norm = ::boba::norm_frobenius(Phi_out);
    if (not(is_tiny(phi_norm)))
    {
      Phi_out *= 1.0 / phi_norm;
    }
    else
    {
      phi_norm = 1.0;
    }
  }
  return std::make_pair(Phi_out, phi_norm);
#endif
}

/**
 * \brief Local operator application (amen_solve2's bfun3) for a block coordinate operator core.
 *
 * Same result as the dense bfun3: y(ry1, n, ry2) = sum Phi1(ry1, rx1, ra1) A(ra1, n, m, ra2) x(rx1, m, rx2)
 * Phi2(rx2, ra2, ry2).  The middle contraction runs over the stored blocks only, one product per block.
 */
template <execution_space space, typename data_t>
Tensor<3, space, data_t> bfun3(
  const Tensor<3, space, data_t>& Phi1,
  const BCOOTensor<4, space, data_t>& A,
  const Tensor<3, space, data_t>& Phi2,
  const Tensor<3, space, data_t>& x)
{
  BOBA_CALI_MARK
  static_assert(space == host_space, "BCOOTensor tensor-train kernels are host-only");
#ifndef BOBA_ENABLE_EIGEN
  boba_error("BCOOTensor tensor-train kernels require BOBA_ENABLE_EIGEN");
#else
  using MatX = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  using Stride = Eigen::OuterStride<>;
  using EIndex = Eigen::Index;
  auto xPhi2 = tensor_contraction<1>(
    {"rx1", "m", "rx2"}, x, {"rx2", "ra2", "ry2"}, Phi2, {"rx1", "m", "ra2", "ry2"});
  const auto X1 = static_cast<EIndex>(xPhi2.sizes(0));
  const auto M = static_cast<EIndex>(xPhi2.sizes(1));
  const auto R2 = static_cast<EIndex>(xPhi2.sizes(2));
  const auto Y2 = static_cast<EIndex>(xPhi2.sizes(3));
  const auto R1 = static_cast<EIndex>(A.sizes(0));
  const auto N = static_cast<EIndex>(A.sizes(1));
  boba_always_assert(M == static_cast<EIndex>(A.sizes(2)) && R2 == static_cast<EIndex>(A.sizes(3)),
                     "bfun3 (BCOO): size mismatch");
  // Y_m(ra2, (rx1, ry2)) = xPhi2(rx1, m, ra2, ry2), contiguous per m
  std::vector<MatX> Y(static_cast<size_t>(M));
  const data_t* xp = xPhi2.data();
  for (EIndex m = 0; m < M; ++m)
  {
    MatX Ym(R2, X1 * Y2);
    for (EIndex ry2 = 0; ry2 < Y2; ++ry2)
    {
      for (EIndex ra2 = 0; ra2 < R2; ++ra2)
      {
        for (EIndex rx1 = 0; rx1 < X1; ++rx1)
        {
          Ym(ra2, rx1 + X1 * ry2) = xp[rx1 + X1 * (m + M * (ra2 + R2 * ry2))];
        }
      }
    }
    Y[static_cast<size_t>(m)] = std::move(Ym);
  }
  Tensor<4, space, data_t> AxPhi2({static_cast<index_t>(R1), static_cast<index_t>(N), static_cast<index_t>(X1),
                                   static_cast<index_t>(Y2)});
  AxPhi2.fill_with_zeros();
  data_t* out = AxPhi2.data();
  const data_t* block_data = A.m_values.data();
  const auto indices_view = A.m_indices.const_view();
  const auto block_size = static_cast<size_t>(R1 * R2);
  for (index_t q = 0; q < A.nonzero_blocks(); ++q)
  {
    const auto n = static_cast<EIndex>(indices_view({q, 0}));
    const auto m = static_cast<size_t>(indices_view({q, 1}));
    Eigen::Map<const MatX> Anm(block_data + static_cast<size_t>(q) * block_size, R1, R2);
    Eigen::Map<MatX, 0, Stride> On(out + R1 * n, R1, X1 * Y2, Stride(R1 * N));
    On.noalias() += Anm * Y[m];
  }
  return tensor_contraction<2>(
    {"ry1", "rx1", "ra1"}, Phi1, {"ra1", "n", "rx1", "ry2"}, AxPhi2, {"ry1", "n", "ry2"});
#endif
}

/**
 * \brief Dense projected matrix Phi1 A Phi2 for a block coordinate operator core (via the dense core).
 *
 * Used where a dense local system is assembled anyway; the operator core is expanded once.
 */
template <execution_space space, typename data_t>
Matrix<space, data_t> bfun3_matrix(
  const Tensor<3, space, data_t>& Phi1,
  const BCOOTensor<4, space, data_t>& A,
  const Tensor<3, space, data_t>& Phi2)
{
  return bfun3_matrix(Phi1, to_dense(A), Phi2);
}

} // namespace boba
