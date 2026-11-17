// Benchmark: AMEn environment update (compute_next_Phi) with a block-sparse operator core, three ways:
//   dense Tensor<4> core, Pierson's SparseTensor<4> (COO) core via the generic compute_next_Phi, and the packed
//   CoreSliceSparsity core via compute_next_Phi_packed.  The core has the explicit-S pattern: cells coupled to their
//   neighbours (|ci - cj| <= 1), nb basis functions per cell, every nonzero (n, m) slice a dense ra1 x ra2 block.
// Usage: bench_phi_sparse_vs_packed ra1 ra2 cells nb rx ry [skip_dense]
#include "BOBA/boba.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace boba;
using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

int main(int argc, char **argv)
{
  boba::init();
  if (argc < 7) { std::printf("usage: ra1 ra2 cells nb rx ry [skip_dense]\n"); return 1; }
  const index_t RA1 = std::atol(argv[1]), RA2 = std::atol(argv[2]), NC = std::atol(argv[3]), NB = std::atol(argv[4]);
  const index_t RX = std::atol(argv[5]), RY = std::atol(argv[6]);
  const bool skipDense = argc > 7 && std::atoi(argv[7]) != 0;
  const index_t N = NC * NB;
  std::mt19937 gen(7); std::normal_distribution<double> nd;

  // packed core: column-major slice order (m outer, then n)
  CoreSliceSparsity<double> P; P.ra1 = RA1; P.n = N; P.m = N; P.ra2 = RA2;
  for (index_t m = 0; m < N; ++m)
    for (index_t n = 0; n < N; ++n)
      if (std::abs(long(n / NB) - long(m / NB)) <= 1) P.pairs.push_back({n, m});
  const size_t blk = size_t(RA1 * RA2);
  P.blocks.resize(P.pairs.size() * blk);
  for (auto &v : P.blocks) v = nd(gen);
  const double nnz = double(P.blocks.size());
  std::printf("core %ld x %ld x %ld x %ld, %zu of %ld slices (%.1f%%), nnz %.3g (packed %.2f GB, COO %.2f GB, dense %.2f GB)\n",
              long(RA1), long(N), long(N), long(RA2), P.pairs.size(), long(N * N), 100.0 * P.pairs.size() / double(N * N), nnz,
              nnz * 8 / 1e9, nnz * 16 / 1e9, double(RA1) * N * N * RA2 * 8 / 1e9);

  // COO sparse core (Pierson's format): column-major linear indices
  Vector<host_space, index_t> lin({index_t(nnz)}); Vector<host_space, double> val({index_t(nnz)});
  {
    auto lv = lin.view(); auto vv = val.view(); index_t e = 0;
    for (size_t q = 0; q < P.pairs.size(); ++q)
    {
      const index_t n = P.pairs[q].first, m = P.pairs[q].second;
      for (index_t b = 0; b < RA2; ++b)
        for (index_t a = 0; a < RA1; ++a, ++e)
        {
          lv({e}) = a + RA1 * (n + N * (m + N * b));
          vv({e}) = P.blocks[q * blk + size_t(a + RA1 * b)];
        }
    }
  }
  const auto tc = clk::now();
  SparseTensor<4, host_space, double> Sp({RA1, N, N, RA2}, lin, val);
  std::printf("COO construction %.2f s\n", since(tc));

  Tensor<3, host_space, double> x({RX, N, RX}), y({RY, N, RY});
  for (index_t i = 0; i < x.size(); ++i) x.data()[i] = nd(gen);
  for (index_t i = 0; i < y.size(); ++i) y.data()[i] = nd(gen);
  Tensor<3, host_space, double> PhiL({RX, RY, RA1}), PhiR({RY, RA2, RX});
  for (index_t i = 0; i < PhiL.size(); ++i) PhiL.data()[i] = nd(gen);
  for (index_t i = 0; i < PhiR.size(); ++i) PhiR.data()[i] = nd(gen);

  for (bool lr : {true, false})
  {
    const auto &Phi = lr ? PhiL : PhiR;
    const auto t1 = clk::now();
    auto pk = compute_next_Phi_packed<4>(Phi, x, P, y, lr, 1.0);
    const double tP = since(t1);
    const auto t2 = clk::now();
    auto sp = compute_next_Phi<4>(Phi, x, Sp, y, lr, 1.0);
    const double tS = since(t2);
    const double diff = norm_difference_frobenius(pk.first, sp.first) / norm_frobenius(pk.first);
    std::printf("%s: packed %.3f s, SparseTensor %.3f s (%.1fx), rel. difference %.1e", lr ? "left-to-right" : "right-to-left",
                tP, tS, tS / tP, diff);
    if (!skipDense)
    {
      Tensor<4, host_space, double> D = densify_core_slices(P);
      const auto t3 = clk::now();
      auto dn = compute_next_Phi<4>(Phi, x, D, y, lr, 1.0);
      std::printf(", dense %.3f s, dense vs packed %.1e", since(t3), norm_difference_frobenius(pk.first, dn.first) / norm_frobenius(pk.first));
    }
    std::printf("\n");
  }
  return 0;
}
