// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <random>

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::execution_space host = boba::host_space;

namespace
{

// A TT-matrix core A(ra1, n, m, ra2) with the sparsity of cell-local operators: n and m index (cell, basis) pairs, and
// cells couple only to their neighbours, so a fraction of the (n, m) slices are nonzero, each a dense ra1 x ra2 block.
boba::Tensor<4, host, double> make_block_sparse_core(boba::index_t ra1, boba::index_t cells, boba::index_t basis,
                                                     boba::index_t ra2, unsigned seed)
{
  std::mt19937 generator(seed);
  std::normal_distribution<double> normal;
  const boba::index_t n = cells * basis;
  boba::Tensor<4, host, double> core({ra1, n, n, ra2});
  core.fill_with_zeros();
  auto core_view = core.view();
  for (boba::index_t col = 0; col < n; ++col)
  {
    for (boba::index_t row = 0; row < n; ++row)
    {
      const auto row_cell = static_cast<long>(row / basis);
      const auto col_cell = static_cast<long>(col / basis);
      if (std::abs(row_cell - col_cell) > 1)
      {
        continue;
      }
      for (boba::index_t b = 0; b < ra2; ++b)
      {
        for (boba::index_t a = 0; a < ra1; ++a)
        {
          core_view({a, row, col, b}) = normal(generator);
        }
      }
    }
  }
  return core;
}

template <size_t dimension>
boba::Tensor<dimension, host, double> random_tensor(boba::Array<boba::index_t, dimension> sizes, unsigned seed)
{
  std::mt19937 generator(seed);
  std::normal_distribution<double> normal;
  boba::Tensor<dimension, host, double> tensor(sizes);
  double* data = tensor.data();
  for (boba::index_t i = 0; i < tensor.size(); ++i)
  {
    data[i] = normal(generator);
  }
  return tensor;
}

void test_format(bool& check)
{
  const auto dense = make_block_sparse_core(3, 4, 2, 2, 11);
  const boba::Tensor<4, space, double> dense_space(dense);

  // Dense -> BCOO keeps exactly the nonzero (n, m) blocks: 4 cells, neighbours only -> (4 + 2 * 3) * 2 * 2 = 40.
  auto bcoo = boba::from_dense_bcoo(dense_space);
  pass_or_fail_bool(check, bcoo.nonzero_blocks() == 40);
  pass_or_fail_bool(check, bcoo.nnz() == 40 * 3 * 2);
  pass_or_fail(check, boba::norm_difference_frobenius(boba::Tensor<4, host, double>(boba::to_dense(bcoo)), dense), 1e-15);

  // Blocks are sorted column-major in (n, m): m nondecreasing, n increasing within a column.
  {
    boba::BCOOTensor<4, host, double> const host_bcoo = bcoo;
    auto indices = host_bcoo.m_indices.const_view();
    bool sorted = true;
    for (boba::index_t k = 1; k < host_bcoo.nonzero_blocks(); ++k)
    {
      const bool same_column = indices({k, 1}) == indices({k - 1, 1});
      sorted = sorted && (indices({k, 1}) > indices({k - 1, 1}) || (same_column && indices({k, 0}) > indices({k - 1, 0})));
    }
    pass_or_fail_bool(check, sorted);
  }

  // BCOO <-> COO round trip.
  auto coo = boba::convert_to_SparseTensor(bcoo);
  pass_or_fail_bool(check, coo.nnz() == bcoo.nnz());
  auto bcoo_again = boba::convert_to_BCOOTensor(coo);
  pass_or_fail_bool(check, bcoo_again.nonzero_blocks() == bcoo.nonzero_blocks());
  pass_or_fail(check, boba::norm_difference_frobenius(boba::Tensor<4, host, double>(boba::to_dense(bcoo_again)), dense), 1e-15);

  // Unsorted coordinates with a duplicate are sorted and summed.
  {
    boba::Matrix<host, boba::index_t> indices({3, 2});
    auto indices_view = indices.view();
    indices_view({0, 0}) = 1;
    indices_view({0, 1}) = 1;
    indices_view({1, 0}) = 0;
    indices_view({1, 1}) = 1;
    indices_view({2, 0}) = 1;
    indices_view({2, 1}) = 1;
    boba::Tensor<3, host, double> values({1, 1, 3});
    values.data()[0] = 1.0;
    values.data()[1] = 2.0;
    values.data()[2] = 3.0;
    boba::BCOOTensor<4, host, double> summed({1, 2, 2, 1}, indices, values);
    pass_or_fail_bool(check, summed.nonzero_blocks() == 2);
    const auto summed_dense = boba::to_dense(summed);
    auto summed_view = summed_dense.const_view();
    pass_or_fail(check, std::abs(summed_view({0, 0, 1, 0}) - 2.0), 1e-15);
    pass_or_fail(check, std::abs(summed_view({0, 1, 1, 0}) - 4.0), 1e-15);
  }

  // Swapping the interior modes transposes each TT-matrix core.
  {
    auto transposed = bcoo;
    boba::permute<space, 4, double>(transposed, boba::Array<boba::index_t, 4>{0, 2, 1, 3});
    auto expected = dense;
    boba::permute<host, 4, double>(expected, boba::Array<boba::index_t, 4>{0, 2, 1, 3});
    pass_or_fail(check, boba::norm_difference_frobenius(boba::Tensor<4, host, double>(boba::to_dense(transposed)), expected), 1e-15);
  }
}

void test_kernels(bool& check)
{
  if constexpr (space != host)
  {
    return;
  }
  else
  {
    const boba::index_t ra1 = 4;
    const boba::index_t ra2 = 3;
    const boba::index_t rx = 3;
    const boba::index_t ry = 2;
    const auto dense = make_block_sparse_core(ra1, 5, 2, ra2, 7);
    const auto bcoo = boba::from_dense_bcoo(dense);
    const boba::index_t n = dense.sizes(1);
    const auto x = random_tensor<3>({rx, n, rx}, 1);
    const auto y = random_tensor<3>({ry, n, ry}, 2);

    // Environment updates, both sweep directions, against the dense kernel.
    const auto phi_left = random_tensor<3>({rx, ry, ra1}, 3);
    const auto phi_right = random_tensor<3>({ry, ra2, rx}, 4);
    for (bool left_to_right : {true, false})
    {
      const auto& phi = left_to_right ? phi_left : phi_right;
      const auto dense_phi = boba::compute_next_Phi<4>(phi, x, dense, y, left_to_right, 1.0);
      const auto bcoo_phi = boba::compute_next_Phi<4>(phi, x, bcoo, y, left_to_right, 1.0);
      pass_or_fail(check, boba::norm_difference_frobenius(bcoo_phi.first, dense_phi.first) / boba::norm_frobenius(dense_phi.first),
                   1e-13);
    }

    // Local operator application and its dense projected matrix.
    const auto Phi1 = random_tensor<3>({ry, rx, ra1}, 5);
    const auto Phi2 = random_tensor<3>({rx, ra2, ry}, 6);
    const auto x_site = random_tensor<3>({rx, n, rx}, 8);
    const auto dense_apply = boba::bfun3(Phi1, dense, Phi2, x_site);
    const auto bcoo_apply = boba::bfun3(Phi1, bcoo, Phi2, x_site);
    pass_or_fail(check, boba::norm_difference_frobenius(bcoo_apply, dense_apply) / boba::norm_frobenius(dense_apply), 1e-13);
    const auto dense_matrix = boba::bfun3_matrix(Phi1, dense, Phi2);
    const auto bcoo_matrix = boba::bfun3_matrix(Phi1, bcoo, Phi2);
    pass_or_fail(check, boba::norm_difference_frobenius(bcoo_matrix, dense_matrix) / boba::norm_frobenius(dense_matrix), 1e-13);

    // One core of a TT-matrix/TT-vector product, against the COO contraction.
    const auto coo = boba::convert_to_SparseTensor(bcoo);
    auto expected = boba::tensor_contraction<1>(
      {"ml", "row", "col", "mr"}, coo, {"l", "col", "r"}, x, {"ml", "l", "row", "mr", "r"});
    const auto product = boba::matrix_core_vector_core_product(bcoo, x);
    const auto expected_core = boba::reshape<3>(expected, product.sizes());
    pass_or_fail(check, boba::norm_difference_frobenius(product, expected_core) / boba::norm_frobenius(expected_core), 1e-13);
  }
}

} // namespace

int main()
{
  boba::init();

  bool check = true;
  test_format(check);
  test_kernels(check);

  const int result = final_check(check);
  boba::finalize();
  return result;
}
