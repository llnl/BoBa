// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::default_execution_space;

template <typename data_t>
void test_htucker(bool& check)
{
  constexpr size_t dimension = 4;
  constexpr double tolerance = 1.0e-9;

  auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(dimension));
  auto sizes = boba::filled_array<dimension>(12_z);

  boba::HierarchicalTucker<dimension, space, data_t> ht(sizes, dim_tree);
  ht.fill_with_random();

  auto dense = ht.decompress();
  auto dense_norm = boba::norm_frobenius(dense);

  const auto& transfer_tensors = ht.get_transfer_tensors();
  const auto& basis_matrices = ht.get_basis_matrices();
  boba::HierarchicalTucker<dimension, space, data_t> reconstructed(
    transfer_tensors, basis_matrices, dim_tree);

  pass_or_fail_bool(check, reconstructed.get_sizes() == sizes);
  pass_or_fail_bool(check, reconstructed.get_dim_tree() == dim_tree);
  pass_or_fail(check,
               boba::norm_difference_frobenius(reconstructed.decompress(), dense) / dense_norm,
               tolerance);

  auto gramians = ht.compute_reduced_gramians();
  pass_or_fail_bool(check, ht.get_is_orthog());
  pass_or_fail(check,
               boba::norm_difference_frobenius(ht.decompress(), dense) / dense_norm,
               tolerance);

  const auto& is_leaf = dim_tree.get_is_leaf();
  for (size_t node = 1; node < ht.get_num_nodes(); ++node)
  {
    if (!is_leaf[node])
    {
      continue;
    }

    pass_or_fail_bool(check, gramians[node].has_value());

    const auto& gramian = gramians[node].value();
    auto gramian_norm = boba::norm_frobenius(gramian);
    pass_or_fail(check,
                 boba::norm_difference_frobenius(gramian, gramian.conjugate_transpose()) / gramian_norm,
                 tolerance);

    auto unfolding = boba::unfold(dense, dim_tree.get_dims_of_node(node));
    auto full_gramian = unfolding * unfolding.conjugate_transpose();

    const auto& basis = ht.get_basis_matrix(node);
    auto reconstructed_gramian = basis * gramian * basis.conjugate_transpose();
    pass_or_fail(check,
                 boba::norm_difference_frobenius(reconstructed_gramian, full_gramian) /
                   boba::norm_frobenius(full_gramian),
                 tolerance);
  }

  ht.round();
  pass_or_fail(check,
               boba::norm_difference_frobenius(ht.decompress(), dense) / dense_norm,
               tolerance);

  boba::HierarchicalTucker<dimension, space, data_t> compressed(sizes, dim_tree);
  compressed.compress(dense);
  pass_or_fail(check,
               boba::norm_difference_frobenius(compressed.decompress(), dense) / dense_norm,
               tolerance);
}

int main()
{
  boba::splash();
  boba::init();

  bool check = true;

  test_htucker<double>(check);
  test_htucker<boba::complex<double>>(check);

  boba::finalize();
  return final_check(check);
}
