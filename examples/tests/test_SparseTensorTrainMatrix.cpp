// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = boba::host_space;

// Build one matrix factor for each tensor-train mode.  The diagonal and two
// off-diagonal entries give the dense cores a reproducible, sparse pattern.
template <size_t dimension>
boba::Array<boba::Matrix<space, double>, dimension> make_matrix_term(double scale)
{
  boba::Array<boba::Matrix<space, double>, dimension> matrices;
  for (size_t d = 0; d < dimension; ++d)
  {
    // Each mode uses a 3-by-3 matrix, with values that vary by mode.
    boba::Matrix<space, double> matrix({3, 3});
    matrix.fill_with_zeros();
    matrix({0, 0}) = scale * (1.0 + d);
    matrix({1, 1}) = scale * (2.0 + d);
    matrix({2, 2}) = scale * (3.0 + d);
    matrix({0, 1}) = scale * 0.25 * (1.0 + d);
    matrix({1, 2}) = -scale * 0.5 * (1.0 + d);
    matrices[d] = matrix;
  }
  return matrices;
}

template <size_t dimension>
void test_sparse_tensor_train_matrix(bool& check)
{
  // Use three modes with matching row and column extents for both the matrix
  // and the tensor-train vector used in the matvec check.
  const auto sizes = boba::filled_array<dimension>(3_z);

  // Sum two dense tensor-train matrix terms so the test exercises conversion
  // of a nontrivial dense TTM rather than a single rank-one term.
  auto dense_ttm = boba::make_ttm_from_matrices(make_matrix_term<dimension>(1.0));
  dense_ttm += boba::make_ttm_from_matrices(make_matrix_term<dimension>(-0.3));

  // Retain every nonzero dense-core entry when constructing the sparse TTM.
  auto sparse_ttm = boba::from_dense(dense_ttm);

  // Conversion must preserve the logical matrix dimensions while storing
  // fewer entries than the fully dense cores.
  pass_or_fail_bool(check, sparse_ttm.rows() == dense_ttm.rows());
  pass_or_fail_bool(check, sparse_ttm.cols() == dense_ttm.cols());
  pass_or_fail_bool(check, sparse_ttm.get_number_elements() > 0);
  pass_or_fail_bool(check, sparse_ttm.get_number_elements() < dense_ttm.get_number_elements());

  // A constant input makes the expected dense and sparse matvecs easy to
  // compare without introducing a separate input-construction dependency.
  boba::TensorTrain<dimension, space, double> input(sizes);
  input.fill_with(1.0);

  // Applying equivalent dense and sparse TTMs to the same TT input must
  // produce identical decompressed tensors.
  auto dense_result = dense_ttm.TensorTrainMatrix_vector_multiply(input);
  auto sparse_result = sparse_ttm.TensorTrainMatrix_vector_multiply(input);
  pass_or_fail(
    check,
    boba::norm_difference_frobenius(dense_result.decompress(), sparse_result.decompress()),
    1.0e-14);

  // Converting back to a dense TTM must restore the original core values,
  // including explicit zeros that were omitted from sparse storage.
  auto reconstructed_ttm = boba::to_dense(sparse_ttm);
  pass_or_fail(
    check,
    boba::norm_difference_frobenius(dense_ttm.decompress(), reconstructed_ttm.decompress()),
    1.0e-14);
}

int main()
{
  // Initialize the BoBa runtime before constructing tensors.
  boba::init();

  bool check = true;
  test_sparse_tensor_train_matrix<3>(check);

  // Report all accumulated checks before releasing runtime resources.
  const int result = final_check(check);
  boba::finalize();
  return result;
}
