// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_EIGEN_TENSOR
#include "unsupported/Eigen/CXX11/Tensor"
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_EIGEN_TENSOR

/**
 * @brief Computes an elementwise inner product with Eigen array maps.
 * @tparam A_view_t Left operand view type.
 * @tparam B_view_t Right operand view type.
 * @param tensor_A Left operand tensor view.
 * @param tensor_B Right operand tensor view.
 * @return Sum of the elementwise product.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto eigen_inner_product(
  A_view_t const& tensor_A,
  B_view_t const& tensor_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  using data_t = std::remove_const_t<typename A_view_t::data_t>;
  using eigen_vector_t = Eigen::Array<data_t, Eigen::Dynamic, 1>;

  Eigen::Map<const eigen_vector_t> tensor_A_map(
    tensor_A.const_data(),
    static_cast<Eigen::Index>(tensor_A.size()));
  Eigen::Map<const eigen_vector_t> tensor_B_map(
    tensor_B.const_data(),
    static_cast<Eigen::Index>(tensor_B.size()));

  return tensor_A_map.cwiseProduct(tensor_B_map).sum();
}

/**
 * @brief Wraps a full tensor view in an Eigen tensor map.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param view Tensor view to map.
 * @return Eigen tensor map over the view storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
Eigen::TensorMap<Eigen::Tensor<data_t, dimension>> get_eigen_map(const TensorView<DefaultAccessor<data_t>, dimension>& view)
{
  std::array<index_t, dimension> std_sizes = make_std_array<index_t>(view.sizes());
  return {view.data(), std_sizes};
}

/**
 * @brief Wraps a const full tensor view in an Eigen tensor map.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param view Tensor view to map.
 * @return Const Eigen tensor map over the view storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
Eigen::TensorMap<const Eigen::Tensor<data_t, dimension>> get_eigen_const_map(const TensorView<DefaultAccessor<data_t const>, dimension>& view)
{
  std::array<index_t, dimension> std_sizes = make_std_array<index_t>(view.sizes());
  return {view.const_data(), std_sizes};
}

/**
 * @brief Wraps a mutable subtensor view in an Eigen slice expression.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param view Subtensor view to map.
 * @return Eigen slice expression covering the subtensor.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
auto get_eigen_map(const SubtensorView<DefaultAccessor<data_t>, dimension>& view)
{
  std::array<index_t, dimension> full_sizes = make_std_array<index_t>(view.full_sizes());
  std::array<index_t, dimension> lower_bound = make_std_array<index_t>(view.lower_bound());
  std::array<index_t, dimension> subtensor_sizes = make_std_array<index_t>(view.sizes());
  Eigen::TensorMap<Eigen::Tensor<data_t, dimension>> full_map(view.data(), full_sizes);
  return full_map.slice(lower_bound, subtensor_sizes);
}

/**
 * @brief Wraps a const subtensor view in an Eigen slice expression.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param view Subtensor view to map.
 * @return Const Eigen slice expression covering the subtensor.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
auto get_eigen_const_map(const SubtensorView<DefaultAccessor<data_t const>, dimension>& view)
{
  std::array<index_t, dimension> full_sizes = make_std_array<index_t>(view.full_sizes());
  std::array<index_t, dimension> lower_bound = make_std_array<index_t>(view.lower_bound());
  std::array<index_t, dimension> subtensor_sizes = make_std_array<index_t>(view.sizes());
  Eigen::TensorMap<const Eigen::Tensor<data_t, dimension>> full_map(view.const_data(), full_sizes);
  return full_map.slice(lower_bound, subtensor_sizes);
}

//
// Permute
//
/**
 * @brief Applies an Eigen tensor permutation.
 * @tparam dimension Tensor rank.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param old_tensor Input tensor view.
 * @param tensor_permutation Output tensor view.
 * @param permutations Axis permutation order.
 */
template <size_t dimension, typename data_t>
void eigen_tensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  BOBA_CALI_MARK
  // https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html

  auto old_map = get_eigen_const_map(old_tensor);
  auto new_map = get_eigen_map(tensor_permutation);

  std::array<size_t, dimension> shuffle;
  for (size_t d = 0; d < dimension; d++)
  {
    shuffle[d] = permutations[d];
  }

  new_map = old_map.shuffle(shuffle);
}

//
// Contract
//
/**
 * @brief Contracts one or more index pairs between two full tensor views.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void eigen_tensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK
  // https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html
  auto tensor_A_map = get_eigen_const_map(tensor_A);
  auto tensor_B_map = get_eigen_const_map(tensor_B);
  auto tensor_C_map = get_eigen_map(tensor_C);

  Eigen::array<Eigen::IndexPair<size_t>, contractions> contraction_dims;
  for (size_t c = 0; c < contractions; c++)
  {
    contraction_dims[c] = Eigen::IndexPair<size_t>(contraction_dimensions_A[c], contraction_dimensions_B[c]);
  }

  tensor_C_map = tensor_A_map.contract(tensor_B_map, contraction_dims);
  checkpoint();
}

//
// Contract
//
/**
 * @brief Contracts one or more index pairs between two subtensor views.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void eigen_tensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  SubtensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK

  // https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html
  auto tensor_A_map = get_eigen_const_map(tensor_A);
  auto tensor_B_map = get_eigen_const_map(tensor_B);
  auto tensor_C_map = get_eigen_map(tensor_C);

  Eigen::array<Eigen::IndexPair<size_t>, contractions> contraction_dims;
  for (size_t c = 0; c < contractions; c++)
  {
    contraction_dims[c] = Eigen::IndexPair<size_t>(contraction_dimensions_A[c], contraction_dimensions_B[c]);
  }

  tensor_C_map = tensor_A_map.contract(tensor_B_map, contraction_dims);
  checkpoint();
}

//
// Reduce
//
/**
 * @brief Reduces axes of a full tensor view by summation.
 */
template <size_t reductions, size_t dimension_A, size_t dimension_C, typename data_t>
void eigen_tensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  BOBA_CALI_MARK
  // https://eigen.tuxfamily.org/dox-devel/unsupported/eigen_tensors.html
  auto tensor_A_map = get_eigen_const_map(tensor_A);
  auto tensor_C_map = get_eigen_map(tensor_C);

  Eigen::array<size_t, reductions> contraction_dims;
  for (size_t i = 0; i < reductions; i++)
  {
    contraction_dims[i] = contraction_dimensions[i];
  }

  tensor_C_map = tensor_A_map.sum(contraction_dims);
}

#else

/**
 * @brief Reports that Eigen tensor inner-product support is unavailable.
 */
template <typename A_view_t, typename B_view_t>
[[nodiscard]]
auto eigen_inner_product(
  A_view_t const& tensor_A,
  B_view_t const& tensor_B) -> std::remove_const_t<typename A_view_t::data_t>
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  using data_t = std::remove_const_t<typename A_view_t::data_t>;
  boba_error("eigen_inner_product requires BOBA_EIGEN_TENSOR.");
  return data_t{};
}

/**
 * @brief Reports that Eigen tensor permutation support is unavailable.
 */
template <size_t dimension, typename data_t>
void eigen_tensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  ::boba::detail::ignore(old_tensor);
  ::boba::detail::ignore(tensor_permutation);
  ::boba::detail::ignore(permutations);
  boba_error("eigen_tensor_permute requires BOBA_EIGEN_TENSOR.");
}

/**
 * @brief Reports that Eigen tensor contraction support is unavailable.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void eigen_tensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions_A);
  ::boba::detail::ignore(contraction_dimensions_B);
  boba_error("eigen_tensor_contract requires BOBA_EIGEN_TENSOR.");
}

/**
 * @brief Reports that Eigen tensor contraction support is unavailable.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void eigen_tensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  SubtensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions_A);
  ::boba::detail::ignore(contraction_dimensions_B);
  boba_error("eigen_tensor_contract requires BOBA_EIGEN_TENSOR.");
}

/**
 * @brief Reports that Eigen tensor reduction support is unavailable.
 */
template <size_t reductions, size_t dimension_A, size_t dimension_C, typename data_t>
void eigen_tensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions);
  boba_error("eigen_tensor_reduce requires BOBA_EIGEN_TENSOR.");
}

#endif

} // namespace detail
} // namespace boba
