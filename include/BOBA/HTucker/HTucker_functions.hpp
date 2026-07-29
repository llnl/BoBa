// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <span>

namespace boba
{
/**
 * @brief Checks for NaN entries in all tensor components.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void nan_check(HierarchicalTucker<dimension, space, data_t> const& x)
{
  const auto& is_leaf = x.dim_tree.get_is_leaf();

  for (size_t node = 0; node < x.num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      boba_assert(x.basis_matrices[node].has_value(), "Basis matrix is uninitialized.");
      ::boba::nan_check(*x.basis_matrices[node]);
    }
    else
    {
      boba_assert(x.transfer_tensors[node].has_value(), "Transfer tensor is uninitialized.");
      ::boba::nan_check(*x.transfer_tensors[node]);
    }
  }
}

// -------------------------------------------------------------------------
// Product operations involving HierarchicalTucker objects
// -------------------------------------------------------------------------

/**
 * @brief Tensor (Kronecker) product of two HierarchicalTucker tensors.
 *
 * @param x Left-hand side HierarchicalTucker tensor.
 * @param y Right-hand side HierarchicalTucker tensor.
 * @return A new HierarchicalTucker tensor representing the tensor product.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
tensor_product(const HierarchicalTucker<dimension, space, data_t>& x,
               const HierarchicalTucker<dimension, space, data_t>& y)
{
  using HT_type = HierarchicalTucker<dimension, space, data_t>;
  using tensor3_type = ::boba::Tensor<3, space, data_t>;
  using matrix_type = ::boba::Matrix<space, data_t>;

  boba_always_assert(x.get_dim_tree() == y.get_dim_tree(), "tensor_product: Incompatible dimension trees.");

  constexpr auto num_nodes = HT_type::num_nodes;
  const auto& dim_tree = x.get_dim_tree();
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Initialize containers for the output transfer tensors and basis matrices
  ::boba::Array<std::optional<tensor3_type>, num_nodes> output_transfer_tensors;
  ::boba::Array<std::optional<matrix_type>, num_nodes> output_basis_matrices;

  // Access the transfer tensors and basis matrices of the input HTuckers
  const auto& transfer_tensors_x = x.get_transfer_tensors();
  const auto& basis_matrices_x = x.get_basis_matrices();

  const auto& transfer_tensors_y = y.get_transfer_tensors();
  const auto& basis_matrices_y = y.get_basis_matrices();

  for (size_t node = 0; node < num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      boba_assert(basis_matrices_x[node].has_value(),
                  "Missing basis matrix for x at node " + std::to_string(node) + " in tensor_product.");
      boba_assert(basis_matrices_y[node].has_value(),
                  "Missing basis matrix for y at node " + std::to_string(node) + " in tensor_product.");

      // Get the basis matrices of the current node for both x and y
      auto basis_matrix_x = basis_matrices_x[node].value();
      auto basis_matrix_y = basis_matrices_y[node].value();

      // Compute the Kronecker product node-wise basis for x and y
      auto output_basis_matrix = ::boba::tensor_product(basis_matrix_x, basis_matrix_y);
      output_basis_matrices[node] = std::move(output_basis_matrix);
    }
    else
    {
      boba_assert(transfer_tensors_x[node].has_value(),
                  "Missing transfer tensor for x at node " + std::to_string(node) + " in tensor_product.");
      boba_assert(transfer_tensors_y[node].has_value(),
                  "Missing transfer tensor for y at node " + std::to_string(node) + " in tensor_product.");

      // Get the transfer tensors of the current node for both x and y
      auto transfer_tensor_x = transfer_tensors_x[node].value();
      auto transfer_tensor_y = transfer_tensors_y[node].value();

      // Compute the Kronecker product of the two transfer tensors
      auto output_transfer_tensor = ::boba::tensor_product(transfer_tensor_x, transfer_tensor_y);
      output_transfer_tensors[node] = std::move(output_transfer_tensor);
    }
  }

  // Construct the output HierarchicalTucker object using the transfer tensors and basis matrices
  HierarchicalTucker<dimension, space, data_t> output(output_transfer_tensors, output_basis_matrices, dim_tree);

  return output;
}

/**
 * @brief An elementwise (Hadamard) product of two HierarchicalTucker tensors.
 *
 * @param x Left-hand side HierarchicalTucker tensor.
 * @param y Right-hand side HierarchicalTucker tensor.
 * @return A new HierarchicalTucker tensor representing the elementwise product.
 *
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
elementwise_product(const HierarchicalTucker<dimension, space, data_t>& x,
                    const HierarchicalTucker<dimension, space, data_t>& y)
{
  return x * y;
}

/**
 * @brief An approximate elementwise (Hadamard) product of two HierarchicalTucker tensors.
 *
 * @param x Left-hand side HierarchicalTucker tensor.
 * @param y Right-hand side HierarchicalTucker tensor.
 * @param relative_tolerance Relative tolerance for truncation.
 * @param absolute_tolerance Absolute tolerance for truncation.
 * @return A new HierarchicalTucker tensor representing an approximate elementwise product.
 *
 * @note For the exact Hadamard product use the overload for the multiplication operator *.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
elementwise_product(HierarchicalTucker<dimension, space, data_t>& x,
                    HierarchicalTucker<dimension, space, data_t>& y,
                    data_t relative_tolerance,
                    data_t absolute_tolerance)
{
  using HT_type = HierarchicalTucker<dimension, space, data_t>;
  using tensor3_type = ::boba::Tensor<3, space, data_t>;
  using matrix_type = ::boba::Matrix<space, data_t>;
  using real_data_t = ::boba::real_type_t<data_t>;

  boba_always_assert(x.get_dim_tree() == y.get_dim_tree(), "elementwise_product: Incompatible dimension trees.");
  boba_always_assert(x.get_sizes() == y.get_sizes(), "elementwise_product: Incompatible mode sizes.");

  // Compute reduced Gramians for the input HTuckers
  auto gramians_x = x.compute_reduced_gramians();
  auto gramians_y = y.compute_reduced_gramians();

  constexpr auto num_nodes = HT_type::num_nodes;
  const auto& dim_tree = x.get_dim_tree();
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Build the SVD object with the tolerances specified in the input arguments
  SVD<space, data_t> svd;
  svd.tolerance_relative = ::boba::pow(relative_tolerance, 2.0);
  svd.tolerance_absolute = ::boba::pow(absolute_tolerance, 2.0);

  // Truncated bases for the reduced Gramians of each node
  ::boba::Array<std::optional<matrix_type>, num_nodes> truncated_gramian_bases_x;
  ::boba::Array<std::optional<matrix_type>, num_nodes> truncated_gramian_bases_y;

  // Loop over all nodes except the root, for which the Gramian is 1 (for both x and y)
  // This loop builds the node-wise truncated bases for the Gramians of x and y.
  // Truncation is performed using the singular values of the Kronecker product of the Gramians
  for (size_t node = 1; node < num_nodes; ++node)
  {
    boba_assert(gramians_x[node].has_value(),
                "Missing Gramian for x at node " + std::to_string(node) + " in elementwise_product.");

    boba_assert(gramians_y[node].has_value(),
                "Missing Gramian for y at node " + std::to_string(node) + " in elementwise_product.");

    // Calculate eigendecomposition of the reduced Gramians G = U diag(Lambda) U^* (for both x and y)
    //
    // Note: The singular values are the square roots of the eigenvalues Lambda for each of the Gramians.
    // For this reason, we truncate the SVDs using the squares of the tolerances.
    //
    // Note: We use the SVD for now, but this can later be replaced with an eigendecomposition due to symmetry
    svd(gramians_x[node].value());
    const auto gramian_basis_x = svd.U;
    auto gramian_singular_values_x = svd.S;

    svd(gramians_y[node].value());
    const auto gramian_basis_y = svd.U;
    auto gramian_singular_values_y = svd.S;

    auto gramian_singular_values_x_view = gramian_singular_values_x.view();
    auto gramian_singular_values_y_view = gramian_singular_values_y.view();

    // Compute the square roots of the eigenvalues (in-place) to get the singular values
    ::boba::loop<space, 1>(gramian_singular_values_x.size(),
                           [=] __boba_host_device__(index_t i)
    {
      gramian_singular_values_x_view({i}) = ::boba::sqrt(gramian_singular_values_x_view({i}));
    });

    // Compute the square roots of the eigenvalues (in-place) to get the singular values
    ::boba::loop<space, 1>(gramian_singular_values_y.size(),
                           [=] __boba_host_device__(index_t i)
    {
      gramian_singular_values_y_view({i}) = ::boba::sqrt(gramian_singular_values_y_view({i}));
    });

    // Take the Kronecker product of the singular values and copy to the host for processing
    // Note that we make x vary the fastest in the Kronecker product
    auto kron_prod_singular_values = ::boba::tensor_product(gramian_singular_values_y, gramian_singular_values_x);

    // Multiindexer to convert the sorted flat indices into multi-indices for the Kronecker product
    // This respects the convention that x varies the fastest
    ::boba::Multiindexer<2> kron_mid({gramian_singular_values_x.size(), gramian_singular_values_y.size()});

    // Copy the singular values of the Kronecker product to the host for processing
    ::boba::Vector<host_space, real_data_t> kron_prod_singular_values_host = kron_prod_singular_values;

    // Sort the singular values of the Kronecker product in descending order on the host
    std::vector<index_t> order(kron_prod_singular_values_host.size());
    std::iota(order.begin(), order.end(), index_t(0));

    std::sort(order.begin(), order.end(), [&](index_t i, index_t j)
    {
      return kron_prod_singular_values_host({i}) > kron_prod_singular_values_host({j});
    });

    // Determine the truncation rank k based on the tolerances and truncate the Gramian bases for x and y
    index_t last_significant_value_index = 0;
    real_data_t max_sigma = kron_prod_singular_values_host({order[0]});

    for (size_t idx = 0; idx < kron_prod_singular_values_host.size(); ++idx)
    {
      index_t sorted_index = order[idx];
      real_data_t sigma = kron_prod_singular_values_host({sorted_index});
      bool relative_check = sigma > relative_tolerance * max_sigma;
      bool absolute_check = sigma > absolute_tolerance;
      if (relative_check && absolute_check)
      {
        last_significant_value_index = idx;
      }
      else
      {
        break;
      }
    }

    index_t significant_singular_values = last_significant_value_index + 1;

    // Storage for the column indices of the truncated bases for x and y in the Kronecker product
    ::boba::Vector<host_space, index_t> column_indices_x_host({significant_singular_values});
    ::boba::Vector<host_space, index_t> column_indices_y_host({significant_singular_values});

    for (size_t idx = 0; idx < significant_singular_values; ++idx)
    {
      // Convert the sorted index into multi-index and extract ix, iy
      auto sorted_idx = order[idx];
      auto mid = kron_mid.multiindex(sorted_idx);
      auto [ix, iy] = mid;
      column_indices_x_host({idx}) = ix;
      column_indices_y_host({idx}) = iy;
    }

    // Copy the column indices for x and y to the device
    ::boba::Vector<space, index_t> column_indices_x = column_indices_x_host;
    ::boba::Vector<space, index_t> column_indices_y = column_indices_y_host;

    // Form the truncated bases for x and y at the current node by copying the appropriate
    // columns from the original Gramian bases for x and y
    truncated_gramian_bases_x[node] = std::move(gramian_basis_x.extract_columns(column_indices_x));
    truncated_gramian_bases_y[node] = std::move(gramian_basis_y.extract_columns(column_indices_y));
  }

  // Initialize containers for the output transfer tensors and basis matrices
  ::boba::Array<std::optional<tensor3_type>, num_nodes> output_transfer_tensors;
  ::boba::Array<std::optional<matrix_type>, num_nodes> output_basis_matrices;

  for (size_t node = 1; node < num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      // Get the basis matrices of the current node for both x and y, and their truncated Gramians bases
      auto basis_matrix_x = x.get_basis_matrix(node);
      auto basis_matrix_y = y.get_basis_matrix(node);

      auto truncated_gramian_basis_x = truncated_gramian_bases_x[node].value();
      auto truncated_gramian_basis_y = truncated_gramian_bases_y[node].value();

      // Truncate the bases using their respective Gramians
      auto truncated_basis_matrix_x = basis_matrix_x * truncated_gramian_basis_x;
      auto truncated_basis_matrix_y = basis_matrix_y * truncated_gramian_basis_y;

      // Form the truncated Kronecker product of the bases (implicitly) using the element-wise product of the two matrices
      auto output_truncated_basis_matrix = ::boba::elementwise_product(truncated_basis_matrix_x, truncated_basis_matrix_y);
      output_basis_matrices[node] = std::move(output_truncated_basis_matrix);
    }
    else
    {
      const auto& children = dim_tree.get_children_of_node(node);
      auto transfer_tensor_x = x.get_transfer_tensor(node);
      auto transfer_tensor_y = y.get_transfer_tensor(node);

      auto tmp_x = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(truncated_gramian_bases_x[children[0]].value()), {"l", "j", "k"}, transfer_tensor_x, {"i", "j", "k"});

      auto tmp2_x = ::boba::tensor_contraction_single_index({"l", "j"}, ::boba::get_conj(truncated_gramian_bases_x[children[1]].value()), {"i", "l", "k"}, tmp_x, {"i", "j", "k"});

      auto truncated_transfer_tensor_x = ::boba::tensor_contraction_single_index({"i", "j", "l"}, tmp2_x, {"l", "k"}, truncated_gramian_bases_x[node].value(), {"i", "j", "k"});

      auto tmp_y = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(truncated_gramian_bases_y[children[0]].value()), {"l", "j", "k"}, transfer_tensor_y, {"i", "j", "k"});

      auto tmp2_y = ::boba::tensor_contraction_single_index({"l", "j"}, ::boba::get_conj(truncated_gramian_bases_y[children[1]].value()), {"i", "l", "k"}, tmp_y, {"i", "j", "k"});

      auto truncated_transfer_tensor_y = ::boba::tensor_contraction_single_index({"i", "j", "l"}, tmp2_y, {"l", "k"}, truncated_gramian_bases_y[node].value(), {"i", "j", "k"});

      // Form the truncated Kronecker product of the transfer tensors (implicitly) using the element-wise product of the two tensors
      auto output_transfer_tensor = ::boba::elementwise_product(truncated_transfer_tensor_x, truncated_transfer_tensor_y);
      output_transfer_tensors[node] = std::move(output_transfer_tensor);
    }
  }

  // The root node transfer tensor is treated as a special case
  const auto& children = dim_tree.get_children_of_node(0);
  auto transfer_tensor_x = x.get_transfer_tensor(0);
  auto transfer_tensor_y = y.get_transfer_tensor(0);

  // Truncate the root node transfer tensors for x and y using the previously computed truncated Gramian bases for the children of the root node
  auto tmp_x = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(truncated_gramian_bases_x[children[0]].value()), {"l", "j", "k"}, transfer_tensor_x, {"i", "j", "k"});

  auto truncated_transfer_tensor_x = ::boba::tensor_contraction_single_index({"l", "j"}, ::boba::get_conj(truncated_gramian_bases_x[children[1]].value()), {"i", "l", "k"}, tmp_x, {"i", "j", "k"});

  auto tmp_y = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(truncated_gramian_bases_y[children[0]].value()), {"l", "j", "k"}, transfer_tensor_y, {"i", "j", "k"});

  auto truncated_transfer_tensor_y = ::boba::tensor_contraction_single_index({"l", "j"}, ::boba::get_conj(truncated_gramian_bases_y[children[1]].value()), {"i", "l", "k"}, tmp_y, {"i", "j", "k"});

  // Form the truncated Kronecker product of the transfer tensors (implicitly) using the element-wise product of the two tensors
  auto output_transfer_tensor = ::boba::elementwise_product(truncated_transfer_tensor_x, truncated_transfer_tensor_y);
  output_transfer_tensors[0] = std::move(output_transfer_tensor);

  // Construct the output HierarchicalTucker object using the transfer tensors and basis matrices
  HierarchicalTucker<dimension, space, data_t> output(output_transfer_tensors, output_basis_matrices, dim_tree);

  return output;
}

// -------------------------------------------------------------------------
// sum and round
// -------------------------------------------------------------------------

/**
 * @brief Computes the sum of a sequence of HierarchicalTucker tensors,
 * rounding after each addition.
 *
 * Each tensor in @p sequence is added to an accumulator, followed by a rounding
 * operation to maintain a low-rank representation.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space (e.g., host or device).
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::span of HierarchicalTucker tensors to be summed and rounded.
 * @return A new HierarchicalTucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
sum_and_round(std::span<const HierarchicalTucker<dimension, space, data_t>> sequence)
{
  using ht_t = HierarchicalTucker<dimension, space, data_t>;

  boba_always_assert(!sequence.empty(), "Cannot perform sum_and_round on an empty sequence.");

  // Use the first tensor to infer shape and tree structure
  const auto& first = sequence.front();
  ht_t output(first.get_sizes(), first.get_dim_tree());

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
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence std::vector of HierarchicalTucker tensors to be summed and rounded.
 * @return A new HierarchicalTucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
sum_and_round(const std::vector<HierarchicalTucker<dimension, space, data_t>>& sequence)
{
  using ht_t = HierarchicalTucker<dimension, space, data_t>;
  return sum_and_round(std::span<const ht_t>{sequence});
}

/**
 * @brief Lightweight adapter for converting std::initializer_list into
 * std::span for addition and rounding.
 *
 * @note This overload is convenient, but it may copy the input tensors into the
 * initializer-list backing array. Prefer the std::span or std::vector overloads
 * for performance-sensitive code.
 *
 * @tparam dimension  Number of tensor dimensions.
 * @tparam space      Execution space.
 * @tparam data_t  Value type for tensor entries.
 * @param sequence Initializer list of HierarchicalTucker tensors to be summed and rounded.
 * @return A new HierarchicalTucker tensor representing the accumulated and rounded result.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
HierarchicalTucker<dimension, space, data_t>
sum_and_round(const std::initializer_list<HierarchicalTucker<dimension, space, data_t>> sequence)
{
  using ht_t = HierarchicalTucker<dimension, space, data_t>;
  return sum_and_round(std::span<const ht_t>{sequence.begin(), sequence.size()});
}

// -------------------------------------------------------------------------
// Inner product and Frobenius norm
// -------------------------------------------------------------------------

/**
 * @brief Computes the scalar (inner) product of two HierarchicalTucker tensors.
 *
 * Evaluates \f$ \langle x, y \rangle \f$ recursively by contracting
 * basis matrices at the leaves and propagating partial contractions up the
 * dimension tree.
 *
 * @param x Left-hand side HierarchicalTucker tensor.
 * @param y Right-hand side HierarchicalTucker tensor.
 * @return Scalar inner product value.
 *
 * @note Requires both HierarchicalTucker tensors to have identical dimension trees and mode sizes.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
data_t inner_product(
  HierarchicalTucker<dimension, space, data_t> const& x,
  HierarchicalTucker<dimension, space, data_t> const& y)
{
  boba_always_assert(x.get_dim_tree() == y.get_dim_tree(), "HierarchicalTucker inner_product: dimension trees must match.");
  boba_always_assert(x.get_sizes() == y.get_sizes(), "HierarchicalTucker inner_product: mode sizes must match.");

  const auto& dim_tree = x.get_dim_tree();

  const auto& x_transfer_tensors = x.get_transfer_tensors();
  const auto& x_basis_matrices = x.get_basis_matrices();

  const auto& y_transfer_tensors = y.get_transfer_tensors();
  const auto& y_basis_matrices = y.get_basis_matrices();

  const auto& post_order_nodes = dim_tree.get_post_order_nodes();
  const auto& is_leaf = dim_tree.get_is_leaf();

  using M_type = typename HierarchicalTucker<dimension, space, data_t>::M_type;
  constexpr size_t num_nodes = HierarchicalTucker<dimension, space, data_t>::num_nodes;

  // Stores intermediate partial inner products at each node (root holds the final answer)
  ::boba::Array<M_type, num_nodes> partial_inner_products;

  for (size_t node : post_order_nodes)
  {
    if (is_leaf[node])
    {
      boba_assert(x_basis_matrices[node].has_value() && y_basis_matrices[node].has_value(),
                  "Missing basis matrices at leaf node " + std::to_string(node));

      const auto& x_basis = x_basis_matrices[node].value();
      const auto& y_basis = y_basis_matrices[node].value();

      auto local_partial = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(x_basis), {"l", "j"}, y_basis, {"i", "j"});
      partial_inner_products[node].emplace(std::move(local_partial));
    }
    else
    {
      // Internal nodes recursively merge partials from the children
      const size_t left_child = dim_tree.get_left_child_of_node(node);
      const size_t right_child = dim_tree.get_right_child_of_node(node);

      boba_assert(partial_inner_products[left_child].has_value(),
                  "Missing partial inner product at left child of node " + std::to_string(node));
      boba_assert(partial_inner_products[right_child].has_value(),
                  "Missing partial inner product at right child of node " + std::to_string(node));
      boba_assert(x_transfer_tensors[node].has_value(),
                  "Missing x-transfer tensor at node " + std::to_string(node));
      boba_assert(y_transfer_tensors[node].has_value(),
                  "Missing y-transfer tensor at node " + std::to_string(node));

      const auto& partial_left = partial_inner_products[left_child].value();
      const auto& partial_right = partial_inner_products[right_child].value();
      const auto& x_transfer = x_transfer_tensors[node].value();
      const auto& y_transfer = y_transfer_tensors[node].value();

      // Propagate partials from children up to the current node for y, then merge with x
      auto propagated_left = ::boba::tensor_contraction_single_index({"i", "l"}, partial_left, {"l", "j", "k"}, y_transfer, {"i", "j", "k"});
      auto y_merged = ::boba::tensor_contraction_single_index({"j", "l"}, partial_right, {"i", "l", "k"}, propagated_left, {"i", "j", "k"});
      auto local_partial = ::boba::tensor_contraction_double_index({"l", "m", "i"}, ::boba::get_conj(x_transfer), {"l", "m", "j"}, y_merged, {"i", "j"});
      partial_inner_products[node].emplace(std::move(local_partial));

      // Free child data to save memory
      partial_inner_products[left_child].reset();
      partial_inner_products[right_child].reset();
    }
  }

  // Inner product stored at the root node
  boba_assert(partial_inner_products[0].has_value(), "The inner product at the root is missing.");

  return partial_inner_products[0]->sum_reduce();
}

/**
 * @brief Computes the Frobenius norm of an HierarchicalTucker object.
 *
 * \f$ \| A \|_F = \sqrt{ \langle A, A \rangle } \f$
 *
 * @return Frobenius norm of the tensor.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
data_t norm_frobenius(HierarchicalTucker<dimension, space, data_t> const& x)
{
  data_t result = ::boba::inner_product(x, x);
  return ::boba::sqrt(result);
}

// -------------------------------------------------------------------------
// Fast Fourier Transforms
// -------------------------------------------------------------------------

/**
 * \brief
 * fft along one dimension of the decomposition, consistent with fft_along_dimension
 */

template <size_t dimension, execution_space space, typename data_t>
[[nodiscard]]
HierarchicalTucker<dimension, space, complex<data_t>> fft_along_dimension(
  const HierarchicalTucker<dimension, space, complex<data_t>>& input,
  index_t transform_dimension,
  fft_operation operation)
{
  auto output = input;
  auto extent_dimension = 0_z;

  // Get the dimension tree and the corresponding node index from the dim2idx map
  const auto& dim_tree = input.get_dim_tree();
  const auto& dim2idx = dim_tree.get_dim2idx();
  const auto node_idx = dim2idx[transform_dimension];

  // Transform the basis matrix along the row dimensions
  auto basis_matrix = input.get_basis_matrix(node_idx);
  auto transformed_basis = fft_along_dimension<2, space, data_t>(basis_matrix, extent_dimension, operation);

  // Set the basis in the output
  output.set_basis_matrix(node_idx, boba::Matrix<space, complex<data_t>>(transformed_basis));

  return output;
}

// -------------------------------------------------------------------------------------
// I/O for HierarchicalTucker objects
// -------------------------------------------------------------------------------------

/**
 * \brief Write a HierarchicalTucker decomposition to files in a way consistent with
 * Tensor::write_to_file.
 *
 * The dimension tree is written to `<prefix>_dimension_tree`. Each leaf basis
 * matrix is written to `<prefix>_basis_matrix_<node>`, and each non-leaf
 * transfer tensor is written to `<prefix>_transfer_tensor_<node>`.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_file(const HierarchicalTucker<dimension, space, data_t>& htucker,
                   std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = htucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  const auto& dim_tree = htucker.get_dim_tree();
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Write the dimension tree
  dim_tree.write_to_file(print_filename + "_dimension_tree");

  // Write the transfer tensors and basis matrices to files
  for (size_t node = 0; node < htucker.get_num_nodes(); ++node)
  {
    if (is_leaf[node])
    {
      write_to_file(htucker.get_basis_matrix(node),
                    print_filename + "_basis_matrix_" + std::to_string(node));
    }
    else
    {
      write_to_file(htucker.get_transfer_tensor(node),
                    print_filename + "_transfer_tensor_" + std::to_string(node));
    }
  }
}

/**
 * \brief Read a HierarchicalTucker decomposition from files generated by
 * write_to_file.
 *
 * The dimension tree is read from `<prefix>_dimension_tree`. Each leaf basis
 * matrix is read from `<prefix>_basis_matrix_<node>`, and each non-leaf
 * transfer tensor is read from `<prefix>_transfer_tensor_<node>`. The loaded
 * components are installed with reset_components, which updates the internal
 * sizes and ranks and checks consistency.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_file(HierarchicalTucker<dimension, space, data_t>& htucker,
                    std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = htucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  // Reconstruct the dimension tree from the file contents
  ::boba::DimensionTree dim_tree;
  dim_tree.read_from_file(print_filename + "_dimension_tree");
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Next we need to rebuild the transfer tensors and basis matrices
  constexpr size_t num_nodes = HierarchicalTucker<dimension, space, data_t>::num_nodes;
  using B_type = typename HierarchicalTucker<dimension, space, data_t>::B_type;
  using U_type = typename HierarchicalTucker<dimension, space, data_t>::U_type;

  ::boba::Array<B_type, num_nodes> transfer_tensors;
  ::boba::Array<U_type, num_nodes> basis_matrices;

  for (size_t node = 0; node < num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      basis_matrices[node].emplace();
      read_from_file(basis_matrices[node].value(),
                     print_filename + "_basis_matrix_" + std::to_string(node));
    }
    else
    {
      transfer_tensors[node].emplace();
      read_from_file(transfer_tensors[node].value(),
                     print_filename + "_transfer_tensor_" + std::to_string(node));
    }
  }

  // Bulk reset and validate the loaded components
  htucker.reset_components(std::move(transfer_tensors),
                           std::move(basis_matrices),
                           std::move(dim_tree));
}

/**
 * \brief Write a HierarchicalTucker decomposition to a MATLAB mat-file.
 *
 * Each leaf basis matrix is written to `<prefix>_basis_matrix_<node>`, and each
 * non-leaf transfer tensor is written to `<prefix>_transfer_tensor_<node>`.
 * The dimension tree is written separately to the sidecar file
 * `<prefix>_dimension_tree`.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_mat_file(const HierarchicalTucker<dimension, space, data_t>& htucker,
                       std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = htucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  const auto& dim_tree = htucker.get_dim_tree();
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Write the dimension tree as a sidecar file
  dim_tree.write_to_file(print_filename + "_dimension_tree");

  // Open the mat-file, overwriting previous contents
  ::boba::detail::MatFile mat_file(print_filename, "w");

  // Write the transfer tensors and basis matrices into the mat file
  for (size_t node = 0; node < htucker.get_num_nodes(); ++node)
  {
    if (is_leaf[node])
    {
      mat_file.write_array(print_filename + "_basis_matrix_" + std::to_string(node),
                           htucker.get_basis_matrix(node));
    }
    else
    {
      mat_file.write_array(print_filename + "_transfer_tensor_" + std::to_string(node),
                           htucker.get_transfer_tensor(node));
    }
  }
}

/**
 * \brief Read a HierarchicalTucker decomposition from a MATLAB mat-file.
 *
 * The dimension tree is read from the sidecar file `<prefix>_dimension_tree`.
 * Each leaf basis matrix is read from `<prefix>_basis_matrix_<node>`, and each
 * non-leaf transfer tensor is read from `<prefix>_transfer_tensor_<node>`.
 * The loaded components are installed with reset_components, which updates the
 * internal sizes and ranks and checks consistency.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_mat_file(HierarchicalTucker<dimension, space, data_t>& htucker,
                        std::string_view filename = "")
{
  std::string print_filename = "";

  if (filename.empty())
  {
    print_filename = htucker.name();
  }
  else
  {
    print_filename = std::string(filename);
  }

  // Reconstruct the dimension tree from the sidecar file.
  ::boba::DimensionTree dim_tree;
  dim_tree.read_from_file(print_filename + "_dimension_tree");
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Open the mat-file for reading.
  ::boba::detail::MatFile mat_file(print_filename, "r");

  // Next we need to rebuild the transfer tensors and basis matrices
  constexpr size_t num_nodes = HierarchicalTucker<dimension, space, data_t>::num_nodes;
  using B_type = typename HierarchicalTucker<dimension, space, data_t>::B_type;
  using U_type = typename HierarchicalTucker<dimension, space, data_t>::U_type;

  ::boba::Array<B_type, num_nodes> transfer_tensors;
  ::boba::Array<U_type, num_nodes> basis_matrices;

  // Read the transfer tensors and basis matrices from the mat file
  for (size_t node = 0; node < num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      basis_matrices[node].emplace();
      mat_file.read_array(print_filename + "_basis_matrix_" + std::to_string(node),
                          basis_matrices[node].value());
    }
    else
    {
      transfer_tensors[node].emplace();
      mat_file.read_array(print_filename + "_transfer_tensor_" + std::to_string(node),
                          transfer_tensors[node].value());
    }
  }

  // Bulk reset and validate the loaded components
  htucker.reset_components(std::move(transfer_tensors),
                           std::move(basis_matrices),
                           std::move(dim_tree));
}

/**
 * \brief Writes a HierarchicalTucker decomposition to an HDF5 file.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void write_to_hdf5_file(const HierarchicalTucker<dimension, space, data_t>& htucker,
                        std::string_view filename,
                        std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = htucker.name();
  }

  std::string print_filename = std::string(filename);

  // Open an HDF5 file
  ::boba::detail::HDF5File h5_file(filename, "w");

  const auto& dim_tree = htucker.get_dim_tree();
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Write the dimension tree as a sidecar file
  dim_tree.write_to_file(print_filename + "_dimension_tree");

  // Write the transfer tensors and basis matrices into the HDF5 file
  for (size_t node = 0; node < htucker.get_num_nodes(); ++node)
  {
    if (is_leaf[node])
    {
      h5_file.write_array(object_name + "_basis_matrix_" + std::to_string(node),
                          htucker.get_basis_matrix(node));
    }
    else
    {
      h5_file.write_array(object_name + "_transfer_tensor_" + std::to_string(node),
                          htucker.get_transfer_tensor(node));
    }
  }
}

/**
 * \brief Reads a HierarchicalTucker decomposition from an HDF5 file.
 *
 * The dimension tree is read from `<prefix>_dimension_tree`. Each leaf basis
 * matrix is read from `<prefix>_basis_matrix_<node>`, and each non-leaf
 * transfer tensor is read from `<prefix>_transfer_tensor_<node>`. The loaded
 * components are installed with reset_components, which updates the internal
 * sizes and ranks and checks consistency.
 */
template <size_t dimension, ::boba::execution_space space, typename data_t>
void read_from_hdf5_file(HierarchicalTucker<dimension, space, data_t>& htucker,
                         std::string_view filename,
                         std::string object_name = "")
{
  if (object_name.empty())
  {
    object_name = htucker.name();
  }

  std::string print_filename = std::string(filename);

  // Open the HDF5 file (read only)
  ::boba::detail::HDF5File h5_file(filename, "r");

  // Reconstruct the dimension tree from the file contents
  ::boba::DimensionTree dim_tree;
  dim_tree.read_from_file(print_filename + "_dimension_tree");
  const auto& is_leaf = dim_tree.get_is_leaf();

  // Next we need to rebuild the transfer tensors and basis matrices
  constexpr size_t num_nodes = HierarchicalTucker<dimension, space, data_t>::num_nodes;
  using B_type = typename HierarchicalTucker<dimension, space, data_t>::B_type;
  using U_type = typename HierarchicalTucker<dimension, space, data_t>::U_type;

  ::boba::Array<B_type, num_nodes> transfer_tensors;
  ::boba::Array<U_type, num_nodes> basis_matrices;

  // Read the transfer tensors and basis matrices from the HDF5 file
  for (size_t node = 0; node < num_nodes; ++node)
  {
    if (is_leaf[node])
    {
      basis_matrices[node].emplace();
      h5_file.read_array(object_name + "_basis_matrix_" + std::to_string(node),
                         basis_matrices[node].value());
    }
    else
    {
      transfer_tensors[node].emplace();
      h5_file.read_array(object_name + "_transfer_tensor_" + std::to_string(node),
                         transfer_tensors[node].value());
    }
  }

  // Now do a bulk reset of the transfer tensors, basis matrices, and dimension tree
  // using the data loaded from the files
  htucker.reset_components(std::move(transfer_tensors),
                           std::move(basis_matrices),
                           std::move(dim_tree));
}

} // namespace boba
