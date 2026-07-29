// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <filesystem>
#include <numeric>

namespace boba
{

/**
 * @brief Hierarchical Tucker decomposition.
 *
 * Implements the Hierarchical Tucker (HierarchicalTucker) tensor format and associated
 * operations.
 *
 * @tparam dimension Number of tensor dimensions
 * @tparam space Execution space
 * @tparam data_t Data type for tensor entries
 * @tparam index_t Index type for tensor dimensions
 */

template <size_t dimension, ::boba::execution_space space, typename _data_t>
class HierarchicalTucker
{

public:
  /**
   * @brief Underlying real-valued scalar type.
   */
  using data_t = _data_t;
  using real_data_t = real_type_t<data_t>;

  /**
   * @brief Complex (or real) HierarchicalTucker with data_t entries.
   */
  using HierarchicalTucker_type = HierarchicalTucker<dimension, space, data_t>;

  /**
   * @brief Real-valued HierarchicalTucker with real_data_t entries.
   */
  using HierarchicalTucker_real_type = HierarchicalTucker<dimension, space, real_data_t>;

  /**
   * @brief Optional transfer tensor using the current scalar type (data_t).
   */
  using B_type = std::optional<::boba::Tensor<3, space, data_t>>;

  /**
   * @brief Raw transfer tensor type (non-optional, data_t).
   */
  using B_raw_type = typename B_type::value_type;

  /**
   * @brief Optional real-valued transfer tensor.
   */
  using B_real_type = std::optional<::boba::Tensor<3, space, real_data_t>>;

  /**
   * @brief Raw real-valued transfer tensor type.
   */
  using B_raw_real_type = typename B_real_type::value_type;

  /**
   * @brief Optional basis matrix using the current scalar type (data_t).
   */
  using U_type = std::optional<::boba::Matrix<space, data_t>>;

  /**
   * @brief Raw basis matrix type (non-optional, data_t).
   */
  using U_raw_type = typename U_type::value_type;

  /**
   * @brief Optional real-valued basis matrix.
   */
  using U_real_type = std::optional<::boba::Matrix<space, real_data_t>>;

  /**
   * @brief Raw real-valued basis matrix type.
   */
  using U_raw_real_type = typename U_real_type::value_type;

  /**
   * @brief Type alias for intermediate matrices used in contractions.
   */
  using M_type = std::optional<::boba::Matrix<space, data_t>>;

  /**
   * @brief Type alias for reduced Gramian matrices.
   */
  using G_type = std::optional<::boba::Matrix<space, data_t>>;

  /**
   * @brief Total number of nodes in the dimension tree.
   */
  static constexpr size_t num_nodes = 2 * dimension - 1;

private:
  // -------------------------------------------------------------------------------------
  // Attributes and members
  // -------------------------------------------------------------------------------------

  /**
   * @brief Name identifier for the HierarchicalTucker object (default used for file I/O).
   */
  std::string m_name = "HierarchicalTucker";

  /**
   * @brief Relative truncation tolerance used in SVD-based rank reduction (default).
   */
  real_data_t svd_tolerance_relative = static_cast<real_data_t>(1.0e-12);

  /**
   * @brief Absolute truncation tolerance used in SVD-based rank reduction (default).
   */
  real_data_t svd_tolerance_absolute = static_cast<real_data_t>(1.0e-12);

  /**
   * @brief Indicates whether this tensor has been orthogonalized.
   */
  bool is_orthog = false;

  /**
   * @brief Transfer tensors/cores defined at the non-leaf nodes of the tree.
   *
   * Each tensor describes the interaction coefficients between the bases
   * associated with a pair of child dimensions.
   * The nodes are ordered in breadth-first traversal order.
   */
  ::boba::Array<B_type, num_nodes> transfer_tensors;

  /**
   * @brief Basis matrices for different nodes of the dimension tree.
   *
   * Each matrix represents the basis of the column space corresponding to
   * the matricization at that node.
   * The nodes are ordered in breadth-first traversal order.
   */
  ::boba::Array<U_type, num_nodes> basis_matrices;

  /**
   * @brief Hierarchical ranks associated with each node.
   *
   * Defines the rank of the subspace represented by each node in the
   * hierarchical decomposition.
   * The nodes are ordered in breadth-first traversal order.
   */
  ::boba::Array<size_t, num_nodes> ranks;

  /**
   * @brief Mode sizes (tensor extents) for each dimension.
   */
  ::boba::Array<size_t, dimension> sizes;

  /**
   * @brief Dimension tree defining the node relationships and traversal order.
   *
   * The DimensionTree encodes the topology of the HierarchicalTucker decomposition,
   * providing methods for obtaining post-order, pre-order, and breadth-first
   * traversals, as well as parent/child relationships between nodes.
   */
  ::boba::DimensionTree dim_tree;

public:
  // -------------------------------------------------------------------------------------
  // Class constructors/destructors
  // -------------------------------------------------------------------------------------

  /**
   * @brief Default constructor.
   *
   * Constructs an empty HierarchicalTucker object with uninitialized members.
   */
  HierarchicalTucker() = default;

  /**
   * @brief Copy constructor.
   */
  HierarchicalTucker(HierarchicalTucker_type const&) = default;

  /**
   * @brief Move constructor.
   */
  HierarchicalTucker(HierarchicalTucker_type&&) = default;

  /**
   * @brief Copy assignment operator.
   */
  HierarchicalTucker& operator=(HierarchicalTucker_type const&) = default;

  /**
   * @brief Move assignment operator.
   */
  HierarchicalTucker& operator=(HierarchicalTucker_type&&) = default;

  /**
   * @brief Construct a zero HierarchicalTucker object from sizes and a dimension tree.
   *
   * Initializes the tensor with zeros, rank-1 components, and stores the
   * provided dimension tree.
   *
   * @param[in] input_sizes mode sizes for each dimension
   * @param[in] _dim_tree dimension tree defining the hierarchical structure
   */
  HierarchicalTucker(::boba::Array<index_t, dimension> input_sizes,
                     ::boba::DimensionTree _dim_tree)
      : sizes(std::move(input_sizes)),
        dim_tree(std::move(_dim_tree))
  {
    boba_always_assert(sizes > 0, "Invalid size");
    fill_with_zeros();
  }

  /**
   * @brief Construct an HierarchicalTucker object from transfer tensors, bases, and a dimension tree.
   *
   * This constructor checks the consistency of the provided components and
   * computes the node ranks and mode sizes.
   *
   * @param[in] _transfer_tensors transfer tensors for each node
   * @param[in] _basis_matrices basis matrices for each node
   * @param[in] _dim_tree dimension tree defining the node relationships
   */
  HierarchicalTucker(::boba::Array<B_type, num_nodes> _transfer_tensors,
                     ::boba::Array<U_type, num_nodes> _basis_matrices,
                     ::boba::DimensionTree _dim_tree)
  {
    reset_components(std::move(_transfer_tensors), std::move(_basis_matrices), std::move(_dim_tree));
  }

  /**
   * @brief Reset this HierarchicalTucker object from transfer tensors, basis
   * matrices, and a dimension tree.
   *
   * This replaces the stored HTD components, updates the mode sizes and
   * hierarchical ranks, and checks that the resulting internal structure is
   * consistent.
   *
   * @param[in] new_transfer_tensors Transfer tensors for each node.
   * @param[in] new_basis_matrices Basis matrices for each node.
   * @param[in] new_dim_tree Dimension tree defining the node relationships.
   */
  void reset_components(::boba::Array<B_type, num_nodes> new_transfer_tensors,
                        ::boba::Array<U_type, num_nodes> new_basis_matrices,
                        ::boba::DimensionTree new_dim_tree)
  {
    transfer_tensors = std::move(new_transfer_tensors);
    basis_matrices = std::move(new_basis_matrices);
    dim_tree = std::move(new_dim_tree);

    // Fill in hierarchical ranks and mode extents, then check consistency.
    update_sizes_and_ranks();
    check_consistency();

    is_orthog = false;
  }

  /**
   * @brief Recompute `sizes` and `ranks` from the current tensors.
   *
   * Assumes `dim_tree`, `basis_matrices`, and `transfer_tensors` are already
   * populated, and that `sizes` and `ranks` have the correct length.
   */
  void update_sizes_and_ranks()
  {
    const auto& is_leaf = dim_tree.get_is_leaf();
    const auto& dims = dim_tree.get_dims();

    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (is_leaf[node])
      {
        // Leaf: get size from basis matrix U[node]
        const auto& U = basis_matrices[node];
        boba_assert(U.has_value(), "Leaf node missing basis matrix.");

        sizes[dims[node][0]] = U->sizes(0);
        ranks[node] = U->sizes(1);
      }
      else
      {
        // Non-leaf: get rank from transfer tensor B[node]
        const auto& B = transfer_tensors[node];
        boba_assert(B.has_value(), "Non-leaf node missing transfer tensor.");

        ranks[node] = B->sizes(2);
      }
    }
  }

  /**
   * @brief Verifies the structural and dimensional consistency of the HierarchicalTucker object.
   *
   * Ensures that:
   * - The number of nodes matches the dimension tree.
   * - The root node transfer tensor exists and has rank 1.
   * - Each node’s child and parent dimensions are consistent with the tree structure.
   *
   * @throws Assertion failure if any inconsistency is detected.
   */
  void check_consistency() const
  {
    const size_t num_nodes_basis = basis_matrices.size();
    const size_t num_nodes_transfer = transfer_tensors.size();
    const size_t num_nodes_tree = dim_tree.get_num_nodes();

    boba_always_assert(num_nodes_basis == num_nodes_tree && num_nodes_transfer == num_nodes_tree,
                       "Number of nodes in basis_matrices and transfer_tensors must match dimension tree.");

    // The root node transfer tensor is always a matrix
    boba_assert(transfer_tensors[0].has_value(), "The root node does not have a transfer tensor.");
    boba_always_assert(transfer_tensors[0]->sizes(2) == 1, "The root node transfer tensor is not a matrix.");

    const auto& is_leaf = dim_tree.get_is_leaf();

    for (size_t node = 1; node < num_nodes; ++node)
    {

      const size_t parent = dim_tree.get_parent_of_node(node);
      const bool node_is_left_child = dim_tree.is_left_child(node);

      if (is_leaf[node])
      {

        boba_assert(transfer_tensors[parent].has_value(),
                    "Missing transfer tensor at node " + std::to_string(parent) + ".");

        boba_assert(basis_matrices[node].has_value(),
                    "Missing basis matrix at node " + std::to_string(node) + ".");

        const auto& parent_transfer_tensor = transfer_tensors[parent].value();
        const auto& node_basis_matrix = basis_matrices[node].value();

        // Validate dimension compatibility with parent transfer tensor
        if (node_is_left_child)
        {
          boba_always_assert(node_basis_matrix.sizes(1) == parent_transfer_tensor.sizes(0),
                             "Leaf basis matrix at node " + std::to_string(node) +
                               " has incompatible rank with parent's transfer tensor (left child mismatch).");
        }
        else
        {
          boba_always_assert(node_basis_matrix.sizes(1) == parent_transfer_tensor.sizes(1),
                             "Leaf basis matrix at node " + std::to_string(node) +
                               " has incompatible rank with parent's transfer tensor (right child mismatch).");
        }
      }
      else
      {
        boba_assert(transfer_tensors[parent].has_value(),
                    "Missing transfer tensor at node " + std::to_string(parent) + ".");

        boba_assert(transfer_tensors[node].has_value(),
                    "Missing transfer tensor at node " + std::to_string(node) + ".");

        const auto& parent_transfer_tensor = transfer_tensors[parent].value();
        const auto& node_transfer_tensor = transfer_tensors[node].value();

        // Validate dimension compatibility with parent node
        if (node_is_left_child)
        {
          boba_always_assert(node_transfer_tensor.sizes(2) == parent_transfer_tensor.sizes(0),
                             "Transfer tensor at node " + std::to_string(node) +
                               " is incompatible with parent transfer tensor (left child mismatch).");
        }
        else
        {
          boba_always_assert(node_transfer_tensor.sizes(2) == parent_transfer_tensor.sizes(1),
                             "Transfer tensor at node " + std::to_string(node) +
                               " is incompatible with parent transfer tensor (right child mismatch).");
        }
      }
    }
  }

  /**
   * @brief Copy constructor between HierarchicalTucker objects in different execution spaces.
   *
   * @tparam rhs_space execution space of the right-hand-side HierarchicalTucker
   * @param[in] rhs source HierarchicalTucker object to copy from
   */
  template <execution_space rhs_space>
  HierarchicalTucker(HierarchicalTucker<dimension, rhs_space, data_t> const& rhs)
      : m_name(rhs.m_name),
        svd_tolerance_relative(rhs.svd_tolerance_relative),
        svd_tolerance_absolute(rhs.svd_tolerance_absolute),
        is_orthog(rhs.is_orthog),
        transfer_tensors(rhs.transfer_tensors),
        basis_matrices(rhs.basis_matrices),
        ranks(rhs.ranks),
        sizes(rhs.sizes),
        dim_tree(rhs.dim_tree)
  {
  }

  /**
   * @brief Copy assignment operator between HierarchicalTucker objects in different execution spaces.
   *
   * @tparam rhs_space execution space of the right-hand-side HierarchicalTucker
   * @param[in] rhs source HierarchicalTucker object to assign from
   * @return reference to this object after assignment
   */
  template <execution_space rhs_space>
  HierarchicalTucker& operator=(HierarchicalTucker<dimension, rhs_space, data_t> const& rhs)
  {
    m_name = rhs.m_name;
    svd_tolerance_relative = rhs.svd_tolerance_relative;
    svd_tolerance_absolute = rhs.svd_tolerance_absolute;
    is_orthog = rhs.is_orthog;
    transfer_tensors = rhs.transfer_tensors;
    basis_matrices = rhs.basis_matrices;
    ranks = rhs.ranks;
    sizes = rhs.sizes;
    dim_tree = rhs.dim_tree;
    return *this;
  }

  /// \brief destructor
  ~HierarchicalTucker() = default;

  /**
   * @brief Initialize this object to represent a zero HierarchicalTucker tensor.
   *
   * Equivalent to creating a zero tensor of the given sizes and tree
   * structure. All ranks are set to 1, and the data arrays are filled with
   * zeros.
   */
  void fill_with_zeros()
  {
    checkpoint();
    const auto& is_leaf = dim_tree.get_is_leaf();
    const auto& dim2idx = dim_tree.get_dim2idx();

    // At leaf nodes, transfer tensors are empty and basis matrices are non-empty.
    for (size_t d = 0; d < dimension; ++d)
    {
      auto node = dim2idx[d];
      transfer_tensors[node] = std::nullopt;
      basis_matrices[node].emplace();
      basis_matrices[node]->resize({sizes[d], 1});
      basis_matrices[node]->fill_with_zeros();
    }

    // At non-leaf nodes, transfer tensors are non-empty while basis matrices are empty.
    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (!is_leaf[node])
      {
        transfer_tensors[node].emplace();
        transfer_tensors[node]->resize({1, 1, 1});
        transfer_tensors[node]->fill_with_zeros();
        basis_matrices[node] = std::nullopt;
      }
    }

    // Set all node ranks to 1.
    ranks = ::boba::filled_array<num_nodes, size_t>(1);
    checkpoint();
  }

  /**
   * @brief Fill this HierarchicalTucker tensor with a constant value.
   *
   * Equivalent to creating a full tensor whose entries are all equal to
   * @p value.
   *
   * @param[in] value constant value used to fill all tensor entries
   */
  void fill_with(data_t value)
  {
    const auto& is_leaf = dim_tree.get_is_leaf();

    // Absorb the constant factor into the root transfer tensor.
    transfer_tensors[0]->fill_with(value);

    for (size_t node = 1; node < num_nodes; ++node)
    {
      if (is_leaf[node])
      {
        // Leaf nodes: basis matrices are non-empty.
        boba_assert(basis_matrices[node].has_value(), "Basis matrix is uninitialized.");
        basis_matrices[node]->fill_with(static_cast<data_t>(1));
      }
      else
      {
        // Non-leaf nodes: transfer tensors are non-empty.
        boba_assert(transfer_tensors[node].has_value(), "Transfer tensor is uninitialized.");
        transfer_tensors[node]->fill_with(static_cast<data_t>(1));
      }
    }
  }

  /**
   * @brief Fill this HierarchicalTucker tensor with random values and random ranks.
   *
   * Each entry is drawn uniformly from (0,1), and ranks are sampled
   * uniformly from the integers [1,10]. The root node retains rank 1.
   */
  void fill_with_random()
  {
    const auto& is_leaf = dim_tree.get_is_leaf();
    const auto& dim2idx = dim_tree.get_dim2idx();

    // Create random number generator for ranks in [1,10].
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, 10);

    // Root rank is fixed to 1; sample ranks for remaining nodes.
    std::generate(ranks.data() + 1, ranks.data() + num_nodes, [&]
    {
      return distrib(gen);
    });

    // Generate leaf node bases for each dimension.
    for (size_t d = 0; d < dimension; ++d)
    {
      // Leaf node: basis matrix is of size N_t × rank_t.
      size_t rank_t = ranks[dim2idx[d]];

      boba_assert(basis_matrices[dim2idx[d]].has_value(), "Basis matrix is uninitialized.");
      basis_matrices[dim2idx[d]]->resize({sizes[d], rank_t});
      basis_matrices[dim2idx[d]]->fill_with_random();
    }

    // Generate transfer tensors for non-leaf nodes.
    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (!is_leaf[node])
      {
        // Non-leaf: transfer tensor is of size rank_tl × rank_tr × rank_t.
        auto children = dim_tree.get_children_of_node(node);
        size_t rank_tl = ranks[children[0]];
        size_t rank_tr = ranks[children[1]];
        size_t rank_t = ranks[node];

        boba_assert(transfer_tensors[node].has_value(), "Transfer tensor is uninitialized.");
        transfer_tensors[node]->resize({rank_tl, rank_tr, rank_t});
        transfer_tensors[node]->fill_with_random();
      }
    }
  }

  // -------------------------------------------------------------------------------------
  // Operators and overloads for arithmetic involving HierarchicalTucker objects
  // -------------------------------------------------------------------------------------

  /**
   * @brief Compare two HierarchicalTucker objects for equivalence.
   *
   * Equivalence means that the objects have identical fields.
   *
   * @param[in] other_ht right-hand side HierarchicalTucker object to compare against
   */
  bool operator==(HierarchicalTucker_type const& other_ht) const
  {
    // Check basic properties for equality
    if (dim_tree != other_ht.dim_tree)
    {
      return false;
    }
    if (num_nodes != other_ht.num_nodes)
    {
      return false;
    }
    if (sizes != other_ht.sizes)
    {
      return false;
    }
    if (ranks != other_ht.ranks)
    {
      return false;
    }
    if (is_orthog != other_ht.is_orthog)
    {
      return false;
    }

    // Next, compare the transfer tensors and basis matrices
    const auto& other_transfer_tensors = other_ht.get_transfer_tensors();
    const auto& other_basis_matrices = other_ht.get_basis_matrices();

    // Define a helper lambda to compare optionals containing tensors/matrices
    auto optionals_are_equal = [&](const auto& optA, const auto& optB)
    {
      // Case 1: Both have values so we can compare numerically with is_tiny()
      if (optA.has_value() && optB.has_value())
      {
        const auto& A = optA.value();
        const auto& B = optB.value();

        auto diff = ::boba::norm_difference_inf(A, B);
        return ::boba::is_tiny(diff);
      }

      // Case 2: One has value, one does not (mismatch)
      if (optA.has_value() != optB.has_value())
        return false;

      // Case 3: Both empty (equal)
      return true;
    };

    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (!optionals_are_equal(transfer_tensors[node], other_transfer_tensors[node]))
      {
        return false;
      }
      if (!optionals_are_equal(basis_matrices[node], other_basis_matrices[node]))
      {
        return false;
      }
    }

    return true;
  }

  /**
   * @brief In-place addition of another HierarchicalTucker object.
   *
   * Concatenates the transfer tensors and basis matrices of @p rhs into this
   * object, assuming both have identical dimension trees and mode sizes.
   *
   * @param rhs Right-hand side HierarchicalTucker object to add.
   * @return Reference to this object after addition.
   */
  HierarchicalTucker_type& operator+=(HierarchicalTucker_type const& rhs)
  {
    // Ensure that the dimension trees and sizes match.
    boba_always_assert(dim_tree == rhs.dim_tree, "Incompatible dimension trees.");
    boba_always_assert_equal(sizes, rhs.sizes, "Incompatible mode sizes.");

    const auto& is_leaf = dim_tree.get_is_leaf();
    const auto& dims = dim_tree.get_dims();

    // Concatenate the root node transfer tensors (special case: root is a matrix)
    concat_root_transfer_tensors(transfer_tensors[0], rhs.transfer_tensors[0]);

    // Concatenate the remaining nodes, updating the sizes and ranks as we traverse the tree
    for (size_t node = 1; node < num_nodes; ++node)
    {
      if (is_leaf[node])
      {
        concat_leaf_bases(basis_matrices[node], rhs.basis_matrices[node]);
        sizes[dims[node][0]] = basis_matrices[node]->sizes(0);
        ranks[node] = basis_matrices[node]->sizes(1);
      }
      else
      {
        concat_non_root_transfer_tensors(transfer_tensors[node], rhs.transfer_tensors[node]);
        ranks[node] = transfer_tensors[node]->sizes(2);
      }
    }

    // This HierarchicalTucker object is not orthogonalized
    is_orthog = false;

    return *this;
  }

  /**
   * @brief Addition of two HierarchicalTucker objects.
   *
   * @param rhs Right-hand side HierarchicalTucker object to add.
   * @return New HierarchicalTucker object representing the sum.
   */
  HierarchicalTucker_type operator+(HierarchicalTucker_type const& rhs) const
  {
    HierarchicalTucker_type output{*this};
    output += rhs;
    return output;
  }

  /**
   * @brief In-place scalar multiplication.
   *
   * Multiplies this HierarchicalTucker tensor by a scalar value. The scalar
   * is absorbed into the root node transfer tensor.
   *
   * @param scalar Scalar multiplier.
   * @return Reference to this object after scaling.
   */
  template <typename Scalar>
    requires std::is_convertible_v<Scalar, data_t>
  HierarchicalTucker_type& operator*=(Scalar scalar)
  {
    boba_assert(transfer_tensors[0].has_value(), "Root node transfer tensor is not initialized.");

    // Absorb the scalar into the root transfer tensor.
    (*transfer_tensors[0]) *= static_cast<data_t>(scalar);
    return *this;
  }

  /**
   * @brief Scalar multiplication.
   *
   * @param scalar Scalar multiplier.
   * @return New HierarchicalTucker object representing the scaled tensor.
   */
  template <typename Scalar>
    requires std::is_convertible_v<Scalar, data_t>
  HierarchicalTucker_type operator*(Scalar scalar) const
  {
    HierarchicalTucker_type output{*this};
    output *= scalar;
    return output;
  }

  /**
   * @brief Defines scalar * HierarchicalTucker.
   *
   * @param scalar Scalar multiplier.
   * @param rhs An HierarchicalTucker to be scaled.
   * @return New HierarchicalTucker object representing the scaled tensor.
   */
  template <typename Scalar>
    requires std::is_convertible_v<Scalar, data_t>
  friend HierarchicalTucker_type operator*(Scalar scalar, HierarchicalTucker_type const& rhs)
  {
    HierarchicalTucker_type output{rhs};
    output *= scalar;
    return output;
  }

  /**
   * @brief Defines the element-wise (Hadamard) product between HTuckers.
   *
   * @param other An HierarchicalTucker to be multiplied element-wise.
   * @return New HierarchicalTucker object representing the element-wise product.
   */
  HierarchicalTucker_type operator*(HierarchicalTucker_type const& other) const
  {
    boba_always_assert(dim_tree == other.get_dim_tree(), "Hadamard product: Dimension trees must be identical");

    HierarchicalTucker_type output{other};

    const auto& is_leaf = dim_tree.get_is_leaf();
    const auto& dims = dim_tree.get_dims();

    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (is_leaf[node])
      {
        boba_assert(basis_matrices[node].has_value(), "Basis matrix is uninitialized.");
        boba_assert(other.basis_matrices[node].has_value(), "Basis matrix is uninitialized.");

        const auto& U1 = basis_matrices[node].value();
        const auto& U2 = other.basis_matrices[node].value();

        // Compute the column-wise Khatri-Rao product of U1 and U2
        auto U_node_out = ::boba::mode_n_tensor_product(U1, U2, 1);
        output.basis_matrices[node] = std::move(U_node_out);

        // Update the size and rank of the current node
        output.sizes[dims[node][0]] = output.basis_matrices[node]->sizes(0);
        output.ranks[node] = output.basis_matrices[node]->sizes(1);
      }
      else
      {
        boba_assert(transfer_tensors[node].has_value(), "Transfer tensor is uninitialized.");
        boba_assert(other.transfer_tensors[node].has_value(), "Transfer tensor is uninitialized.");

        const auto& B1 = transfer_tensors[node].value();
        const auto& B2 = other.transfer_tensors[node].value();

        // Compute the Kronecker product of B1 and B2
        auto B_node_out = ::boba::tensor_product(B1, B2);
        output.transfer_tensors[node] = std::move(B_node_out);

        // Update the rank of the current node
        output.ranks[node] = output.transfer_tensors[node]->sizes(2);
      }
    }

    return output;
  }

  /**
   * @brief In-place subtraction of another HierarchicalTucker object.
   *
   * @param rhs Right-hand side HierarchicalTucker object to subtract.
   * @return Reference to this object after subtraction.
   */
  HierarchicalTucker_type& operator-=(HierarchicalTucker_type const& rhs)
  {
    HierarchicalTucker_type minus_rhs{rhs};
    minus_rhs *= -1;
    *this += minus_rhs;
    return *this;
  }

  /**
   * @brief Subtraction of two HierarchicalTucker objects.
   *
   * @param rhs Right-hand side HierarchicalTucker object to subtract.
   * @return New HierarchicalTucker object representing the difference.
   */
  HierarchicalTucker_type operator-(HierarchicalTucker_type const& rhs) const
  {
    HierarchicalTucker_type output{*this};
    output -= rhs;
    return output;
  }

  /**
   * @brief Concatenates the root node transfer tensors of two HierarchicalTucker objects.
   *
   * Performs a block-diagonal concatenation of two rank-1 (matrix-like) root transfer tensors.
   *
   * @param B1 Reference to the root transfer tensor of the first HierarchicalTucker object (modified in-place).
   * @param B2 Constant reference to the root transfer tensor of the second HierarchicalTucker object.
   *
   * @throws Assertion failure if input tensors are uninitialized or not rank-1.
   */
  void concat_root_transfer_tensors(B_type& B1, const B_type& B2)
  {
    boba_assert(B1.has_value() && B2.has_value(),
                "Non-root node transfer tensors must be initialized.");

    auto B1_sizes = B1->sizes();
    auto B2_sizes = B2->sizes();

    boba_always_assert(B1_sizes[2] == 1 && B2_sizes[2] == 1,
                       "Input transfer tensors must have rank 1 at the root nodes.");

    ::boba::Array<size_t, 3> output_size{B1_sizes[0] + B2_sizes[0],
                                         B1_sizes[1] + B2_sizes[1],
                                         1};

    ::boba::Tensor<3, space, data_t> output(output_size);
    output.fill_with(static_cast<data_t>(0));

    auto output_view = output.view();
    auto B1_view = B1->const_view();
    auto B2_view = B2->const_view();

    // Fill first block (B1)
    ::boba::loop<space, 3>(B1_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      output_view(mid) = B1_view(mid);
    });

    // Fill second block (B2)
    ::boba::loop<space, 3>(B2_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      size_t i_out = mid[0] + B1_sizes[0];
      size_t j_out = mid[1] + B1_sizes[1];
      size_t k_out = mid[2];
      output_view({i_out, j_out, k_out}) = B2_view(mid);
    });

    B1 = std::move(output);
  }

  /**
   * @brief Concatenates transfer tensors at internal (non-root) nodes.
   *
   * Performs a block-diagonal concatenation of two 3-tensors.
   *
   * @param B1 Reference to the target transfer tensor (modified in-place).
   * @param B2 Constant reference to the second transfer tensor.
   *
   * @throws Assertion failure if either tensor is uninitialized.
   */
  void concat_non_root_transfer_tensors(B_type& B1, const B_type& B2)
  {
    boba_assert(B1.has_value() && B2.has_value(),
                "Non-root node transfer tensors must be initialized.");

    auto B1_sizes = B1->sizes();
    auto B2_sizes = B2->sizes();
    auto output_size = B1_sizes + B2_sizes;

    ::boba::Tensor<3, space, data_t> output(output_size);
    output.fill_with(static_cast<data_t>(0));

    auto output_view = output.view();
    auto B1_view = B1->const_view();
    auto B2_view = B2->const_view();

    // Fill first block (B1)
    ::boba::loop<space, 3>(B1_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      output_view(mid) = B1_view(mid);
    });

    // Fill second block (B2)
    ::boba::loop<space, 3>(B2_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto mid_out = mid + B1_sizes;
      output_view(mid_out) = B2_view(mid);
    });

    B1 = std::move(output);
  }

  /**
   * @brief Concatenates basis matrices at the leaf nodes.
   *
   * Performs column-wise concatenation of two leaf basis matrices.
   *
   * @param U1 Reference to the target basis matrix (modified in-place).
   * @param U2 Constant reference to the second basis matrix.
   *
   * @throws Assertion failure if inputs are uninitialized or incompatible in row dimension.
   */
  void concat_leaf_bases(U_type& U1, const U_type& U2)
  {
    boba_assert(U1.has_value() && U2.has_value(),
                "Leaf node matrices must be initialized.");

    auto U1_sizes = U1->sizes();
    auto U2_sizes = U2->sizes();

    boba_always_assert(U1->sizes(0) == U2->sizes(0), "Incompatible U matrix sizes.");

    ::boba::Array<size_t, 2> output_size({U1_sizes[0], U1_sizes[1] + U2_sizes[1]});
    ::boba::Matrix<space, data_t> output(output_size);
    output.fill_with(static_cast<data_t>(0));

    auto output_view = output.view();
    auto U1_view = U1->const_view();
    auto U2_view = U2->const_view();

    // Fill first block (U1)
    ::boba::loop<space, 2>(U1_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 2> mid)
    {
      output_view(mid) = U1_view(mid);
    });

    // Fill second block (U2)
    ::boba::loop<space, 2>(U2_sizes,
                           [=] __boba_host_device__(::boba::Array<size_t, 2> mid)
    {
      size_t i_out = mid[0];
      size_t j_out = mid[1] + U1_sizes[1];
      output_view({i_out, j_out}) = U2_view(mid);
    });

    U1 = std::move(output);
  }

  // -------------------------------------------------------------------------------------
  // Methods to orthogonalize, compress, and uncompress an HierarchicalTucker object
  // -------------------------------------------------------------------------------------

  /**
   * @brief Orthogonalizes the bases of the HierarchicalTucker tensor.
   *
   * Performs a recursive QR factorization from the leaves to the root,
   * pushing the upper triangular factors into parent transfer tensors.
   * After completion, the object is marked as orthogonalized.
   */
  void orthogonalize()
  {
    // Do nothing if the tensor has already been orthogonalized
    if (is_orthog)
    {
      return;
    }

    const auto& post_order_nodes = dim_tree.get_post_order_nodes();
    const auto& is_leaf = dim_tree.get_is_leaf();

    // Initialize QR factorization object
    QR<space, data_t> qr;

    // Define the index labels for tensor contractions
    // These labels will change depending on whether the current node is a left (l) or right (r) child
    const auto R_index_labels_l = ::boba::Array<std::string, 2>{"i", "l"};
    const auto R_index_labels_r = ::boba::Array<std::string, 2>{"j", "l"};

    const auto transfer_tensor_index_labels_l = ::boba::Array<std::string, 3>{"l", "j", "k"};
    const auto transfer_tensor_index_labels_r = ::boba::Array<std::string, 3>{"i", "l", "k"};

    for (size_t idx = 0; idx < num_nodes - 1; ++idx)
    {
      size_t node = post_order_nodes[idx];
      size_t parent_node = dim_tree.get_parent_of_node(node);

      if (is_leaf[node])
      {
        // Leaf: QR factorization of the basis U_t
        boba_assert(basis_matrices[node].has_value(),
                    "Missing basis matrix at node " + std::to_string(node) + " during orthogonalize().");
        qr(basis_matrices[node].value());
        basis_matrices[node].emplace(qr.Q);
      }
      else
      {
        // Internal node: QR factorization of the unfolding B_t
        boba_assert(transfer_tensors[node].has_value(),
                    "Missing transfer tensor at node " + std::to_string(node) + " during orthogonalize().");
        auto unfolding = ::boba::unfold(transfer_tensors[node].value(), std::vector<size_t>{0, 1}, std::vector<size_t>{2});
        qr(unfolding);

        // Set the sizes for the new transfer tensor for the current node
        auto n_tl = transfer_tensors[node]->sizes(0);
        auto n_tr = transfer_tensors[node]->sizes(1);
        auto rank_t = qr.Q.sizes(1);

        // Fold Q into a tensor of size n_tl x n_tr x rank_t and store in B_t
        auto Q_tensor = ::boba::fold(qr.Q, ::boba::Array<size_t, 3>{n_tl, n_tr, rank_t}, std::vector<size_t>{0, 1}, std::vector<size_t>{2});
        transfer_tensors[node] = std::move(Q_tensor);
      }

      // Propagate the R matrix into the transfer tensor of the parent node before moving to the next node
      auto is_left = dim_tree.is_left_child(node);
      const auto& R_index_labels = (is_left) ? R_index_labels_l : R_index_labels_r;
      const auto& transfer_tensor_index_labels = (is_left) ? transfer_tensor_index_labels_l : transfer_tensor_index_labels_r;

      boba_assert(transfer_tensors[parent_node].has_value(),
                  "Missing transfer tensor at node " + std::to_string(parent_node) + " during orthogonalize().");

      auto updated_parent_tensor = ::boba::tensor_contraction_single_index(R_index_labels, qr.R, transfer_tensor_index_labels, transfer_tensors[parent_node].value(), {"i", "j", "k"});

      transfer_tensors[parent_node] = std::move(updated_parent_tensor);
    }

    is_orthog = true;
  }

  /**
   * @brief Compresses a boba::Tensor into a boba::HierarchicalTucker object.
   *
   * The code builds the hierarchical representation from the leaves to the root
   * using truncated SVDs along each dimension (post-order traversal).
   *
   * @param input  Input boba::Tensor to compress.
   *
   * @note The resulting HierarchicalTucker is not marked as orthogonalized.
   */
  void compress(::boba::Tensor<dimension, space, data_t>& input)
  {
    boba_always_assert_equal(dim_tree.get_num_dims(), input.get_dimension(), "The input tensor and the current dimension tree have inconsistent dimensions.");

    const auto& tree_dims = dim_tree.get_dims();
    const auto& post_order_nodes = dim_tree.get_post_order_nodes();
    const auto& is_leaf = dim_tree.get_is_leaf();

    SVD<space, data_t> svd;
    svd.tolerance_relative = svd_tolerance_relative;
    svd.tolerance_absolute = svd_tolerance_absolute;

    for (size_t node : post_order_nodes)
    {
      auto input_unfolding = ::boba::unfold(input, tree_dims[node]);

      // For non-root nodes, we use the SVD of the unfolding and store the basis
      // At the root node, the unfolding is taken along all dimensions, so the tensor is vectorized
      if (node > 0)
      {
        svd(input_unfolding);
        basis_matrices[node].emplace(svd.U);
        ranks[node] = basis_matrices[node]->sizes(1);
      }
      else
      {
        basis_matrices[node].emplace(input_unfolding);
        ranks[node] = 1;
      }

      if (!is_leaf[node])
      {
        size_t left_child = dim_tree.get_left_child_of_node(node);
        size_t right_child = dim_tree.get_right_child_of_node(node);

        boba_assert(basis_matrices[left_child].has_value(),
                    "Left child basis is missing for node " + std::to_string(node));
        boba_assert(basis_matrices[right_child].has_value(),
                    "Right child basis is missing for node " + std::to_string(node));
        boba_assert(basis_matrices[node].has_value(),
                    "Basis matrix is missing for node " + std::to_string(node));

        const auto& left_child_basis = basis_matrices[left_child].value();
        const auto& right_child_basis = basis_matrices[right_child].value();
        const auto& node_basis_matrix = basis_matrices[node].value();

        size_t n_tl = left_child_basis.sizes(0);
        size_t n_tr = right_child_basis.sizes(0);
        size_t rank_t = node_basis_matrix.sizes(1);

        // Fold node basis into a tensor of size n_tl x n_tr x rank_t
        auto node_basis_tensor = ::boba::fold(
          node_basis_matrix,
          ::boba::Array<size_t, 3>{n_tl, n_tr, rank_t},
          std::vector<size_t>{0, 1},
          std::vector<size_t>{2});

        auto left_contracted_tensor = ::boba::tensor_contraction_single_index({"l", "i"}, ::boba::get_conj(left_child_basis), {"l", "j", "k"}, node_basis_tensor, {"i", "j", "k"});

        auto node_transfer_tensor = ::boba::tensor_contraction_single_index({"l", "j"}, ::boba::get_conj(right_child_basis), {"i", "l", "k"}, left_contracted_tensor, {"i", "j", "k"});
        transfer_tensors[node] = std::move(node_transfer_tensor);

        // We no longer need the bases from the non-leaf children, so we can clear them to save some memory
        if (!is_leaf[left_child])
        {
          basis_matrices[left_child].reset();
        }
        if (!is_leaf[right_child])
        {
          basis_matrices[right_child].reset();
        }
      }
    }

    // The basis at the root node can be safely deleted to save memory
    basis_matrices[0].reset();
    is_orthog = false;
  }

  /**
   * @brief Computes reduced Gramians for an HierarchicalTucker object.
   *
   * The reduced Gramians are calculated using a recursive algorithm that
   * proceeds from the root nodes down to the leaf nodes.
   *
   * @return Array of Gramian matrices at each node.
   */
  ::boba::Array<G_type, num_nodes> compute_reduced_gramians()
  {
    // Make sure the tensor is orthogonalized to simplify the calculations
    orthogonalize();

    ::boba::Array<G_type, num_nodes> gramians;
    const auto& is_leaf = dim_tree.get_is_leaf();

    // Initialize the root node Gramian to 1
    gramians[0].emplace();
    gramians[0]->resize({1, 1});
    gramians[0]->fill_with(static_cast<data_t>(1));

    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (!is_leaf[node])
      {
        const auto& children = dim_tree.get_children_of_node(node);

        boba_assert(transfer_tensors[node].has_value(),
                    "Missing transfer tensor at node " + std::to_string(node) + " during compute_reduced_gramians().");
        boba_assert(gramians[node].has_value(),
                    "Missing Gramian at node " + std::to_string(node) + " during compute_reduced_gramians().");

        const auto& node_transfer_tensor = transfer_tensors[node].value();
        const auto& node_gramian = gramians[node].value();

        auto node_transfer_tensor_conj = ::boba::get_conj(node_transfer_tensor);

        auto B_contracted = ::boba::tensor_contraction_single_index({"i", "j", "l"}, node_transfer_tensor_conj, {"k", "l"}, node_gramian, {"i", "j", "k"});

        auto left_gramian = ::boba::tensor_contraction_double_index({"m", "j", "k"}, node_transfer_tensor_conj, {"i", "j", "k"}, B_contracted, {"m", "i"});

        auto right_gramian = ::boba::tensor_contraction_double_index({"i", "n", "k"}, node_transfer_tensor_conj, {"i", "j", "k"}, B_contracted, {"n", "j"});

        gramians[children[0]].emplace(std::move(left_gramian));
        gramians[children[1]].emplace(std::move(right_gramian));
      }
    }

    return gramians;
  }

  /**
   * @brief Truncates an HierarchicalTucker tensor using recursive reduced Gramians.
   *
   * Performs hierarchical rounding by computing local SVDs of the Gramians
   * and updating both bases and transfer tensors accordingly.
   */
  void round()
  {
    // First, orthogonalize the HTD to simplify the recusive calculations
    orthogonalize();

    // Build the reduced Gramian matrices G^{(t)} = X^{(t)} X^{(t), T} at each node
    auto gramians = compute_reduced_gramians();

    SVD<space, data_t> svd;
    svd.tolerance_relative = ::boba::pow(svd_tolerance_relative, 2.0);
    svd.tolerance_absolute = ::boba::pow(svd_tolerance_absolute, 2.0);

    const auto& is_leaf = dim_tree.get_is_leaf();

    // Define the index labels for tensor contractions
    // These labels will change depending on whether the current node is a left (l) or right (r) child
    const auto gramian_index_labels_l = ::boba::Array<std::string, 2>{"l", "i"};
    const auto gramian_index_labels_r = ::boba::Array<std::string, 2>{"l", "j"};

    const auto transfer_tensor_index_labels_l = ::boba::Array<std::string, 3>{"l", "j", "k"};
    const auto transfer_tensor_index_labels_r = ::boba::Array<std::string, 3>{"i", "l", "k"};

    for (size_t node = 1; node < num_nodes; ++node)
    {
      boba_assert(gramians[node].has_value(),
                  "Missing Gramian at node " + std::to_string(node) + " during rounding.");

      // Apply SVD truncation to the Gramian at the current node
      //
      // TODO: replace this with an eigendecomposition since the Gramians are symmetric
      svd(gramians[node].value());
      const auto& gramian_basis = svd.U;

      if (is_leaf[node])
      {
        boba_assert(basis_matrices[node].has_value(),
                    "Missing basis matrix at leaf node " + std::to_string(node) + " during rounding.");
        auto& node_basis = basis_matrices[node].value();
        node_basis = node_basis * gramian_basis;
        ranks[node] = node_basis.sizes(1);
      }
      else
      {
        boba_assert(transfer_tensors[node].has_value(),
                    "Missing transfer tensor at node " + std::to_string(node) + " during rounding.");

        auto& node_transfer_tensor = transfer_tensors[node].value();

        auto truncated_transfer_tensor = ::boba::tensor_contraction_single_index({"i", "j", "l"}, node_transfer_tensor, {"l", "k"}, gramian_basis, {"i", "j", "k"});

        transfer_tensors[node] = std::move(truncated_transfer_tensor);
        ranks[node] = transfer_tensors[node]->sizes(2);
      }

      size_t parent_node = dim_tree.get_parent_of_node(node);

      // Propagate the gramian basis matrix into the transfer tensor of the parent node before moving to the next node
      auto is_left = dim_tree.is_left_child(node);
      const auto& gramian_index_labels = (is_left) ? gramian_index_labels_l : gramian_index_labels_r;
      const auto& transfer_tensor_index_labels = (is_left) ? transfer_tensor_index_labels_l : transfer_tensor_index_labels_r;

      boba_assert(transfer_tensors[parent_node].has_value(),
                  "Missing transfer tensor at node " + std::to_string(parent_node) + " during rounding.");

      auto& parent_transfer_tensor = transfer_tensors[parent_node].value();

      auto updated_parent_transfer_tensor = ::boba::tensor_contraction_single_index(gramian_index_labels, ::boba::get_conj(gramian_basis), transfer_tensor_index_labels, parent_transfer_tensor, {"i", "j", "k"});

      transfer_tensors[parent_node] = std::move(updated_parent_transfer_tensor);
      ranks[parent_node] = transfer_tensors[parent_node]->sizes(2);
    }
  }

  /**
   * @brief Converts the HierarchicalTucker representation back into a boba::Tensor.
   *
   * Performs upward contractions from the leaves to the root,
   * reconstructing the full multidimensional tensor in dense form.
   *
   * @return boba::Tensor corresponding to this HierarchicalTucker object.
   */
  ::boba::Tensor<dimension, space, data_t> decompress() const
  {
    const auto& post_order_nodes = dim_tree.get_post_order_nodes();
    const auto& is_leaf = dim_tree.get_is_leaf();

    // Make local copies of the bases to avoid modifying the current HierarchicalTucker instance
    auto local_basis_matrices = basis_matrices;

    for (size_t node : post_order_nodes)
    {
      if (is_leaf[node])
      {
        continue;
      }
      else
      {
        const auto& children = dim_tree.get_children_of_node(node);

        boba_assert(transfer_tensors[node].has_value(),
                    "Missing transfer tensor at node " + std::to_string(node) + " during conversion to tensor.");

        boba_assert(local_basis_matrices[children[0]].has_value(),
                    "Missing basis matrix at node " + std::to_string(children[0]) + " during conversion to tensor.");

        boba_assert(local_basis_matrices[children[1]].has_value(),
                    "Missing basis matrix at node " + std::to_string(children[1]) + " during conversion to tensor.");

        const auto& node_transfer_tensor = transfer_tensors[node].value();
        const auto& left_child_basis = local_basis_matrices[children[0]].value();
        const auto& right_child_basis = local_basis_matrices[children[1]].value();

        auto left_contracted_basis = ::boba::tensor_contraction_single_index({"i", "l"}, left_child_basis, {"l", "j", "k"}, node_transfer_tensor, {"i", "j", "k"});

        auto node_basis = ::boba::tensor_contraction_single_index({"j", "l"}, right_child_basis, {"i", "l", "k"}, left_contracted_basis, {"i", "j", "k"});

        // Convert the tensor U^{(t)} into a matrix and store the basis
        local_basis_matrices[node].emplace(::boba::unfold(node_basis, std::vector<size_t>{0, 1}, std::vector<size_t>{2}));

        // We no longer need the bases from the children, so we can clear them to save some memory
        local_basis_matrices[children[0]].reset();
        local_basis_matrices[children[1]].reset();
      }
    }

    // The root node's basis matrix represents the full tensor in folded form
    auto output = ::boba::fold(local_basis_matrices[0].value(), sizes, dim_tree.get_dims_of_node(0));

    return output;
  }

  // -------------------------------------------------------------------------------------
  // Setters and getters for HierarchicalTucker objects
  // -------------------------------------------------------------------------------------

  /**
   * @brief Sets the relative truncation tolerance for SVD operations.
   *
   * @param tolerance  Relative tolerance used during SVD-based truncation.
   */
  void set_svd_relative_tolerance(data_t tolerance) noexcept
  {
    svd_tolerance_relative = tolerance;
  }

  /**
   * @brief Sets the absolute truncation tolerance for SVD operations.
   *
   * @param tolerance  Absolute tolerance used during SVD-based truncation.
   */
  void set_svd_absolute_tolerance(data_t tolerance) noexcept
  {
    svd_tolerance_absolute = tolerance;
  }

  /**
   * @brief Sets the name of this HierarchicalTucker object.
   *
   * @param _name  Descriptive name or label for the object.
   */
  void set_name(std::string _name) noexcept
  {
    m_name = _name;
  }

  /**
   * @brief Sets the transfer tensor at a given node for this HierarchicalTucker object.
   *
   * @param node Node index -- must be a non-leaf node of the tensor.
   * @param new_transfer_tensor New transfer tensor to set at the specified node.
   */
  void set_transfer_tensor(size_t node, B_raw_type new_transfer_tensor) noexcept
  {
    boba_assert(!dim_tree.get_is_leaf_of_node(node),
                "Cannot set transfer tensor at leaf node " + std::to_string(node) + ".");

    boba_assert(transfer_tensors[node].has_value(),
                "Existing transfer tensor at node " + std::to_string(node) + " is empty.");

    boba_always_assert_equal(transfer_tensors[node]->sizes(), new_transfer_tensor.sizes(), "Incompatible transfer tensor sizes at node " + std::to_string(node) + ".");

    transfer_tensors[node] = std::move(new_transfer_tensor);

    // Since the components have changed, the tensor is no longer orthogonalized
    is_orthog = false;
  }

  /**
   * @brief Sets the basis matrix at a given node for this HierarchicalTucker object.
   *
   * @param node Node index -- must be a leaf node of the tensor.
   * @param new_basis_matrix New basis matrix to set at the specified node.
   */
  void set_basis_matrix(size_t node, U_raw_type new_basis_matrix) noexcept
  {
    boba_assert(dim_tree.get_is_leaf_of_node(node),
                "Cannot set basis matrix at non-leaf node " + std::to_string(node) + ".");

    boba_assert(basis_matrices[node].has_value(),
                "Existing basis matrix at node " + std::to_string(node) + " is empty.");

    boba_always_assert_equal(basis_matrices[node]->sizes(), new_basis_matrix.sizes(), "Incompatible basis matrix sizes at node " + std::to_string(node) + ".");

    basis_matrices[node] = std::move(new_basis_matrix);

    // Since the components have changed, the tensor is no longer orthogonalized
    is_orthog = false;
  }

  /**
   * @brief Retrieves the relative truncation tolerance used for SVD operations.
   *
   * @return Relative tolerance value.
   */
  real_data_t get_svd_relative_tolerance() const noexcept
  {
    return svd_tolerance_relative;
  }

  /**
   * @brief Retrieves the absolute truncation tolerance used for SVD operations.
   *
   * @return Absolute tolerance value.
   */
  real_data_t get_svd_absolute_tolerance() const noexcept
  {
    return svd_tolerance_absolute;
  }

  /**
   * @brief Retrieves the name of this HierarchicalTucker object.
   *
   * @return Descriptive name assigned to the object.
   */
  std::string const& name() const noexcept
  {
    return m_name;
  }

  /**
   * @brief Returns the number of nodes in this HierarchicalTucker object.
   *
   * @return Number of nodes.
   */
  constexpr size_t get_num_nodes() const noexcept
  {
    return num_nodes;
  }

  /**
   * @brief Returns the array of transfer tensors for this HierarchicalTucker object.
   *
   * @return Constant reference to the array of transfer tensors.
   */
  const ::boba::Array<B_type, num_nodes>& get_transfer_tensors() const noexcept
  {
    return transfer_tensors;
  }

  /**
   * @brief Returns the array of basis matrices for the HierarchicalTucker object.
   *
   * @return Constant reference to the array of basis matrices.
   */
  const ::boba::Array<U_type, num_nodes>& get_basis_matrices() const noexcept
  {
    return basis_matrices;
  }

  /**
   * @brief Gets the transfer tensor at a given node for this HierarchicalTucker object.
   *
   * @param node Node index -- must be a non-leaf node of the tensor.
   * @return A std::optional<B_type> containing the transfer tensor at the requested node
   */
  B_raw_type get_transfer_tensor(size_t node) const noexcept
  {
    boba_assert(!dim_tree.get_is_leaf_of_node(node),
                "Cannot get transfer tensor at leaf node " + std::to_string(node) + ".");

    boba_assert(transfer_tensors[node].has_value(),
                "Existing transfer tensor at node " + std::to_string(node) + " is empty.");

    return transfer_tensors[node].value();
  }

  /**
   * @brief Gets the basis matrix at a given node for this HierarchicalTucker object.
   *
   * @param node Node index -- must be a leaf node of the tensor.
   * @return A std::optional<U_type> containing the basis matrix at the requested node
   */
  U_raw_type get_basis_matrix(size_t node) const noexcept
  {
    boba_assert(dim_tree.get_is_leaf_of_node(node),
                "Cannot get basis matrix at non-leaf node " + std::to_string(node) + ".");

    boba_assert(basis_matrices[node].has_value(),
                "Existing basis matrix at node " + std::to_string(node) + " is empty.");

    return basis_matrices[node].value();
  }

  /**
   * @brief Retrieves the array of hierarchical ranks for all nodes.
   *
   * @return Constant reference to the array of ranks.
   */
  const ::boba::Array<size_t, num_nodes>& get_ranks() const noexcept
  {
    return ranks;
  }

  /**
   * @brief Retrieves the hierarchical rank of a specific node.
   *
   * @param i  Index of the node.
   * @return Rank associated with node @p i.
   *
   * @throws Assertion failure if @p i is out of bounds.
   */
  size_t get_ranks(size_t i) const
  {
    boba_always_assert_lt(i, num_nodes, "invalid node index");
    return ranks[i];
  }

  /**
   * @brief Returns a formatted string of all hierarchical ranks.
   *
   * The output format is: <code>( r₀, r₁, r₂, ... )</code>.
   *
   * @return String describing all ranks.
   */
  std::string get_ranks_string() const
  {
    return (" ( " + make_delimited_string(get_ranks()) + " ) ");
  }

  /**
   * @brief Retrieves the mode sizes (extents) of this HierarchicalTucker object.
   *
   * @return Constant reference to the array of mode sizes.
   */
  const ::boba::Array<size_t, dimension>& get_sizes() const noexcept
  {
    return sizes;
  }

  /**
   * @brief Retrieves the mode sizes (extents) of this HierarchicalTucker object.
   *
   * @return Bool indicating whether or not the tensor has been orthogonalized.
   */
  bool get_is_orthog() const noexcept
  {
    return is_orthog;
  }

  /**
   * @brief Retrieves the number of elements stored at a specific node.
   *
   * For leaf nodes, this is the size of the basis matrix U.
   * For internal nodes, this is the size of the transfer tensor B.
   *
   * @param i  Node index.
   * @return Number of stored elements for node @p i.
   *
   * @throws Assertion failure if @p i is out of bounds.
   */
  size_t get_number_elements(size_t i) const
  {
    boba_always_assert_lt(i, num_nodes, "invalid node index");
    const auto& is_leaf = dim_tree.get_is_leaf();

    if (is_leaf[i])
    {
      return basis_matrices[i]->size();
    }
    else
    {
      return transfer_tensors[i]->size();
    }
  }

  /**
   * @brief Computes the total number of elements stored across all nodes.
   *
   * This includes all basis matrices and transfer tensors.
   *
   * @return Total number of scalar elements in the HierarchicalTucker representation.
   */
  size_t get_number_elements() const
  {
    size_t sum_elements = 0;

    for (size_t node = 0; node < num_nodes; node++)
    {
      sum_elements += get_number_elements(node);
    }

    return sum_elements;
  }

  /**
   * @brief Computes the equivalent full tensor size.
   *
   * Returns the total number of elements that a full tensor of the same dimensions would have.
   *
   * @return Full tensor size as a double (to avoid overflow).
   */
  double get_full_size() const
  {
    double full_size = 1.0;

    for (size_t d = 0; d < dimension; d++)
    {
      full_size *= sizes[d];
    }

    return full_size;
  }

  /**
   * @brief Computes the compression rate of the HierarchicalTucker tensor.
   *
   * The compression rate is defined as:
   * \f[
   * \text{CR} = \frac{N_\text{full}}{N_\text{stored}}
   * \f]
   * where \( N_\text{full} \) is the number of entries in the full tensor and
   * \( N_\text{stored} \) is the number of stored HierarchicalTucker elements.
   *
   * The result is truncated to two decimal places.
   *
   * @return Compression rate as a float.
   */
  float get_compression_rate() const
  {
    auto cr = static_cast<double>(get_full_size()) /
              static_cast<double>(get_number_elements());
    return static_cast<float>(std::floor(cr * 100.0) / 100.0);
  }

  /**
   * @brief Retrieves the dimension tree associated with this HierarchicalTucker object.
   *
   * @return Constant reference to the dimension tree.
   */
  const ::boba::DimensionTree& get_dim_tree() const noexcept
  {
    return dim_tree;
  }

  // -------------------------------------------------------------------------------------
  // Printing utilities for HierarchicalTucker objects
  // -------------------------------------------------------------------------------------

  /**
   * @brief Prints all transfer tensors and basis matrices in the HierarchicalTucker object.
   *
   * Primarily useful for inspecting smaller tensors interactively.
   *
   * @param label Optional label printed before the output for context.
   */
  void print(const std::string& label = "") const
  {
    if (!label.empty())
    {
      std::cout << label << std::endl;
    }

    std::cout << "HierarchicalTucker transfer tensors:" << std::endl;
    for (size_t i = 0; i < transfer_tensors.size(); ++i)
    {
      std::cout << "  transfer_tensors[" << i << "]:";
      if (transfer_tensors[i].has_value())
      {
        std::cout << std::endl;
        transfer_tensors[i]->print();
      }
      else
      {
        std::cout << " {}" << std::endl;
      }
    }

    std::cout << "HierarchicalTucker basis matrices:" << std::endl;
    for (size_t i = 0; i < basis_matrices.size(); ++i)
    {
      std::cout << "  basis_matrices[" << i << "]:";
      if (basis_matrices[i].has_value())
      {
        std::cout << std::endl;
        basis_matrices[i]->print();
      }
      else
      {
        std::cout << " {}" << std::endl;
      }
    }
  }

  /**
   * @brief Prints either the transfer tensor or basis matrix at a given node.
   *
   * @param node Index of the node to print.
   * @param label Optional label printed before the dimension output.
   *
   * @throws Assertion failure if @p node is invalid.
   */
  void print_at_node(size_t node, const std::string& label = "") const
  {
    boba_always_assert_lt(node, num_nodes, "invalid node index");
    const auto& is_leaf = dim_tree.get_is_leaf();

    if (!label.empty())
    {
      std::cout << label << std::endl;
    }

    if (is_leaf[node])
    {
      std::cout << "HierarchicalTucker basis matrix at node " << node << ":" << std::endl;
      if (basis_matrices[node].has_value())
      {
        basis_matrices[node]->print();
      }
      else
      {
        std::cout << " {}" << std::endl;
      }
    }
    else
    {
      std::cout << "HierarchicalTucker transfer tensor at node " << node << ":" << std::endl;
      if (transfer_tensors[node].has_value())
      {
        transfer_tensors[node]->print();
      }
      else
      {
        std::cout << " {}" << std::endl;
      }
    }
  }

  /**
   * @brief Prints the dimensions of all transfer tensors and basis matrices.
   *
   * @param label Optional label printed before the dimension output.
   */
  void print_dimensions(const std::string& label = "") const
  {
    print_transfer_tensor_dimensions(label);
    print_basis_matrix_dimensions(label);
    check_consistency();
  }

  /**
   * @brief Prints the dimensions of all transfer tensors in the HierarchicalTucker object.
   *
   * @param label Optional label printed before the dimension output.
   */
  void print_transfer_tensor_dimensions(const std::string& label = "") const
  {
    if (!label.empty())
    {
      std::cout << label << std::endl;
    }

    std::cout << "HierarchicalTucker transfer tensors:" << std::endl;

    for (size_t i = 0; i < transfer_tensors.size(); ++i)
    {
      std::cout << "  transfer_tensors[" << i << "]: ";
      if (transfer_tensors[i].has_value())
      {
        std::cout << transfer_tensors[i]->sizes() << std::endl;
      }
      else
      {
        std::cout << "{}" << std::endl;
      }
    }
  }

  /**
   * @brief Prints the dimensions of all basis matrices in the HierarchicalTucker object.
   *
   * @param label Optional label printed before the dimension output.
   */
  void print_basis_matrix_dimensions(const std::string& label = "") const
  {
    if (!label.empty())
    {
      std::cout << label << std::endl;
    }

    std::cout << "HierarchicalTucker basis matrices:" << std::endl;

    for (size_t i = 0; i < basis_matrices.size(); ++i)
    {
      std::cout << "  basis_matrices[" << i << "]: ";
      if (basis_matrices[i].has_value())
      {
        std::cout << basis_matrices[i]->sizes() << std::endl;
      }
      else
      {
        std::cout << "{}" << std::endl;
      }
    }
  }

}; // end of HierarchicalTucker

} // namespace boba
