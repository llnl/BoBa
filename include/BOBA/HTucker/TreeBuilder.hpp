// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// These includes should be combined with
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

namespace boba
{

// Forward declaration
class DimensionTree;

/*
 * \class TreeBuilder
 * \brief Base helper class for constructing binary dimension trees.
 *
 * TreeBuilder provides shared storage and logic used by the specialized tree
 * builders (balanced, tensor-train, and custom). It maintains the number of
 * dimensions, the number of nodes, the dimension sets at each node, and the
 * corresponding child indices. Derived classes define how the tree topology is
 * generated.
 */
class TreeBuilder
{

protected:
  size_t num_dims;
  size_t num_nodes;

  // Since size_t is unsigned, we define a new invalid index.
  // This is marked static as it is used with boba::DimensionTree.
  static constexpr size_t INVALID_IDX = ::boba::highest_value<size_t>();

  // dims[i] gives the vector of physical dimensions owned by node i.
  std::vector<std::vector<size_t>> dims;

  // children[i] = [left_child, right_child] for node i.
  std::vector<std::array<size_t, 2>> children;

  explicit TreeBuilder(size_t ndims)
      : num_dims(ndims)
  {
  }

  // Grant access of the protected class attributes to DimensionTree
  friend class DimensionTree;

public:
  /// \brief default constructor
  TreeBuilder() = default;

  /// \brief copy constructor
  TreeBuilder(TreeBuilder const&) = default;

  /// \brief move constructor
  TreeBuilder(TreeBuilder&&) = default;

  /// \brief copy assignment operator
  TreeBuilder& operator=(TreeBuilder const&) = default;

  /// \brief move assignment operator
  TreeBuilder& operator=(TreeBuilder&&) = default;

  /// \brief destructor
  virtual ~TreeBuilder() = default;
};

/*
 * \class BalancedTreeBuilder
 * \brief
 * Builder for balanced binary dimension trees.
 *
 * \details
 * A balanced tree divides each node’s dimension set into two subsets
 * of nearly equal size, producing a tree of minimal depth.
 */
class BalancedTreeBuilder : public TreeBuilder
{

public:
  /**
   * \brief Construct a balanced dimension tree builder.
   *
   * The root is initialized with all dimensions and each node is recursively
   * partitioned into two subsets of nearly equal size, producing a tree of
   * minimal depth.
   *
   * \param[in] _num_dims number of tensor dimensions
   */
  explicit BalancedTreeBuilder(size_t _num_dims)
      : TreeBuilder(_num_dims)
  {
    num_nodes = 2 * num_dims - 1;
    dims.resize(num_nodes);
    children.resize(num_nodes);

    // The root node contains all the dimensions
    dims[0].resize(num_dims);

    for (size_t i = 0; i < num_dims; ++i)
    {
      dims[0][i] = i;
    }

    std::queue<size_t> q;
    q.push(0);

    size_t next_free_node = 1;

    // Now build the parent/child relationships between nodes of the tree
    // following a breadth-first traversal
    while (!q.empty())
    {
      size_t node = q.front();
      q.pop();

      const auto& current_dims = dims[node];

      if (current_dims.size() > 1)
      {
        // The current node's children are built from a partition of the
        // dimensions owned by the current node
        size_t left_child = next_free_node++;
        size_t right_child = next_free_node++;

        children[node] = {left_child, right_child};

        // Partition the dimension set at the midpoint (ceiling)
        size_t split = (current_dims.size() + 1) / 2;
        auto start = current_dims.begin();
        auto end = current_dims.end();

        dims[left_child].assign(start, start + static_cast<std::ptrdiff_t>(split));
        dims[right_child].assign(start + static_cast<std::ptrdiff_t>(split), end);

        // Append the children to the queue
        q.push(left_child);
        q.push(right_child);
      }
      else
      {
        // The node is a leaf node so it contains no children
        children[node] = {INVALID_IDX, INVALID_IDX};
      }
    }
  }
};

/*
 * \class TensorTrainTreeBuilder
 * \brief
 * Builder for maximally unbalanced binary trees (tensor-train topology).
 *
 * \details
 * This builder constructs a chain-like tree where each internal node
 * separates one mode as its left child and groups all remaining modes
 * into its right child. The resulting topology mirrors the structure
 * of tensor trains or matrix product states.
 */
class TensorTrainTreeBuilder : public TreeBuilder
{

public:
  /**
   * \brief Construct a tensor-train dimension tree builder.
   *
   * The tree is built as a maximally skewed binary tree where each split
   * isolates the left-most dimension as a singleton node and assigns the
   * remaining dimensions to the right subtree.
   *
   * \param[in] _num_dims number of tensor dimensions
   */
  explicit TensorTrainTreeBuilder(size_t _num_dims)
      : TreeBuilder(_num_dims)
  {
    num_nodes = 2 * num_dims - 1;
    dims.resize(num_nodes);
    children.resize(num_nodes);

    // The root node contains all the dimensions
    dims[0].resize(num_dims);

    for (size_t i = 0; i < num_dims; ++i)
    {
      dims[0][i] = i;
    }

    std::queue<size_t> q;
    q.push(0);

    size_t next_free_node = 1;

    // Now build the parent/child relationships between nodes of the tree
    // following a breadth-first traversal
    while (!q.empty())
    {
      size_t node = q.front();
      q.pop();

      const auto& current_dims = dims[node];

      if (current_dims.size() > 1)
      {
        // The current node's children are built from a partition of the
        // dimensions owned by the current node
        size_t left_child = next_free_node++;
        size_t right_child = next_free_node++;

        children[node] = {left_child, right_child};

        // Partition the dimension by splitting at the left-most dimension in each node
        // That is, the left child is always a singleton while the right node receives the
        // remaining dimensions
        size_t split = 1;
        auto start = current_dims.begin();
        auto end = current_dims.end();

        dims[left_child].assign(start, start + static_cast<std::ptrdiff_t>(split));
        dims[right_child].assign(start + static_cast<std::ptrdiff_t>(split), end);

        // Append the children to the queue
        q.push(left_child);
        q.push(right_child);
      }
      else
      {
        // The node is a leaf node so it contains no children
        children[node] = {INVALID_IDX, INVALID_IDX};
      }
    }
  }
};

/*
 * \class CustomTreeBuilder
 * \brief
 * Builder for user-defined binary dimension trees.
 *
 * \details
 * This builder enables construction of arbitrary tree topologies by
 * specifying node connectivity directly or by incrementally appending
 * nodes. It supports hybrid structures that mix balanced and unbalanced
 * subtrees to support more general tensor networks.
 */
class CustomTreeBuilder : public TreeBuilder
{

public:
  /**
   * \brief
   * Constructs a custom tree from explicitly provided dimension and child arrays.
   *
   * \param _dims      Vector of dimension sets for each node.
   * \param _children  Array of left/right child indices for each node.
   *
   * The root node must appear first in the input arrays. This constructor
   * directly defines the full topology of the tree.
   */
  explicit CustomTreeBuilder(std::vector<std::vector<size_t>> _dims, std::vector<std::array<size_t, 2>> _children)
      : TreeBuilder(_dims[0].size())
  {
    num_nodes = dims.size();
    dims = std::move(_dims);
    children = std::move(_children);
  }

  // Class constructor which should be used to build a tree node by node
  explicit CustomTreeBuilder(size_t _num_dims)
      : TreeBuilder(_num_dims)
  {
    // The root must contain all dimensions
    std::vector<size_t> root_dims(num_dims);
    std::iota(root_dims.begin(), root_dims.end(), 0);
    dims.push_back(std::move(root_dims));
    children.push_back({INVALID_IDX, INVALID_IDX});
    num_nodes = 1;
  }

  // Append a node specifying its dimensions, parent index, and side (0=left, 1=right)
  void push_node(std::vector<size_t> node_dims, size_t parent, size_t side)
  {
    // Check the input for any errors
    if (parent >= dims.size())
    {
      throw std::out_of_range("Parent index out of range");
    }
    if (side != 0 && side != 1)
    {
      throw std::invalid_argument("Side must be 0 (left) or 1 (right)");
    }
    if (children[parent][side] != INVALID_IDX)
    {
      throw std::runtime_error("Child slot already occupied");
    }

    // Add the node and mark its children as invalid
    dims.push_back(std::move(node_dims));
    children.push_back({INVALID_IDX, INVALID_IDX});

    // The current node is the child of node "parent"
    size_t node = dims.size() - 1;
    children[parent][side] = node;
    num_nodes = dims.size();
  }
};

} // namespace boba
