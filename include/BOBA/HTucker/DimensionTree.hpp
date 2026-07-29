// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "BOBA/boba.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iterator>
#include <queue>
#include <stack>
#include <string>
#include <vector>

namespace boba
{

/**
 * \class DimensionTree
 * \brief Represents the binary dimension tree used in hierarchical Tucker decompositions.
 *
 * A DimensionTree encodes the hierarchical organization of tensor modes into
 * a binary tree structure, where each internal node represents a subset of
 * dimensions and each leaf corresponds to a single tensor mode. The tree
 * topology determines how multi-dimensional data is factorized and combined in
 * hierarchical low-rank tensor formats.
 */
class DimensionTree
{

public:
  /// \brief default constructor
  DimensionTree() = default;

  /// \brief copy constructor
  DimensionTree(DimensionTree const&) = default;

  /// \brief move constructor
  DimensionTree(DimensionTree&&) = default;

  /// \brief copy assignment operator
  DimensionTree& operator=(DimensionTree const&) = default;

  /// \brief move assignment operator
  DimensionTree& operator=(DimensionTree&&) = default;

  /// \brief destructor
  ~DimensionTree() = default;

  /**
   * \brief Construct a DimensionTree from a TreeBuilder.
   *
   * The constructor transfers the topology information from the builder into
   * the new tree, then derives auxiliary mappings such as:
   * - parent relationships between nodes,
   * - the leaf-node lookup table,
   * - node levels and traversal orders,
   * - leaf identification.
   *
   * The constructor also sorts each node's dimension set and validates the
   * resulting tree.
   *
   * \param[in] builder tree builder defining the topology
   */
  explicit DimensionTree(TreeBuilder builder)
  {
    // Transfer topology atttributes from the builder
    // Note, if the builder is supplied as an lvalue, only one copy is performed.
    // Otherwise, for rvalues, the constructor uses move semantics.
    num_dims = builder.num_dims;
    num_nodes = builder.num_nodes;
    dims = std::move(builder.dims);
    children = std::move(builder.children);

    // Build the remaining attributes of the tree
    boba_always_assert_gt(num_dims, 0, "The number of dimensions should be > 0.");
    build_parent();
    build_dim2idx();
    build_node_levels();
    build_level2nodes();
    build_post_order_nodes();
    build_is_leaf();
    sort_dims();
    boba_always_assert(is_valid(), "Tree is not valid.");
  }

  /**
   * \brief Check whether two dimension trees are identical.
   *
   * Two trees are equal if the dimensions and children at each node are the
   * same.
   *
   * \param[in] other_tree tree to compare against
   * \return true if both trees have the same node dimensions and children
   */
  bool operator==(const DimensionTree& other_tree) const
  {
    // Two trees are equal if the dimensions and children at each node
    // are the same
    bool same_children = (children == other_tree.children);
    bool same_dims = (dims == other_tree.dims);
    return same_children && same_dims;
  }

  /**
   * \brief Check whether two dimension trees are different.
   *
   * \param[in] other_tree tree to compare against
   * \return true if the trees differ in dimensions or child relationships
   */
  bool operator!=(const DimensionTree& other_tree) const
  {
    return !(*this == other_tree);
  }

  /**
   * \brief
   * Verifies the structural integrity of the binary dimension tree.
   *
   * \details
   * This routine performs a breadth-first traversal from the root node to ensure that
   * the tree satisfies all validity requirements for Hierarchical Tucker dimension trees:
   *  - The tree is connected and acyclic (every node is visited exactly once).
   *  - The root has no parent, while all other nodes have exactly one parent.
   *  - Each internal node has either zero or two children (binary structure).
   *  - The dimension sets of sibling nodes are disjoint.
   *  - The parent’s dimension set equals the union of its children’s dimension sets.
   *
   * If any of these checks fail, the function prints a diagnostic message and returns false.
   */
  bool is_valid() const
  {
    // Traverse the tree from root to leaves (BFS) and ensure that each node is visited once
    std::queue<size_t> q;
    std::vector<size_t> visits(num_nodes, 0);

    // The root is always the first node
    q.push(0);
    boba_always_assert(parent[0] == INVALID_IDX, "The root node should have no parent.");

    // Temporary storage used to check the dimension sets in the tree
    std::vector<size_t> tmp_storage;

    while (!q.empty())
    {
      // Remove the current node from the queue (FIFO)
      const size_t current_node = q.front();
      q.pop();

      // All nodes must be visited (only) once during the traversal. Otherwise, the dimension
      // tree contains at least one cycle
      visits[current_node]++;

      if (visits[current_node] > 1)
      {
        std::cerr << "Cycle detected at node " << current_node << std::endl;
        return false;
      }

      // All nodes except for the root must have a parent
      if ((current_node > 0) && (parent[current_node] == INVALID_IDX))
      {
        std::cerr << "Node " << current_node << " does not have a parent." << std::endl;
        return false;
      }

      // Each node must have either no children or two children
      const auto [left_child, right_child] = children[current_node];

      if ((left_child != INVALID_IDX) && (right_child != INVALID_IDX))
      {
        q.push(left_child);
        q.push(right_child);
      }
      else if ((left_child == INVALID_IDX) && (right_child == INVALID_IDX))
      {
        continue;
      }
      else
      {
        std::cerr << "Node " << current_node << " must have 0 or 2 children." << std::endl;
        return false;
      }

      // Intersections among the dimensions owned by the children of a node must be disjoint
      tmp_storage.clear();
      std::set_intersection(dims[left_child].begin(), dims[left_child].end(), dims[right_child].begin(), dims[right_child].end(), std::back_inserter(tmp_storage));

      if (!tmp_storage.empty())
      {
        std::cerr << "Child nodes must be disjoint at node " << current_node << std::endl;
        return false;
      }

      // The union of the dimensions owned by the children are equivalent to the parent
      tmp_storage.clear();
      std::set_union(dims[left_child].begin(), dims[left_child].end(), dims[right_child].begin(), dims[right_child].end(), std::back_inserter(tmp_storage));

      if (tmp_storage != dims[current_node])
      {
        std::cerr << "Nestedness property does not hold at node " << current_node << std::endl;
        return false;
      }
    }

    // The tree must not contain any orphan nodes
    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (visits[node] != 1)
      {
        std::cerr << "Node " << node << " was not visited exactly once." << std::endl;
        return false;
      }
    }

    return true;
  }

  /**
   * \brief
   * Returns the total number of nodes in the dimension tree.
   */
  size_t get_num_nodes() const
  {
    return num_nodes;
  }

  /**
   * \brief
   * Returns the number of dimensions represented in the tree.
   */
  size_t get_num_dims() const
  {
    return num_dims;
  }

  /**
   * \brief
   * Returns the list of dimension indices owned by each node.
   *
   * Each entry dims[i] corresponds to the set of tensor modes grouped
   * at node i in the tree.
   */
  const std::vector<std::vector<size_t>>& get_dims() const
  {
    return dims;
  }

  /**
   * \brief
   * Returns the parent index of each node in the tree.
   *
   * The root node always has parent[i] == INVALID_IDX.
   */
  const std::vector<size_t>& get_parent() const
  {
    return parent;
  }

  /**
   * \brief
   * Returns the array of left/right child indices for each node.
   *
   * For a leaf node, both entries are INVALID_IDX.
   */
  const std::vector<std::array<size_t, 2>>& get_children() const
  {
    return children;
  }

  /**
   * \brief
   * Returns the dimension to leaf node map for the tree.
   */
  const std::vector<size_t>& get_dim2idx() const
  {
    return dim2idx;
  }

  /**
   * \brief
   * Returns the level (distance from the root) for each node.
   */
  const std::vector<size_t>& get_level() const
  {
    return level;
  }

  /**
   * \brief
   * Returns the number of levels of the dimension tree.
   */
  size_t get_depth() const
  {
    return level.size();
  }

  /**
   * \brief
   * Returns the mapping from each tree level to the nodes contained on that level.
   */
  const std::vector<std::vector<size_t>>& get_level2nodes() const
  {
    return level2nodes;
  }

  /**
   * \brief
   * Returns the node indices in post-order traversal order.
   */
  const std::vector<size_t>& get_post_order_nodes() const
  {
    return post_order_nodes;
  }

  /**
   * \brief
   * Returns a boolean flag for each node indicating whether it is a leaf.
   */
  const std::vector<bool>& get_is_leaf() const
  {
    return is_leaf;
  }

  /**
   * \brief
   * Returns the set of dimensions owned by a specified node.
   */
  const std::vector<size_t>& get_dims_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return dims[node];
  }

  /**
   * \brief
   * Returns the parent index of a specified node.
   */
  size_t get_parent_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return parent[node];
  }

  /**
   * \brief
   * Returns the indices of both children for a specified node.
   */
  const std::array<size_t, 2>& get_children_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return children[node];
  }

  /**
   * \brief
   * Returns the left child index for a specified node.
   */
  size_t get_left_child_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return children[node][0];
  }

  /**
   * \brief
   * Returns the right child index for a specified node.
   */
  size_t get_right_child_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return children[node][1];
  }

  /**
   * \brief
   * Returns true if the given node is the left child of its parent.
   */
  bool is_left_child(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return children[parent[node]][0] == node;
  }

  /**
   * \brief
   * Returns true if the given node is the right child of its parent.
   */
  bool is_right_child(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return children[parent[node]][1] == node;
  }

  /**
   * \brief
   * Returns the leaf-node index corresponding to a given mode.
   */
  size_t get_dim2idx_of_dim(size_t d) const
  {
    boba_assert_lt(d, num_dims, "Invalid dimension.");
    return dim2idx[d];
  }

  /**
   * \brief
   * Returns the level (distance from root) of a specified node.
   */
  size_t get_level_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return level[node];
  }

  /**
   * \brief
   * Returns the list of nodes located on a specified tree level.
   */
  const std::vector<size_t>& get_nodes_on_level(size_t lvl) const
  {
    boba_assert_lt(lvl, level2nodes.size(), "Invalid level.");
    return level2nodes[lvl];
  }

  /**
   * \brief
   * Returns true if the specified node is a leaf node.
   */
  bool get_is_leaf_of_node(size_t node) const
  {
    boba_assert_lt(node, num_nodes, "Invalid node index.");
    return is_leaf[node];
  }

  /**
   * \brief
   * Prints the dimension sets associated with each node.
   */
  void display_dims() const
  {
    std::cout << "dims = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << "[";
      for (size_t j = 0; j < dims[i].size(); ++j)
      {
        std::cout << dims[i][j];
        if (j + 1 < dims[i].size())
        {
          std::cout << ",";
        }
      }
      std::cout << "]";
      if (i + 1 < dims.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the parent index of each node.
   */
  void display_parent() const
  {
    std::cout << "parent = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << parent[i];
      if (i + 1 < parent.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the left and right child indices of each node.
   */
  void display_children() const
  {
    std::cout << "children = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << "[";
      for (size_t j = 0; j < children[i].size(); ++j)
      {
        std::cout << children[i][j];
        if (j + 1 < children[i].size())
        {
          std::cout << ",";
        }
      }
      std::cout << "]";
      if (i + 1 < children.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the mapping from modes to leaf-node indices.
   */
  void display_dim2idx() const
  {
    std::cout << "dim2idx = [";
    for (size_t d = 0; d < num_dims; ++d)
    {
      std::cout << dim2idx[d];
      if (d + 1 < dim2idx.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the level index of each node.
   */
  void display_level() const
  {
    std::cout << "level = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << level[i];
      if (i + 1 < level.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the nodes organized by tree level.
   */
  void display_level2node() const
  {
    std::cout << "level2node = [";
    for (size_t i = 0; i < level2nodes.size(); ++i)
    {
      std::cout << "[";
      for (size_t j = 0; j < level2nodes[i].size(); ++j)
      {
        std::cout << level2nodes[i][j];
        if (j + 1 < level2nodes[i].size())
        {
          std::cout << ",";
        }
      }
      std::cout << "]";
      if (i + 1 < level2nodes.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the node indices in post-order traversal order.
   */
  void display_post_order_nodes() const
  {
    std::cout << "post_order_nodes = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << post_order_nodes[i];
      if (i + 1 < post_order_nodes.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints the leaf-node indicator array.
   */
  void display_is_leaf() const
  {
    std::cout << "is_leaf = [";
    for (size_t i = 0; i < num_nodes; ++i)
    {
      std::cout << is_leaf[i];
      if (i + 1 < is_leaf.size())
      {
        std::cout << ",";
      }
    }
    std::cout << "]" << std::endl;
  }

  /**
   * \brief
   * Prints all key attributes of the dimension tree.
   */
  void display() const
  {

    boba_print("Dimension Tree contents:");
    boba_print(num_dims);
    boba_print(num_nodes);

    display_dims();
    display_parent();
    display_children();
    display_dim2idx();
    display_level();
    display_level2node();
    display_post_order_nodes();
    display_is_leaf();
  }

  /**
   * \brief
   * Writes the minimal information needed to reconstruct a dimension tree to a file.
   *
   * \param filename Path to the output file.
   *
   * The file stores the number of dimensions and nodes, followed by
   * the dimension sets and the left/right child indices of each node.
   */
  void write_to_file(std::string_view filename) const
  {
    const std::string print_filename(filename);
    std::ofstream ofs(print_filename);

    if (!ofs)
    {
      throw std::runtime_error("Could not open file for writing.");
    }

    // First line represents the number of dimensions and the second is the number of nodes
    ofs << num_dims << std::endl;
    ofs << num_nodes << std::endl;

    // For the dims container, we first iterate through all the nodes.
    // Then for each node, we first write the number of dimensions at
    // this node, followed by its elements
    for (const auto& node_dims : dims)
    {
      ofs << node_dims.size();
      for (auto d : node_dims)
      {
        ofs << " " << d;
      }
      ofs << std::endl;
    }

    // For each node, write the left and right child indices
    for (const auto& child : children)
    {
      ofs << child[0] << " " << child[1] << std::endl;
    }
  }

  /**
   * \brief
   * Reads a DimensionTree from a file.
   *
   * The file must contain the minimal topology information needed to
   * reconstruct the tree, including node dimension sets and child indices.
   * After loading, this function rebuilds the parent map, traversal orders,
   * and auxiliary structures before validating the tree.
   *
   * \param filename Path to the output file.
   */
  void read_from_file(std::string_view filename)
  {
    const std::string print_filename = std::string(filename);
    std::ifstream ifs(print_filename);

    if (!ifs)
    {
      throw std::runtime_error("Could not open file for reading.");
    }

    // First line represents the number of dimensions, and the second is the number of nodes
    ifs >> num_dims >> num_nodes;

    if (num_dims == 0 || num_nodes == 0)
    {
      throw std::runtime_error("Invalid dimension tree: num_dims and num_nodes must be > 0.");
    }

    // For each node, read the number of dimensions in its dimension set,
    // followed by the dimension indices themselves.
    dims.resize(num_nodes);

    for (auto& node_dims : dims)
    {
      size_t sz;
      ifs >> sz;

      if (ifs.fail())
      {
        throw std::runtime_error("Failed to read node dim size.");
      }

      node_dims.resize(sz);

      for (size_t i = 0; i < sz; ++i)
      {
        ifs >> node_dims[i];

        if (ifs.fail())
        {
          throw std::runtime_error("Failed to read node dim value.");
        }
      }
    }

    // For each node, read the left and right child indices.
    children.resize(num_nodes);

    for (auto& child : children)
    {
      ifs >> child[0] >> child[1];

      if (ifs.fail())
      {
        throw std::runtime_error("Failed to read child indices.");
      }
    }

    // Rebuild derived tree data
    build_parent();
    build_dim2idx();
    build_node_levels();
    build_level2nodes();
    build_post_order_nodes();
    build_is_leaf();

    sort_dims();

    boba_always_assert(is_valid(), "Tree is not valid.");
  }

private:
  size_t num_dims;
  size_t num_nodes;

  // Since size_t is unsigned, we define a new invalid index
  static constexpr size_t INVALID_IDX = static_cast<size_t>(-1);

  // Containiner for the indices of the parent nodes s.t.
  // parent[i] gives the logical location of the parent of node i
  std::vector<size_t> parent;

  // Container for the indices of the child nodes s.t.
  // children[i] = [left_child, right_child]
  std::vector<std::array<size_t, 2>> children;

  // dims is a vector of vectors s.t.
  // dims[i] is the vector of dimensions owned by node i
  std::vector<std::vector<size_t>> dims;

  // Container for the leaf node indices s.t.
  // dim2idx[d] stores the logical location of dimension d among the leaf nodes
  std::vector<size_t> dim2idx;

  // Container for the levels of each node in the tree s.t.
  // level[i] represents the distance of node i from the root node.
  std::vector<size_t> level;

  // Container for the levels which identifies which nodes belong to a given level
  std::vector<std::vector<size_t>> level2nodes;

  // Container which holds a post-order traversal of the nodes
  std::vector<size_t> post_order_nodes;

  // Container which levels of each node in the tree s.t.
  // is_leaf[i] represents the distance of node i from the root node.
  std::vector<bool> is_leaf;

  /**
   * \brief
   * Infers the parents of each node using the available dims and children.
   */
  void build_parent()
  {
    // All nodes initially point to invalid indices
    parent.resize(num_nodes, INVALID_IDX);

    // The current node is the parent of both the left child and right child
    // Note: This is only true for internal (non-leaf) nodes.
    for (size_t node = 0; node < num_nodes; ++node)
    {
      auto [left_child, right_child] = children[node];

      if ((left_child != INVALID_IDX) && (right_child != INVALID_IDX))
      {
        parent[left_child] = node;
        parent[right_child] = node;
      }
    }
  }

  /**
   * \brief
   * Calculates the distance of each node in the tree from the root node.
   */
  void build_node_levels()
  {
    level.resize(num_nodes, 0);

    // The level of the current node is one more than the level of the parent
    for (size_t node = 1; node < num_nodes; ++node)
    {
      level[node] = level[parent[node]] + 1;
    }
  }

  /**
   * \brief
   * Builds a map that takes in a level and returns the nodes on a given level.
   */
  void build_level2nodes()
  {
    // First, determine how many levels there are in the tree
    size_t max_level = 0;
    for (size_t node_level : level)
    {
      max_level = std::max(max_level, node_level);
    }

    // Now map the nodes to their corresponding levels
    level2nodes.resize(max_level + 1);
    for (size_t node = 0; node < num_nodes; ++node)
    {
      level2nodes[level[node]].push_back(node);
    }
  }

  /**
   * \brief
   * Builds the dimension to leaf node index map for a tree.
   */
  void build_dim2idx()
  {
    dim2idx.resize(num_dims);

    // Look for singleton nodes in the dims array
    for (size_t node = 0; node < num_nodes; ++node)
    {
      if (dims[node].size() == 1)
      {
        dim2idx[dims[node][0]] = node;
      }
    }
  }

  /**
   * \brief
   * Builds the array which holds a post-order traversal of the nodes.
   */
  void build_post_order_nodes()
  {
    // Clear the pre-existing traversal order
    post_order_nodes.clear();

    boba_always_assert(parent[0] == INVALID_IDX, "The root node should have no parent.");

    // Maintain a stack of pairs (node, visited)
    std::stack<std::pair<size_t, bool>> node_stack;

    // Push the root node onto the stack
    node_stack.emplace(std::make_pair(0, false));

    while (!node_stack.empty())
    {
      // Remove the current node from the stack (LIFO)
      auto [node, visited] = node_stack.top();
      node_stack.pop();

      if (visited)
      {
        post_order_nodes.push_back(node);
      }
      else
      {
        // Mark the current node as visited and add it to the
        // stack for processing after its children
        node_stack.emplace(std::make_pair(node, true));

        // Push the right child, then the left child to
        // ensure that the left child is processed first
        const auto [left_child, right_child] = children[node];

        if ((left_child != INVALID_IDX) && (right_child != INVALID_IDX))
        {
          node_stack.emplace(std::make_pair(right_child, false));
          node_stack.emplace(std::make_pair(left_child, false));
        }
      }
    }
  }

  /**
   * \brief
   * Builds the array which identifies each node in the tree as a leaf node.
   */
  void build_is_leaf()
  {
    is_leaf.resize(num_nodes, false);

    for (auto idx : dim2idx)
    {
      is_leaf[idx] = true;
    }
  }

  /**
   * \brief
   * Sorts the dimensions owned by each node in the tree in ascending order.
   */
  void sort_dims()
  {
    for (auto& dims_vector : dims)
    {
      std::sort(dims_vector.begin(), dims_vector.end());
    }
  }
};

} // namespace boba
