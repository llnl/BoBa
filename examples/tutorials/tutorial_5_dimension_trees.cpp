// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

#include "../tests/common.hpp"

int main() {

    boba::splash();
    boba::init();

    bool check = 1;

    /*
    * A dimension tree encodes the hierarchical partitioning of tensor modes
    * used in the hierarchical Tucker decomposition (HTD).
    *
    * Each node t in the tree is associated with a dimension set I_t which comprises
    * a subset of all dimensions {1, 2, ..., d}. The dimension tree satisfies the
    * following properties:
    *
    *  1. Root:
    *     The root node corresponds to the full index set:
    *         I_root = {1, 2, ..., d}.
    *
    *  2. Leaves:
    *     The leaves correspond to individual tensor modes:
    *         I_t = {i} for each leaf node t.
    *
    *  3. Internal nodes:
    *     Each internal node t has exactly two children, tl and tr, whose
    *     dimension sets partition that of their parent. In particular, the
    *     dimensions of I_t have no intersection and are a union of the dimensions
    *     owned by the left and right children.
    *
    *  4. Binary structure:
    *     The tree is strictly binary — every non-leaf node has exactly two
    *     children, and the union of all leaves’ dimension sets equals {1, ..., d}.
    *
    * This tutorial shows how to define different types of dimension trees,
    * including balanced, unbalanced, as well as custom dimension trees to support more generic tensor networks.
    */

    // Balanced dimension trees
    checkpoint();
    {
        boba_print("Checking the balanced dimension tree from a 7-tensor.");

        /*
        * A balanced dimension tree is a binary tree in which each internal node
        * divides its associated dimensions into two subsets that are as equal in
        * size as possible. In other words, for every internal node, the left and
        * right subtrees differ in height by at most one.
        *
        * Balanced dimension trees are commonly used in Hierarchical Tucker (HT)
        * decompositions because they minimize the tree depth and yield more uniform
        * computational workloads across nodes.
        *
        * Example: A balanced dimension tree for a 7-dimensional tensor
        *
        *           [0,1,2,3,4,5,6]
        *             /           \
        *         [0,1,2,3]     [4,5,6]
        *          /     \       /    \
        *       [0,1]   [2,3]  [4,5]  [6]
        *        / \     / \    / \
        *      [0] [1] [2] [3] [4] [5]
        */

        // Constructing a balanced dimension tree consists of two phases. The first
        // phase specifies the builder, which encodes the topology of the tree.
        // The following code specifies a balanced binary tree topology with 7 dimensions.
        auto balanced_builder = boba::BalancedTreeBuilder(7);

        // Next, we need to pass the builder off to the constructor of the dimension tree object
        auto balanced_tree = boba::DimensionTree(balanced_builder);

        // Note: As an alternative, we could build the tree in the following (simpler) way:
        auto alternate_balanced_tree = boba::DimensionTree(boba::BalancedTreeBuilder(7));

        // We can check that the tree has the correct number of dimensions and nodes
        pass_or_fail_bool(check, balanced_tree.get_num_dims() == 7);
        pass_or_fail_bool(check, balanced_tree.get_num_nodes() == 13);

        // To display the contents of the dimension tree, simply call the display() method (useful for debugging)
        balanced_tree.display();

        // The internals of the tree are stored in a level-by-level manner
        const std::vector<std::vector<size_t>> expected_dims = {
            {0,1,2,3,4,5,6},
            {0,1,2,3},          {4,5,6},
            {0,1},    {2,3},    {4,5}, {6},
            {0}, {1}, {2}, {3}, {4}, {5}
        };
        pass_or_fail_bool(check, balanced_tree.get_dims() == expected_dims);

        // Algorithms for HierarchicalTucker require traversals of this dimension tree so we need to be
        // able to query information about the parents and children at the nodes within a tree
        auto num_dims = balanced_tree.get_num_dims();
        auto num_nodes = balanced_tree.get_num_nodes();
        auto dims = balanced_tree.get_dims();
        auto parent = balanced_tree.get_parent();
        auto children = balanced_tree.get_children();
        auto dim2idx = balanced_tree.get_dim2idx();
        auto level = balanced_tree.get_level();
        auto post_order_nodes = balanced_tree.get_post_order_nodes();
        auto is_leaf = balanced_tree.get_is_leaf();

        boba_print(num_nodes);

        // If we want only the information at a particular node of the tree, we
        // can use the get methods with the postfix "_of_node(<node_idx>)"
        const std::vector<size_t> expected_dims_of_node_2 = {4,5,6};
        pass_or_fail_bool(check, balanced_tree.get_dims_of_node(2) == expected_dims_of_node_2);

        // We can also ask for the parent of a given node
        pass_or_fail_bool(check, balanced_tree.get_parent_of_node(3) == 1);

        // Children of a node are returned as an array of indices: 0 -> "left child" and 1 -> "right child"
        const std::array<size_t,2> expected_children_of_node_4 = {9,10};
        pass_or_fail_bool(check, balanced_tree.get_children_of_node(4) == expected_children_of_node_4);

        // An alternative to the above is to ask for left and right child indices of a particular node
        auto left_child = balanced_tree.get_left_child_of_node(4);
        auto right_child = balanced_tree.get_right_child_of_node(4);
        pass_or_fail_bool(check, (left_child == 9) && (right_child == 10));

        // Using spatial differentiation or transforms requires access to the
        // bases which are stored at the leaf nodes of the dimension tree. Therefore, we need
        // to access the map which takes us from a given dimension to the corresponding leaf node
        std::vector<size_t> expected_leaf_indices = {7,8,9,10,11,12,6};
        pass_or_fail_bool(check, dim2idx == expected_leaf_indices);

        // Iterating through the dim2idx map returns only the singleton (leaf) nodes of a tree
        // The code below shows how to iterate through the map
        for(size_t d = 0; d < num_dims; ++d)
        {
            std::cout << "Leaf node for dim " << d << ": ";
            for (auto node : dims[dim2idx[d]])
            {
                std::cout << node;
            }
            std::cout << std::endl;
        }

        // When performing arithmetric on HierarchicalTucker objects, it is necessary that they have the same dimension
        // tree. We can check if two trees are equivalent or different.

        // Self-equivalence
        pass_or_fail_bool(check, balanced_tree == balanced_tree);

        // Identical trees cannot be different
        pass_or_fail_bool(check, boba::DimensionTree(boba::BalancedTreeBuilder(7)) == balanced_tree);

        // Different trees cannot be equivalent
        pass_or_fail_bool(check, boba::DimensionTree(boba::BalancedTreeBuilder(8)) != balanced_tree);

        // It is also useful to have the ability to do a post-order traversal of the dimension tree
        // In a post-order traversal, a given node is processed only once its child nodes have been processed
        std::vector<size_t> expected_post_order_nodes = {7,8,3,9,10,4,1,11,12,5,6,2,0};
        pass_or_fail_bool(check, post_order_nodes == expected_post_order_nodes);

        // This code shows how to iterate through the nodes in a post-order traversal
        boba_print("Post-order traversal of the nodes:");
        for (const auto node : post_order_nodes)
        {
            boba_print(node);
        }

        // It is also possible to take a dimension tree and save it
        // to a file for use at a later time. This file stores minimal information
        // about the tree to this file, which can be used to recover a tree.
        boba_print("Saving the dimension tree to a file...");
        std::string test_filename = "balanced_tree7D.txt";
        balanced_tree.write_to_file(test_filename);

        // A dimension tree can be recovered from a file via the read_from_file method
        // of the DimensionTree class. We reconstruct the dimension tree using
        // minimal information about its topology. Remaining fields can be inferred from
        // this minimal information. Note that loading a tree from a file requires that
        // we verify that it is a "valid" tree structure. This is handled internally by
        // the class.
        boba_print("Loading a dimension tree from a file...");
        boba::DimensionTree loaded_balanced_tree;
        loaded_balanced_tree.read_from_file(test_filename);
        pass_or_fail_bool(check, loaded_balanced_tree == balanced_tree);

        boba_print("End of the balanced dimension tree tutorial.\n");
    }

    // Unbalanced dimension trees (tensor train topology)
    checkpoint();
    {
        boba_print("Checking the unbalanced (tensor-train) dimension tree from a 6-tensor.");

        /*
        * In contrast to a balanced dimension tree, an unbalanced dimension tree is a
        * binary tree in which each internal node partitions its dimensions in a highly
        * uneven way—typically placing a single mode in one branch while grouping all
        * remaining modes in the other. As a result, the heights of the left and right
        * subtrees differ as much as possible.
        *
        * This produces a skewed, chain-like topology that mirrors the structure of
        * tensor trains (also known as matrix product states). For this reason, such
        * trees are often said to have the topology of a tensor train.
        *
        * Example: An unbalanced (tensor train) dimension tree for a 6-dimensional tensor
        *
        *             [0,1,2,3,4,5]
        *               /       \
        *             [0]    [1,2,3,4,5]
        *                       /    \
        *                     [1]  [2,3,4,5]
        *                           /   \
        *                         [2]  [3,4,5]
        *                               /  \
        *                             [3]  [4,5]
        *                                   / \
        *                                 [4] [5]
        */

        // Constructing a tensor-train (unbalanced) dimension tree
        auto tt_builder = boba::TensorTrainTreeBuilder(6);
        auto tt_tree = boba::DimensionTree(tt_builder);

        // Make sure that the tree has the correct number of dimensions and nodes
        pass_or_fail_bool(check, tt_tree.get_num_dims() == 6);
        pass_or_fail_bool(check, tt_tree.get_num_nodes() == 11);

        // Again, the internals of the tree are stored in a level-by-level manner
        const std::vector<std::vector<size_t>> expected_dims = {
            {0,1,2,3,4,5},
            {0}, {1,2,3,4,5},
                 {1}, {2,3,4,5},
                      {2}, {3,4,5},
                           {3}, {4,5},
                                {4}, {5}
        };
        pass_or_fail_bool(check, tt_tree.get_dims() == expected_dims);

        // Display the contents of the dimension tree to verify its correctness
        tt_tree.display();

        // The methods to query the properties of the tree have the same interface as the
        // ones for the balanced tree
        auto num_dims = tt_tree.get_num_dims();
        auto num_nodes = tt_tree.get_num_nodes();
        auto dims = tt_tree.get_dims();
        auto parent = tt_tree.get_parent();
        auto children = tt_tree.get_children();
        auto dim2idx = tt_tree.get_dim2idx();
        auto level = tt_tree.get_level();
        auto post_order_nodes = tt_tree.get_post_order_nodes();
        auto is_leaf = tt_tree.get_is_leaf();

        boba_print(num_nodes);

        // Again, if we want only the information at a particular node of the tree, we
        // can use the get methods with the postfix "_of_node(<node_idx>)"
        const std::vector<size_t> expected_dims_of_node_5 = {2};
        pass_or_fail_bool(check, tt_tree.get_dims_of_node(5) == expected_dims_of_node_5);

        // Checking the parent of a node
        pass_or_fail_bool(check, tt_tree.get_parent_of_node(4) == 2);

        // We can again ask for left and right child indices of a particular node
        const std::array<size_t,2> expected_children_of_node_6 = {7,8};
        pass_or_fail_bool(check, tt_tree.get_children_of_node(6) == expected_children_of_node_6);

        // Check the dim2idx map
        std::vector<size_t> expected_leaf_indices = {1,3,5,7,9,10};
        pass_or_fail_bool(check, dim2idx == expected_leaf_indices);

        // We can iterate through the dim2idx map to print each leaf node
        for(size_t d = 0; d < num_dims; ++d)
        {
            std::cout << "Leaf node for dim " << d << ": ";
            for (auto node : dims[dim2idx[d]])
            {
                std::cout << node;
            }
            std::cout << std::endl;
        }

        // Checking equivalence of trees

        // Self-equivalence
        pass_or_fail_bool(check, tt_tree == tt_tree);

        // Identical trees constructed separately must be equivalent
        pass_or_fail_bool(check, boba::DimensionTree(boba::TensorTrainTreeBuilder(6)) == tt_tree);

        // Trees with different topologies must not be equivalent
        // Here we demonstrate by comparing with a balanced dimension tree
        pass_or_fail_bool(check, boba::DimensionTree(boba::BalancedTreeBuilder(6)) != tt_tree);

        // We can also extract and iterate through the nodes sorted as a post-order traversal
        std::vector<size_t> expected_post_order_nodes = {1,3,5,7,9,10,8,6,4,2,0};
        pass_or_fail_bool(check, post_order_nodes == expected_post_order_nodes);

        boba_print("Post-order traversal of the nodes:");
        for (const auto node : post_order_nodes)
        {
            boba_print(node);
        }

        // Save the contents of the tree to a file
        boba_print("Saving a dimension tree to a file...");
        std::string test_filename = "tt_tree6D.txt";
        tt_tree.write_to_file(test_filename);

        // Use the information from a file to reconstruct the entire tree
        boba_print("Loading a dimension tree from a file...");
        boba::DimensionTree loaded_tt_tree;
        loaded_tt_tree.read_from_file(test_filename);
        pass_or_fail_bool(check, loaded_tt_tree == tt_tree);

        boba_print("End of the unbalanced (tensor-train) dimension tree tutorial.\n");
    }

    // Custom dimension trees
    checkpoint();
    {
        boba_print("Custom dimension tree from a 5-tensor.");

        /*
        * To support more general dimension trees that are neither maximally balanced
        * nor fully unbalanced, we provide a CustomTreeBuilder class. This allows the
        * construction of arbitrary tree topologies suited to specific tensor
        * structures or problem-specific needs.
        *
        * Example: A custom dimension tree for a 5-dimensional tensor
        *
        *                 [0,1,2,3,4]
        *                   /      \
        *                 [0]    [1,2,3,4]
        *                        /       \
        *                    [1,2]       [3,4]
        *                     / \         / \
        *                   [1] [2]     [3] [4]
        *
        * In this example, the right subtree rooted at {1,2,3,4} is balanced,
        * splitting evenly into {1,2} and {3,4}, while the overall tree is unbalanced
        * because the root separates a single mode {0} from the remaining dimensions.
        * Consequently, this topology combines characteristics of both balanced and
        * unbalanced (tensor train–like) trees.
        */

        // To build a dimension tree with a customized topology, we first need to
        // create a custom builder (with the number of dimensions). Note that the constructor
        // initializes the root node of the tree
        auto custom_builder = boba::CustomTreeBuilder(5);

        // Next, we can append nodes to the builder following a level by level traversal
        // In this example, we make the local topology at the root node look like a tensor train
        // tree, and then use a balanced binary tree on the remaining dimensions.
        //
        // The arguments to push represent (respectively) the dimensions owned by the new node (node_dims),
        // the index of the parent node (parent), and an integer argument (side). Here side = 0 or side = 1
        // means that the current node is the left or right child of the node (parent).
        custom_builder.push_node({0}, 0, 0);        // left child of root
        custom_builder.push_node({1,2,3,4}, 0, 1);  // right child of root
        custom_builder.push_node({1,2}, 2, 0);      // left child of node 2
        custom_builder.push_node({3,4}, 2, 1);      // right child of node 2
        custom_builder.push_node({1}, 3, 0);        // left child of node 3
        custom_builder.push_node({2}, 3, 1);        // right child of node 3
        custom_builder.push_node({3}, 4, 0);        // left child of node 4
        custom_builder.push_node({4}, 4, 1);        // right child of node 4

        // Note: it is also valid to pass dims and children directly to the constructor
        // but the above approach is recommended

        // Now that we have specified the connectivity of the tree to the builder, let's
        // make a DimensionTree object
        auto custom_tree = boba::DimensionTree(custom_builder);

        // Check that the number of dimensions and nodes are correct
        pass_or_fail_bool(check, custom_tree.get_num_dims() == 5);
        pass_or_fail_bool(check, custom_tree.get_num_nodes() == 9);

        const std::vector<std::vector<size_t>> expected_dims = {
            {0,1,2,3,4},
            {0}, {1,2,3,4},
                 {1,2},    {3,4},
                 {1}, {2}, {3}, {4}
        };
        pass_or_fail_bool(check, custom_tree.get_dims() == expected_dims);

        // The same methods shown for the balanced and unbalanced trees can also be called...

        // Display the tree contents (useful for debugging)
        custom_tree.display();

        // Save the contents of the tree to a file
        std::string test_filename = "custom_tree5D.txt";
        custom_tree.write_to_file(test_filename);

        // Use the information from a file to reconstruct the entire tree
        boba::DimensionTree loaded_custom_tree;
        loaded_custom_tree.read_from_file(test_filename);
        pass_or_fail_bool(check, loaded_custom_tree == custom_tree);

        boba_print("End of the custom dimension tree tutorial.\n");
    }

    boba::finalize();
    return final_check(check);
}
// clang-format on
