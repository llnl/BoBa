// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off

#include "../tests/common.hpp"

/*
  Demonstrates some basics regarding hierarchical Tucker decompositions
  Here we assume you have been introduced to the basics of Tucker decompositions,
  the concept of dimension trees (see tutorial 5 for more on dimension trees),
  and the hierarchical Tucker format.
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main() {

    boba::splash();
    boba::init();

    bool check = true;

    checkpoint();
    {
        boba_print("Creating and initializing HierarchicalTucker objects.");

        // First create a dimension tree for the 3-tensor (see tutorial 5)
        constexpr size_t num_dims = 3;
        auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(num_dims));

        // Each mode of the tensor has 10 elements
        auto sizes = boba::filled_array<num_dims>(10_z);

        // The simplest way to create an HierarchicalTucker object is to specify the mode sizes and the dimension tree
        // By default, we create an HierarchicalTucker object filled with zeros, which has a hierarchical rank of 1 at all nodes
        // Once created, we can fill the HierarchicalTucker object with constants, or even make a random (low-rank) tensor.
        // The code below shows how to do this.

        // This creates an HierarchicalTucker object of size 10^{num_dims} filled with zeros
        boba::HierarchicalTucker<num_dims, space, double> A_ht(sizes, dim_tree);

        // Let's make an HierarchicalTucker object filled with constants, e.g., ones
        A_ht.fill_with(1.0);

        // Note, we can also create an random HierarchicalTucker object containing random numbers. The class method fill_with_random()
        // makes an HierarchicalTucker object filled with uniform random numbers in (0,1) with uniform random ranks [1,10].
        // A_ht.fill_with_random();

        // In more advanced use cases, an HierarchicalTucker tensor can also be constructed
        // directly from user-supplied basis matrices and transfer tensors.
        // This allows you to build an HierarchicalTucker representation manually (for example,
        // from external data or from your own algorithms), as long as the provided
        // components are structurally consistent with the dimension tree.
        //
        // The basic representation format is as follows: Each node in the dimension tree stores EITHER
        //   - a basis matrix   (leaf nodes), or
        //   - a transfer tensor (internal nodes)
        // In BOBA, we store these using two boba::Arrays whose size equals the number of
        // nodes in the dimension tree:
        //   - basis_matrices[node]:         std::optional<U_type>
        //       U_type is a 2D boba::Matrix of size (mode_size × rank)
        //
        //   - transfer_tensors[node]:       std::optional<B_type>
        //       B_type is a 3D boba::Tensor of size (left_rank × right_rank × parent_rank)
        // Each entry is wrapped in std::optional so that a node can cleanly represent
        // “this node has a basis matrix” or “this node has a transfer tensor,” but
        // not both.  Internal consistency (correct ranks, correct leaf/internal
        // placement, and correct parent–child dimensions) is verified by the
        // HierarchicalTucker constructor.
        //
        // Below, we show how to extract the basis matrices and transfer tensors from an existing
        // HierarchicalTucker object and use them to construct a new one.
        const auto& transfer_tensors = A_ht.get_transfer_tensors();
        const auto& basis_matrices   = A_ht.get_basis_matrices();
        boba::HierarchicalTucker<num_dims, space, double> another_ht(transfer_tensors, basis_matrices, dim_tree);

        // Finally, we can also create an HierarchicalTucker object from a full tensor by compressing it. This requires access to the full tensor, which
        // will generally not be possible for high-dimensional tensors (due to memory constraints). The code below shows how to do this.
        //
        // First, create a boba::Tensor, for example, a tensor filled with ones
        boba::Tensor<num_dims, space, double> tensor_A(sizes);
        tensor_A.fill_with(1.0);

        // We now create an HierarchicalTucker object from this boba::Tensor by compressing it
        boba::HierarchicalTucker<num_dims, space, double> ones_ht(sizes, dim_tree);
        ones_ht.compress(tensor_A);

        // The tensor extents (sizes) should be the same
        boba_always_assert_equal(tensor_A.sizes(), ones_ht.get_sizes(), "Mismatch in sizes!");

        boba_print("This tensor is a rank-1 HTD:");
        pass_or_fail_bool(check, (ones_ht.get_ranks() == boba::filled_array<5>(1_z)) );

        // We can inspect the properties of the HierarchicalTucker object using various print methods
        // Note: All print methods accept an optional string label to identify the output with a particular HierarchicalTucker
        boba_print("Inspecting the properties of the HierarchicalTucker object:");
        ones_ht.print("ones_ht"); // Allows users to inspect the values of the transfer tensors and basis matrices (useful for small tensors)
        ones_ht.print_at_node(4, "ones_ht"); // Inspect either the transfer tensor or basis at a given node, e.g., node index 4
        ones_ht.print_dimensions("ones_ht"); // Inspect the dimensions of the transfer tensors and basis matrices

        // The number of values needed to represent a low-rank tensor is << 10^3
        boba_print("Checking the compression rate of the low-rank tensor:");
        boba_print(tensor_A.get_number_elements());
        boba_print(ones_ht.get_number_elements());
        boba_print(ones_ht.get_compression_rate());

        // Arithmetic is also overloaded for HTuckers and we can use the Frobenius norm to
        // measure the size of the difference between two HTuckers
        boba_print("If two HTuckers are identical, then the Frobenius norm of their difference is exactly zero.");
        auto ones_ht_copy = ones_ht;
        pass_or_fail(check, boba::norm_frobenius(ones_ht - ones_ht_copy), 1.0e-4);

        // We can also verify equivalence using "=="
        pass_or_fail_bool(check, ones_ht == ones_ht_copy);
    }

    checkpoint();
    {
        boba_print("Orthogonalization and rounding.");

        // First create a balanced dimension tree for a 4-tensor
        constexpr size_t num_dims = 4;
        auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(num_dims));

        // Next we'll assume that the mode sizes consist of 32 elements per dimension
        auto sizes = boba::filled_array<num_dims>(32_z);

        // Make two random HierarchicalTucker objects
        boba::HierarchicalTucker<num_dims, space, double> A_ht(sizes, dim_tree);
        A_ht.fill_with_random();

        boba::HierarchicalTucker<num_dims, space, double> B_ht(sizes, dim_tree);
        B_ht.fill_with_random();

        // In high-dimensions, the storage becomes highly favorable for HTDs, which
        // often results in large compression rates.
        boba_print("We can inspect the properties of these HierarchicalTucker objects.");
        boba_print(A_ht.get_ranks());
        boba_print(B_ht.get_ranks());

        A_ht.print_dimensions("A_ht");
        B_ht.print_dimensions("B_ht");

        // In high-dimensions, the storage becomes highly favorable for HierarchicalTucker, which
        // often results in large compression rates.
        boba_print(A_ht.get_number_elements());
        boba_print(A_ht.get_compression_rate());

        // When we add the two HierarchicalTucker objects, the storage can increase in a dramatic way
        // if we do not perform rounding (truncation).
        auto C_ht = A_ht + B_ht;

        boba_print("Let's check the initial compression data for the HierarchicalTucker.");
        boba_print(C_ht.get_ranks());
        boba_print(C_ht.get_number_elements());
        boba_print(C_ht.get_compression_rate());

        boba_print("Applying rounding (SVD truncation) to control HierarchicalTucker ranks.");

        // Default SVD tolerances are quite small, so rounding should have a minimal effect
        boba_print("Default SVD tolerances:");
        boba_print(C_ht.get_svd_relative_tolerance());
        boba_print(C_ht.get_svd_absolute_tolerance());

        // To round an HierarchicalTucker, we simply call the corresponding method round()
        C_ht.round();

        // Record compression stats after rounding with the default tolerances
        auto ranks_default  = C_ht.get_ranks();
        auto elems_default  = C_ht.get_number_elements();
        auto compr_default  = C_ht.get_compression_rate();

        boba_print("After default rounding:");
        boba_print(ranks_default);
        boba_print(elems_default);
        boba_print(compr_default);

        // Increase the truncation tolerance to enforce stronger compression
        boba_print("Increasing truncation tolerance (1e-4) reduces ranks and increases compression:");

        C_ht.set_svd_relative_tolerance(1.0e-4);
        C_ht.set_svd_absolute_tolerance(1.0e-4);
        C_ht.round();

        // Record compression stats after aggressive rounding
        auto ranks_relaxed  = C_ht.get_ranks();
        auto elems_relaxed  = C_ht.get_number_elements();
        auto compr_relaxed  = C_ht.get_compression_rate();

        boba_print("After relaxed rounding:");
        boba_print(ranks_relaxed);
        boba_print(elems_relaxed);
        boba_print(compr_relaxed);

        // Aggressive truncation reduces the storage
        // Note: The ranks can be equal in some places if the HierarchicalTucker is rank-1 at a particular node
        pass_or_fail_bool(check, elems_default  > elems_relaxed);
        pass_or_fail_bool(check, compr_default  < compr_relaxed);
        pass_or_fail_bool(check, ranks_default  >= ranks_relaxed);
    }

    checkpoint();
    {
        boba_print("Obtaining a boba::Tensor from a boba::HierarchicalTucker and more advanced truncation interfaces.");

        // First we create the dimension tree and specify mode sizes for the tensor
        constexpr size_t num_dims = 5;
        auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(num_dims));
        auto sizes = boba::filled_array<num_dims>(8_z);

        // Make an HierarchicalTucker object filled with 1
        boba::HierarchicalTucker<num_dims, space, double> A_ht(sizes, dim_tree);
        A_ht.fill_with(1.0);

        // Instead of adding many tensors together and then calling round, we provide
        // an alternative interface that accepts a collection of HTuckers and
        // performs successive addition and rounding
        // Note: Other interfaces are available (see below)
        auto result_ht = boba::sum_and_round({A_ht, 2*A_ht, 3*A_ht});

        // Create an exact HierarchicalTucker filled with 6
        boba::HierarchicalTucker<num_dims, space, double> exact_ht(sizes, dim_tree);
        exact_ht.fill_with(6.0);

        // Compute the Frobenius norm of the error between the approximations
        // Before computing the norm, one should orthogonalize the argument for stability
        auto error_ht = result_ht - exact_ht;
        error_ht.orthogonalize();
        pass_or_fail(check, boba::norm_frobenius(error_ht), 1.0e-4);

        // To obtain a tensor from an instance of HierarchicalTucker, we simply call HierarchicalTucker::decompress()
        // Note: This preserves the original HierarchicalTucker and provides the tensor as well
        auto result_full = result_ht.decompress();

        // Create an exact tensor filled with 6
        boba::Tensor<num_dims, space, double> exact_full(sizes);
        exact_full.fill_with(6.0);

        // Check that the sizes match and approximation error is reasonable
        pass_or_fail_bool(check, exact_full.sizes() == result_ht.get_sizes());
        pass_or_fail(check, norm_difference_frobenius(result_full, exact_full), 1.0e-4);

        // Users can also store HTuckers in containers and call sum_and_round(), which is
        // overloaded to work with several standard containers. The different code blocks
        // below demonstrate these interfaces

        // (1) Using a std::initializer_list of HTuckers.
        // This is convenient, but the listed HTuckers are copied into the
        // initializer-list backing array.
        {
            boba_print("Checking sum_and_round(): std::initializer_list of HTuckers.");
            std::initializer_list<boba::HierarchicalTucker<num_dims, space, double>> inputs{A_ht, 2*A_ht, 3*A_ht};
            auto result_from_il = boba::sum_and_round(inputs);
            pass_or_fail(check, norm_difference_frobenius(result_from_il.decompress(), result_ht.decompress()), 1.0e-4);
        }

        // (2) Using a std::vector of HTuckers
        // Note: the vector construction below copies the listed HTuckers into the
        // container, but sum_and_round(inputs) reads from the vector without copying it.
        {
            boba_print("Checking sum_and_round(): std::vector of HTuckers.");
            std::vector<boba::HierarchicalTucker<num_dims, space, double>> inputs{A_ht, 2*A_ht, 3*A_ht};
            auto result_from_vector = boba::sum_and_round(inputs);
            pass_or_fail(check, norm_difference_frobenius(result_from_vector.decompress(), result_ht.decompress()), 1.0e-4);
        }

        // (3) Using a std::span of const HTuckers obtained from an existing std::vector.
        // The span is a non-owning view of the existing vector data, and an initializer list could also be used.
        {
            boba_print("Checking sum_and_round(): std::span of const HTuckers.");
            std::vector<boba::HierarchicalTucker<num_dims, space, double>> inputs{A_ht, 2*A_ht, 3*A_ht};
            std::span<const boba::HierarchicalTucker<num_dims, space, double>> inputs_as_span{inputs};
            auto result_from_span = boba::sum_and_round(inputs_as_span);
            pass_or_fail(check, norm_difference_frobenius(result_from_span.decompress(), result_ht.decompress()), 1.0e-4);
        }
    }

    checkpoint();
    {
        boba_print("Kronecker product and the Hadamard product of HierarchicalTucker objects.");

        // First create a balanced dimension tree for a 3-tensor
        constexpr size_t num_dims = 3;
        auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(num_dims));

        // Specify tolerances for truncation
        double relative_tolerance = 1.0e-10;
        double absolute_tolerance = 1.0e-10;

        // Next we'll assume that the mode sizes consist of 16 elements per dimension
        auto sizes = boba::filled_array<num_dims>(16_z);

        // Make an HierarchicalTucker filled with random numbers
        boba::HierarchicalTucker<num_dims, space, double> A_ht(sizes, dim_tree);
        A_ht.fill_with_random();
        A_ht.set_svd_relative_tolerance(relative_tolerance);
        A_ht.set_svd_absolute_tolerance(absolute_tolerance);
        auto A_full = A_ht.decompress();

        // Make another HierarchicalTucker object filled with 1, and then add it to itself (twice) to create a "rank-4" HierarchicalTucker
        boba::HierarchicalTucker<num_dims, space, double> B_ht(sizes, dim_tree);
        B_ht.fill_with(1.0);
        B_ht += B_ht;
        B_ht += B_ht;
        B_ht.set_svd_relative_tolerance(relative_tolerance);
        B_ht.set_svd_absolute_tolerance(absolute_tolerance);
        auto B_full = B_ht.decompress();

        boba_print("Ranks of the input HTuckers:");
        boba_print(A_ht.get_ranks());
        boba_print(A_ht.get_number_elements());
        boba_print(A_ht.get_compression_rate());

        boba_print(B_ht.get_ranks());
        boba_print(B_ht.get_number_elements());
        boba_print(B_ht.get_compression_rate());

        // Compute the exact Kronecker and Hadamard products to use for error measurements
        auto kronecker_exact_full = boba::tensor_product(A_full, B_full);
        auto hadamard_exact_full = boba::elementwise_product(A_full, B_full);

        // To compute the Kronecker product, simply use the tensor_product function
        auto kronecker_ht = boba::tensor_product(A_ht, B_ht);
        pass_or_fail(check, boba::norm_difference_frobenius(kronecker_ht.decompress(), kronecker_exact_full), 1.0e-4);

        boba_print("The tensor (Kronecker) product squares both the sizes and the ranks of the input HTuckers:");
        boba_print(kronecker_ht.get_sizes());
        boba_print(kronecker_ht.get_ranks());
        boba_print(kronecker_ht.get_number_elements());
        boba_print(kronecker_ht.get_compression_rate());

        // The "direct" element-wise product of two HTuckers is provided as an overload of the * operator;
        auto hadamard_direct_ht = A_ht * B_ht;
        pass_or_fail(check, boba::norm_difference_frobenius(hadamard_direct_ht.decompress(), hadamard_exact_full), 1.0e-4);

        boba_print("The 'exact' element-wise product squares the ranks of the input HTuckers:");
        boba_print(hadamard_direct_ht.get_sizes());
        boba_print(hadamard_direct_ht.get_ranks());
        boba_print(hadamard_direct_ht.get_number_elements());
        boba_print(hadamard_direct_ht.get_compression_rate());

        // The "direct" element-wise product of two HTuckers can also be calculated using the elementwise_product function
        auto alternative_hadamard_direct_ht = boba::elementwise_product(A_ht, B_ht);
        pass_or_fail(check, boba::norm_difference_frobenius(alternative_hadamard_direct_ht.decompress(), hadamard_exact_full), 1.0e-4);

        // We also provide an approximate element-wise product of two HTuckers which is more efficient than the exact product
        // This function accepts a pair of HTDs as well as two additional parameters, which specify (relative and absolute) truncation tolerances
        // Note: It is generally recommended to truncate the result of this function once more to obtain better compression
        auto hadamard_approx_ht = boba::elementwise_product(A_ht, B_ht, relative_tolerance, absolute_tolerance);
        pass_or_fail(check, boba::norm_difference_frobenius(hadamard_approx_ht.decompress(), hadamard_exact_full), 1.0e-4);

        boba_print("The 'approximate' element-wise product avoids the rank growth found in the exact element-wise product:");
        boba_print(hadamard_approx_ht.get_sizes());
        boba_print(hadamard_approx_ht.get_ranks());
        boba_print(hadamard_approx_ht.get_number_elements());
        boba_print(hadamard_approx_ht.get_compression_rate());

        boba_print("Rounding the result of the approximate product helps improve compression:");
        hadamard_approx_ht.set_svd_relative_tolerance(relative_tolerance);
        hadamard_approx_ht.set_svd_absolute_tolerance(absolute_tolerance);
        hadamard_approx_ht.round();
        pass_or_fail(check, boba::norm_difference_frobenius(hadamard_approx_ht.decompress(), hadamard_exact_full), 1.0e-4);
        boba_print(hadamard_approx_ht.get_sizes());
        boba_print(hadamard_approx_ht.get_ranks());
        boba_print(hadamard_approx_ht.get_number_elements());
        boba_print(hadamard_approx_ht.get_compression_rate());
    }

    checkpoint();
    {
        boba_print("Reading and writing HierarchicalTucker objects.");

        // In addition to the various print methods which can be used to inspect an HierarchicalTucker, we provide
        // methods to read/write HierarchicalTucker objects from/to files

        // Let's make a random six-dimensional HierarchicalTucker
        constexpr size_t num_dims = 6;
        auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(num_dims));
        auto sizes = boba::filled_array<num_dims>(32_z);

        // Make an HierarchicalTucker object filled with random values
        boba::HierarchicalTucker<num_dims, space, double> A_ht(sizes, dim_tree);
        A_ht.fill_with_random();

        boba_print("Saving the HierarchicalTucker to disk.");
        boba::write_to_file(A_ht, "A_ht_test");

        // To load the HierarchicalTucker from disk, one can call the constructor with a file prefix
        // We check to make sure that these HTuckers are equivalent
        boba_print("Loading the HierarchicalTucker from disk");
        boba::HierarchicalTucker<num_dims, space, double> A_ht_loaded;
        boba::read_from_file(A_ht_loaded, "A_ht_test");
    }

    boba::finalize();
    return final_check(check);
}
// clang-format on
