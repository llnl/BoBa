// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  Demonstrates some basics regarding Tucker decompositions
  Here we assume you have been introduced to the basics of Tucker decompositions
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main() {

  boba::splash();
  boba::init();

  bool check = 1;

  checkpoint();
  {
    boba_print("Tucker from a 3-tensor");

    // Let's fill a 10x10x10 tensor with ones.
    // create a 3-array {10, 10, 10}
    auto sizes = ::boba::filled_array<3>(10_z);
    boba::Tensor<3, space, double> tensor_A(sizes);
    tensor_A.rename("ones");
    tensor_A.fill_with(1.0);

    // Method 1: instantiate a Tucker
    ::boba::Tucker<3, space, double> tuck_A(tensor_A.sizes());

    // Compress the tensor
    tuck_A.compress(tensor_A);

    // We see that the tensor extents are the same, as expected
    boba_print(tensor_A.sizes());
    boba_print(tensor_A.number_nonzeros());
    boba_print(tuck_A.sizes());

    // However this tensor is rank one, as expected
    boba_print(tuck_A.compression_rate());
    boba_print(tuck_A.ranks_string());

    // Thus, the number of values needed to represent to low rank tensor
    // is much less than 10^3
    boba_print(tensor_A.get_number_elements());
    boba_print(tuck_A.get_number_elements());

    boba_print("We also could have just use a special function for constant functions");

    // Let's fill a 10x10x10 tensor with ones.
    ::boba::Tucker<3, space, double> tuck_A2(tensor_A.sizes());

    tuck_A2.fill_with(1.0);

    boba_print(tuck_A2.sizes());
    boba_print(tuck_A2.compression_rate());
    boba_print(tuck_A2.ranks_string());
    boba_print(tuck_A2.get_number_elements());

    // We can also subtract a Tucker from another and check the norm of this difference
    auto difference = tuck_A - tuck_A2;
    auto error_norm = ::boba::norm_frobenius(difference);

    pass_or_fail(check, error_norm, 1.0e-4);
  }

  checkpoint();
  {
    // What is the Tucker decomposition made of?
    auto sizes = ::boba::filled_array<3>(10_z);
    ::boba::Tucker<3, space, double> tuck_A(sizes);
    tuck_A.fill_with(1.0);

    // We can print the Tucker and it will show us all of the individual matrices that make up the Tucker
    tuck_A.print();

    // We can access the cores directly, as they are matrices
    tuck_A.cores[0].print();

    // We can access the main core directly, as it is a tensor
    tuck_A.R_core.print();

    // Which means they can be modified directly
    // Let's modify the first core
    auto core_view = tuck_A.cores[0].view();

    ::boba::loop<space, 2>(core_view.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 2> mid)
      {
        auto [index, rank] = mid;
        core_view({index, rank}) = index;
      });

    tuck_A.print();
  }

  checkpoint();
  {
    boba_print("\n Sparse vs Low Rank example");

    // Let's construct the identity matrix as a tensor
    // create a 2-array {10, 10}
    auto sizes = ::boba::filled_array<2>(10_z);
    boba::Tensor<2, space, double> tensor_A(sizes);
    auto tensor_A_view = tensor_A.view();

    ::boba::loop<space, 2>(tensor_A.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 2> mid)
      {
        auto [i, j] = mid;
        // use a ternary expression
        tensor_A_view(mid) = (i == j) ? 1.0 : 0.0;
      });

    boba_print(tensor_A.size());
    boba_print(tensor_A.number_nonzeros());

    // Let's use a utility in Tucker_utilities.hpp
    auto tuck_A = ::boba::compress_to_tucker(tensor_A);

    // We see that the tensor extents are the same, as expected
    boba_print(tensor_A.sizes());
    boba_print(tuck_A.sizes());

    // The tensor in the above example was dense, but low rank
    // this tensor is sparse, but NOT low rank!
    // Thus, the number of values needed to represent this tensor
    // is more than the size of the original tensor!
    boba_print(tensor_A.get_number_elements());
    boba_print(tuck_A.get_number_elements());
    boba_print(tuck_A.compression_rate());
    boba_print(tuck_A.ranks_string());
  }

  checkpoint();
  {
    boba_print("\n Importance of coordinate systems");

    // Say you have a tensor representing a function
    //  f(x, y, z) = x * z + x^2 * z^2  + x^3 * z^3
    // Note that this is not a function of y, so there is some compression to be gained!
    // Let's make the tensor

    auto sizes = ::boba::filled_array<3>(10_z);
    boba::Tensor<3, space, double> tensor_A(sizes);
    auto tensor_A_view = tensor_A.view();

    ::boba::loop<space, 3>(tensor_A.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
      {
        auto [i, j, k] = mid;
        // Note that we don't use y
        boba::detail::ignore(j);
        double x = double(i)/double(tensor_A_view.sizes(0));
        double z = double(k)/double(tensor_A_view.sizes(2));
        tensor_A_view(mid) = x*z + x*x*z*z + x*x*x*z*z*z;
      });

    boba_print(tensor_A.size());

    // Let's compress this like before
    auto tuck_A = ::boba::compress_to_tucker(tensor_A);

    boba_print("compress tensor");
    boba_print(tuck_A.sizes());
    boba_print(tensor_A.get_number_elements());
    boba_print(tuck_A.get_number_elements());
    boba_print(tuck_A.compression_rate());
    boba_print(tuck_A.ranks_string());

    /*
    We see that the "y" core is rank 1, indicating that we have compressed out
    that dimension entirely.
    Recall that in the tensor train tutorial we had to permute the dimensions to
    achieve this effect. On the other hand, the tensor train compression rate is higher.
    This is the benefit/cost of the symmetry property of the Tucker decomposition.
    */
  }

  boba::finalize();
  return final_check(check);
}
// clang-format on
