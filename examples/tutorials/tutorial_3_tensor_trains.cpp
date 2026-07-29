// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  Demonstrates some basics regarding tensor trains
  Here we assume you have been introduced to the basics of tensor trains
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main() {

  boba::splash();
  boba::init();

  bool check = 1;

  checkpoint();
  {
    boba_print("Tensor train from a 3-tensor");

    // Let's fill a 10x10x10 tensor with ones.
    // create a 3-array {10, 10, 10}
    auto sizes = ::boba::filled_array<3>(10_z);
    boba::Tensor<3, space, double> tensor_A(sizes);
    tensor_A.rename("ones");
    tensor_A.fill_with(1.0);

    // Method 1: instantiate a tensor train
    ::boba::TensorTrain<3, space, double> tt_A(tensor_A.sizes());

    // Compress the tensor
    tt_A.compress(tensor_A);

    // We see that the tensor extents are the same, as expected
    boba_print(tensor_A.sizes());
    boba_print(tensor_A.number_nonzeros());
    boba_print(tt_A.sizes());

    // However this tensor is rank one, as expected
    boba_print(tt_A.compression_rate());
    boba_print(tt_A.ranks_string());

    // Thus, the number of values needed to represent to low rank tensor
    // is much less than 10^3
    boba_print(tensor_A.get_number_elements());
    boba_print(tt_A.get_number_elements());

    boba_print("We also could have just use a special function for constant functions");

    // Let's fill a 10x10x10 tensor with ones.
    ::boba::TensorTrain<3, space, double> tt_A2(tensor_A.sizes());

    tt_A2.fill_with(1.0);

    boba_print(tt_A2.sizes());
    boba_print(tt_A2.compression_rate());
    boba_print(tt_A2.ranks_string());
    boba_print(tt_A2.get_number_elements());

    // We can also subtract a tt from another and check the norm of this difference
    auto difference = tt_A - tt_A2;
    pass_or_fail(check, ::boba::norm_frobenius(difference), 1.0e-4);
  }

  checkpoint();
  {
    // What is the tensor train made of?
    auto sizes = ::boba::filled_array<3>(10_z);
    ::boba::TensorTrain<3, space, double> tt_A(sizes);
    tt_A.fill_with(1.0);

    // We can print the tt and it will show us all of the individual tensors that make up the tt
    tt_A.print();

    // We can access the cores directly, as they are 3-tensors.
    tt_A.cores[0].print();

    // Which means they can be modified directly
    // Let's modify the first core
    auto core_view = tt_A.cores[0].view();

    ::boba::loop<space, 3>(core_view.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
      {
        auto [rank_left, index, rank_right] = mid;
        core_view({rank_left, index, rank_right}) = index;
      });

    tt_A.print();
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

    // Let's use a utility in TensorTrain_utilities.hpp
    auto tt_A = ::boba::compress_to_TensorTrain(tensor_A);

    // We see that the tensor extents are the same, as expected
    boba_print(tensor_A.sizes());
    boba_print(tt_A.sizes());

    // The tensor in the above example was dense, but low rank
    // this tensor is sparse, but NOT low rank!
    // Thus, the number of values needed to represent this tensor
    // is more than the size of the original tensor!
    boba_print(tensor_A.get_number_elements());
    boba_print(tt_A.get_number_elements());
    boba_print(tt_A.compression_rate());
    boba_print(tt_A.ranks_string());
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
    auto tt_A = ::boba::compress_to_TensorTrain(tensor_A);

    boba_print("compress tensor");
    boba_print(tt_A.sizes());
    boba_print(tensor_A.get_number_elements());
    boba_print(tt_A.get_number_elements());
    boba_print(tt_A.compression_rate());
    boba_print(tt_A.ranks_string());

    // But wait! Notice in the ranks string that the middle core,
    // which represents y, has left/right ranks of 3 - even though
    // our function is independent of y!
    // It seems the x-z coupling is "passing through" the y core.
    // Let's try to fix this by putting the x and z cores together

    auto tensor_B = tensor_A;

    ::boba::permute({"x", "y", "z"}, tensor_B, {"x", "z", "y"});
    auto tt_A2 = ::boba::compress_to_TensorTrain(tensor_B);

    boba_print("compress permuted tensor");
    boba_print(tt_A2.sizes());
    boba_print(tt_A2.get_number_elements());
    boba_print(tt_A2.compression_rate());
    boba_print(tt_A2.ranks_string());

    // The compression rate more than DOUBLED!

    // What if we don't know the optimal coordinate system?
    // We can use find_optimal_tensor_permutation
    // This performs a brute forces search for the optimal permuation.
    auto [optimal_mid, optimal_tt] = ::boba::find_optimal_tensor_permutation(tensor_A, tt_A2.svd_tolerance_relative, true);
    boba_print("using find_optimal_tensor_permutation");
    boba_print(optimal_mid);
    boba_print(optimal_tt.get_number_elements());
    boba_print(optimal_tt.compression_rate());
    boba_print(optimal_tt.ranks_string());
  }

  boba::finalize();
  return final_check(check);
}
// clang-format on
