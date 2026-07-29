// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  This tutorial demonstrates how to use the Tensor object in the code
*/

int main(int argc, char *argv[]) {

  boba::splash();
  boba::init();
  bool check = true;

  std::cout << "BoBa Tutorial on Tensors" << std::endl;

  //
  // Tensors
  //
  checkpoint();
  {
    /*
      The building block of tensorial discretizations are Tensors.
      These are characterized by the sizes of each dimension and the memory space
      For example,
        a vector of length 10 is a 1st order tensor of sizes {10}
        a m-by-n matrix is a 2nd order tensor of sizes of {m, n}
        a Rubik's cube is a 3rd order tensor of  sizes {3, 3, 3}
      A tensor will be insantiated on the CPU/host or on a device memory space
      The data is uninitialized upon construction of the object (at runtime)
    */
    constexpr ::boba::execution_space host_space = ::boba::execution_space::CPU;
    constexpr ::boba::execution_space device_space = ::boba::default_execution_space;
    /* ^                 ^                                       ^
       |                 |                    the default execution space is set to
       |                 |       CPU/CUDA/HIP if at compile time the user sets BOBA_CPU=1/BOBA_CUDA=1/BOBA_HIP=1
       |                 |                        If nothing is set, default_execution_space is CPU
       |             enumeration (aka enum) of the execution spaces (see boba.hpp for more)
      marks this variable as a compile time variable
    */

    ::boba::Tensor<3, device_space, double> example_tensor({5, 7, 9});
    /*  ^          ^      ^           ^                     ^-----^
        |          |      |           |                        |
        |          |      |     data type              sizes of the tensor
        |          |    memory space of data
        |      tensor order
      boba namespace

      Note that the sizes form a boba::Array
      Let's not initialize the memory yet.
      There are tons of useful functions that allow us to do things with tensors.
      Check them out in include/BOBA/tensors/Tensor.hpp
    */

    ::boba::Matrix<device_space, double> example_matrix({5, 7});
    /*  ^            ^                                   ^--^
        |            |                                     |
        |        memory space of data        sizes of the matrix (rows then columns)
      boba namespace

      A matrix is a special case of tensor, a 2-tensor
      So any function that exists for tensors can be used on matrices
      Matrix objects also have additional functions specific to matrix operations
      Check them out in include/BOBA/tensors/Matrix.hpp
      For example, we can set out matrix to have ones on the diagonal
      using this command:
    */
    example_matrix.set_to_identity_matrix();

    ::boba::Vector<device_space, double> example_vector({9});
    /*  ^            ^                                           ^
        |            |                                           |
        |        memory space of data                 size of the vector
      boba namespace

      A vector is a special case of tensor, a 1-tensor
      So any function that exists for tensors can be used on vectors
      Vector objects also have additional functions specific to matrix operations
      Check them out in include/BOBA/tensors/Vector.hpp
    */
    example_vector.fill_with(2);

    /*
      Data is accessed through either
        (A) the raw pointer, which can be obtained via the data() or const_data() methods
        (B) views to the data, via the view() or const_view() methods
      A view is simply
        (1) a pointer to the data and
        (2) the information (sizes, strides, etc) needed to access that data
      A critical thing to know is that views have built-in multiindexers
      that are consistent with the size of the containing tensor!

      TIP: when appropriate, get a read-only const view to ensure that the data isn't overwritten
    */
    {
      auto my_first_view = example_vector.view();
      auto my_first_const_view = example_vector.const_view();

      boba_print(my_first_view.sizes());
      boba_print(my_first_view.strides());
      boba_print(my_first_const_view.size());
    }

    /*
      Let's say we wanted to set up the tensor according to:
      example_tensor({i, j, k}) = example_matrix({i, j})*example_vector({k})
      Here we provide different ways of doing this:

      In the first way, we do a 3-dimensional loop over i, j, k.
      First, we need the right views,
      Then we create the appropriate loop,
      Then use the views in the loop to read and write data
    */
    checkpoint();
    {
      auto example_tensor_view = example_tensor.view();
      auto example_matrix_view = example_matrix.const_view();
      auto example_vector_view = example_vector.const_view();

      // example_tensor_view.sizes() is a 3-array, so the loop knows we are
      // doing a 3-dimensional loop
      ::boba::loop<device_space, 3>(example_tensor_view.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
      {
        auto [i, j, k] = mid;
        auto matrix_entry = example_matrix_view({i, j});
        auto vector_entry = example_vector_view(k);
        example_tensor_view(mid) = matrix_entry*vector_entry;
      });
    }

    /*
      Let's do the same operation in a different way.
      This time, we will work with the long indices.
      This kind of thinking is crucial to working in arbitrary dimensions
      (spend a while to think about how you would do this without multiindices)
    */
    checkpoint();
    ::boba::Tensor<3, host_space, double> example_tensor_2({5, 7, 9});
    {
      /*
        Note that example_tensor_2 is defined as on the host_space, but we want to do
        our work on the device_space. These are the same if doing a CPU-only run,
        but if you are running on GPUs with BOBA_CUDA=1 or BOBA_HIP=1, then
        we need to transfer the memory to the device.
        This can be done with BoBa using our host/device constructors.
        We will transfer the data back to the CPU after this computation.
        NOTE: For maximum performance, you want to avoid copying back and forth from the device.
        You will notice that in our examples we try our best to keep data in the execution space as long as possible.
      */
      ::boba::Tensor<3, device_space, double> example_tensor_2_on_device{example_tensor_2};

      auto example_tensor_view = example_tensor_2_on_device.view();
      auto example_matrix_view = example_matrix.const_view();
      auto example_vector_view = example_vector.const_view();

      // example_tensor_view.size() is a scalar, so the loop knows we are
      // doing a 1-dimensional loop
      ::boba::loop<device_space, 1>(example_tensor_view.size(),
        [=]__boba_host_device__(size_t I)
      {
        // Create a multiindex
        // tensor_mid = {i, j, k}
        auto tensor_mid = example_tensor_view.multiindex(I);
        // Get k (for vectors, the long index and multiindex are always the same)
        auto vector_mid = tensor_mid[2];
        // Need {i, j}, so take tensor_mid and drop k
        // Checkout out include/BOBA/objects/Array.hpp see how delete_element works
        auto matrix_mid = delete_element(tensor_mid, 2);
        auto matrix_entry = example_matrix_view(matrix_mid);
        auto vector_entry = example_vector_view(vector_mid);
        example_tensor_view(tensor_mid) = matrix_entry*vector_entry;
      });

      /*
        Now to transfer the data back to the CPU,
        which automatically handles the fact that the tensors live in difference spaces.
        Under the hood, we see the compile-time mismatch of the spaces, and thus can call the correct copy function
      */
      checkpoint();
      example_tensor_2 = example_tensor_2_on_device;
    }

    /*
      These two methods of setting up example_tensor / example_tensor_2  are equivalent,
      but let's verify using the Frobenius norm
    */
    checkpoint();
    ::boba::Tensor<3, device_space, double> example_tensor_2_space{example_tensor_2};

    /*
     In tests and examples, this pass_or_fail function is used to determine if this test passed,
     based on some tolerance, and updates check to be true or false
     The tests pass if the error (second arg) is less than the tolerance (third arg)
    */
    pass_or_fail(check, boba::norm_difference_frobenius(example_tensor_2_space, example_tensor), 1e-09);
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
// clang-format on
