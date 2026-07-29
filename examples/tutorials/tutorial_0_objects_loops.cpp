// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  This tutorial covers some of the basic working objects like arrays,
  as well as some of the computational abstractions such as multiindices and loops.
  We recommend starting with tutorial_inputs_and_debugging before this tutorial.
*/

int main(int argc, char *argv[]) {

  boba::splash();
  boba::init();
  bool check = true;

  std::cout << "BoBa Tutorial for Objects and Loops " << std::endl;

  //
  // Array
  //
  //
  checkpoint();
  {
    /*
      The most fundamental object that is specific to BoBa is the array.
      This is very much like C++'s std::array, except it works well with GPU kernels
      The array is characterized by a compile-time size (that is, it is statically sized).
      You also choose the data type stored in the array at compile time.
      When working with BoBa, please use the BoBa array.
    */
    ::boba::Array<size_t, 7> my_first_array{0, 1, 1, 2, 3, 5, 8};
    /*  ^         ^       ^                 ^-----------------^
        |         |       |                          |
        |         |      length of array         array data
        |       data type
      boba namespace
    */

    // You can also create an array and change the data later
    ::boba::Array<size_t, 7> my_second_array;

    boba_print(my_first_array);

    // Some examples of changing the data
    my_second_array[0] = 0;
    my_second_array[1] = 1;
    for(size_t i = 2; i < 7; i++)
    {
      my_second_array[i] = my_second_array[i-1] + my_second_array[i-2];
    }

    // Sanity check
    boba_always_assert_equal(my_second_array, my_first_array, "These arrays should match");
  }

  //
  // Multiindices
  //
  checkpoint();
  {
    /*
      BoBa makes frequent use of multiindices, and so has a helper class for this.
      Let's say you are indexing into a 3-dimensional space with extents {5, 7, 9}
      You might use a multiindex:
        mid = {i, j, k}
      or the equivalent long index
        id = stride_0*i + stide_1*j + stide_2*k
      We can compute the strides
        stride_0 = 1
        stride_1 = size_0 = 5
        stride_2 = size_0*size_1 = 5*7 = 35
        id = i + 5*j + 35*k
      This tells us how to map to and from the long index and the multiindex
    */
    ::boba::Multiindexer<3> example_multiindexer({5, 7, 9});
    /*  ^                ^                        ^-----^
        |                |                           |
        |                |                sizes of the multiindexer
        |             dimension of multiindexer
      boba namespace

      Note that the extents {5, 7, 9} are stored as a boba::Array
    */
    /*
      Consider the multiindex {i, j, k} = {1, 2, 3}
      the associated "flat index" is
        id = i + 5*j + 35*k = 1 + 5*2 + 35*3 = 1 + 10 + 105 = 116
      We can convert from the long index to the multiindex
    */

    ::boba::Array<size_t, 3> strides = example_multiindexer.precompute_strides({5, 7, 9});

    ::boba::Array<size_t, 3> known_strides{1, 5, 35};

    boba_always_assert_equal(known_strides, strides, "Strides don't match.");

    // Strides are also stored in the multiindexer as a boba::Array
    boba_always_assert_equal(known_strides, example_multiindexer.strides(), "Strides don't match.");

    ::boba::Array<size_t, 3> mid = example_multiindexer.multiindex(116);

    ::boba::Array<size_t, 3> known_mid{1, 2, 3};
    boba_always_assert_equal(known_mid, mid, "Multiindex doesn't match.");

    // Convert back to long index
    size_t id = example_multiindexer.index(mid);

    boba_always_assert_equal(id, size_t(116), "Long index doesn't match.");
  }

  //
  // Loops
  //
  checkpoint();
  {
    /*
      BoBa has performance portable loop abstractions that allow you to run loops on any device.
      You can run on the host space, which will be a CPU, or
      you can run on a device space, such as a GPU.
      When you compile on an appropriate machine, you can specify a flag to
      activate the device space - otherwise the device space will just be the same as the host space.

      Examples:
      |   Flag      |   host_space  | device_space |
      | <no flag>   |     CPU       |     CPU      |
      | BOBA_CUDA=1 |     CPU       |    H100      |
      | BOBA_HIP=1  |     CPU       |    Mi300A    |
    */
    constexpr ::boba::execution_space host_space = ::boba::execution_space::CPU;
    constexpr ::boba::execution_space device_space = ::boba::default_execution_space;

    // The BoBa loop abstraction utilizes lambdas to define

    auto my_host_device_lambda = [=]__boba_host_device__(size_t i){ printf("Hello from %lu\n", i); };
    /* ^     ^                    ^  ^-----------------^    ^         ^
       |     |                    |      |                  |         |
       |     |                    |      |                  |      do something in the lambda
       |     |                    |      |          parameters your lambda takes
       |     |                    |   attribute, must always be this phrase unless it is a host-only lambda
       |     |                 capture clause (by value =, by reference &, or nothing)
       |     name it
      the type must be 'auto'
    */
    // Then you dispatch the lambda with a boba loop
    ::boba::loop< host_space, 1>(0_z, 4_z, my_host_device_lambda);
    /* ^      ^       ^       ^   ^    ^    ^
       |      |       |       |   |    |    lambda that takes in a single integer (which will be the loop indices)
       |      |       |       |   |   final index of loop
       |      |       |       |  first index of loop
       |      |       |     loop dimensionality (1D loop, 2D loop, etc..)
       |      |    compute space to which this loop will be dispatched (chosen at compile time)
       |   loop function
      namepspace boba
    */

    /*
      Mostly, you'll see that we create the lambdas in-place,
      In this case, the lambdas have no name.
      The lambda is temporary and will cease to exist once execution completes.
      Note in this example that the lambda is created in-place as a argument to the loop function
    */
    ::boba::loop<host_space, 1>(0_z, 4_z,
      [=]__boba_host_device__(size_t i)
    {
      printf("Hello from the host_space at index %lu\n", i);
    });

    /*
      If the lower loop index is zero, you can omit it
    */
    ::boba::loop<device_space, 1>(4_z,
      [=]__boba_host_device__(size_t i)
    {
      printf("Hello from the device space at index %lu\n", i);
    });

    // If on a device, these loops are asynchronous with the CPU -
    // the CPU will keep executing the code (except for the loop) while the device is running the loop
    // Thus, sometimes you need to wait for the device to finish executing before continuing
    // This true for things like print statements and when host/device are sharing data
    ::boba::detail::device_sync();

    /*
      We can do multidimensional loops by making arrays describing the index bounds
      The loop indices are passed in as an array
    */
    ::boba::loop<device_space, 2>({0_z, 0_z}, {4_z, 5_z},
      [=]__boba_host_device__(::boba::Array<size_t, 2> ij)
    {
      auto [i, j] = ij; // this is called structured binding, and is equivalent to the following:
      /*
        auto i = ij[0];
        auto j = ij[1];
      */
      printf("Hello from the device space at multiindex {%lu, %lu} \n", i, j);
    });

    /*
      The arrays can come from another object (such as the sizes of a multiindexer)
    */
    auto I = 3_z;
    ::boba::Multiindexer<3> example_multiindexer({I, I+1, I+4});
    ::boba::Array<size_t, 3> loop_upper_bounds = example_multiindexer.sizes();

    // We also have helper functions for arrays
    auto loop_lower_bounds = ::boba::filled_array<3>(0_z);

    ::boba::loop<device_space, 3>(loop_lower_bounds, loop_upper_bounds,
      [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [i, j, k] = mid; // this structured binding is equivalent to the following:
      /*
        auto i = mid[0];
        auto j = mid[1];
        auto k = mid[2];
      */
      printf("Hello from the device space at multiindex {%lu, %lu, %lu} \n", i, j, k);
    });

    /*
      TIP 1: Use structured bindings!
      TIP 2: We can use multiindexers to convert between indices and multiindices
      TIP 3: If the lower bounds are just all zeros, then simply only provide upper bounds
             and BoBa loops will use all zeros for the lower indices automatically
    */
    ::boba::loop<device_space, 3>(example_multiindexer.sizes(),
      [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
    {
      auto [i, j, k] = mid;
      size_t index = example_multiindexer.index(mid);
      printf("Hello again from the device space at index %lu, equivalent to multiindex {%lu, %lu, %lu} \n", index, i, j, k);
    });

  }

  //
  // Arbitrary dimensional loops
  //
  checkpoint();
  {
    // BoBa only supports 1D 2D and 3D literal loops, due to these being the only choices on GPUs.
    // In order to do a 4D or higher loop, one must determine how to recast this as a lower dimensional loop
    // on GPUs. We didn't want to hide a particular choice behind a "4D loop" construct, as users should
    // explicitly decide how they want to recast their higher dimensional loops.
    // Here is ONE such way to do this

    // Create a multiindexer describing your higher dimensional space
    auto I = 3_z;
    ::boba::Multiindexer<5> mider({I, I + 1, I + 4, I + 2, I + 3});

    // Now loop over every flat index of the high-dimensional space
    // Since we are indexing over the flat index, this is a 1D loop
    ::boba::loop<boba::default_execution_space, 1>(mider.size(),
      [=]__boba_host_device__(size_t index)
    {
      // Now we can get the high dimensional multiindex by converting the flat index
      // into the corresponding multiindex
      auto mid = mider.multiindex(index);
      boba::detail::ignore(mid);

      // Alternative, we could use structured bindings:
      auto [i, j, k, l, p] = mider.multiindex(index);
      boba::detail::ignore(i);
      boba::detail::ignore(j);
      boba::detail::ignore(k);
      boba::detail::ignore(l);
      boba::detail::ignore(p);
    });
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
// clang-format on
