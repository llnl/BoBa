// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

#include <cmath>

/*
  Tests functionality of the SimplicialMultiindexer struct that enables indexing of upper-triangular parts of matrices and the higher dimensional analogues
  i.e. sets of multi-indices constrained to have non-increasing indices from left to right
*/

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/**
 * \brief
 * Tests whether or not proposed_permutation is in fact a permutation of array_original.  Does this by looping over all permutations
 * of array_original and returning true if one of them is equal to proposed_permutation.  Returns false otherwise.
 */
template <size_t arr_dim>
__boba_host_device__ bool is_valid_permutation_of_array(::boba::Array<size_t, arr_dim>& array_original, ::boba::Array<size_t, arr_dim>& proposed_permutation)
{
  bool is_valid = false;

  ::boba::Multiindexer<arr_dim> mid(::boba::filled_array<arr_dim>(arr_dim));

  for (size_t i = 0; i < mid.size(); i++)
  {
    ::boba::Array<size_t, arr_dim> indices_permutation = mid.multiindex(i);
    if (is_valid_permutation(indices_permutation))
    {
      ::boba::Array<size_t, arr_dim> current_permutation;
      for (size_t k = 0; k < arr_dim; k++)
      {
        // Fill current permutation with permuted values of original array
        current_permutation[indices_permutation[k]] = array_original[k];
      }
      if (proposed_permutation == current_permutation)
      {
        is_valid = true;
        break; // Break out of the loop once you've found a matching permutation
      }
    }
  }

  return is_valid;
}

template <size_t arr_dim>
void test_permutations_of_array(::boba::Array<size_t, arr_dim>& array, ::boba::SimplicialMultiindexer<arr_dim>& smid, size_t expected_num_perms)
{
  size_t num_perms = smid.number_permutations(array);
  boba_always_assert_equal(expected_num_perms, num_perms, "Wrong number of permutations");

  ::boba::Array<size_t, arr_dim> perm(array);
  boba_print(array);
  boba_print("List of permutations:");
  for (size_t i = 0; i < num_perms; i++)
  {
    boba_print(perm);
    bool valid_permutation = is_valid_permutation_of_array(array, perm);
    boba_assert(valid_permutation, "Found an invalid permutation");
    smid.get_next_permutation(perm);
  }
}

/***********************************************************************************/

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for boba SimplicialMultiindexer functionality" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  ::boba::argparser args(argc, argv);

  size_t n = 6;
  size_t verbosity = 0;

  static constexpr size_t big_dim = 4;

  args.add_optional_argument(n,
                             "-n",
                             "--resolution-multiindexers",
                             "Number of elements in testing of simplicial multi-indexers.");
  args.add_optional_argument(verbosity,
                             "-v",
                             "--verbose",
                             "If true, prints all the multi-indices and corresponding global indices.");

  args.parse_check();
  bool verbose = verbosity != 0;

  bool check = 1;

  /*** Two-dimensional verification ***/
  boba_print("Testing two-dimensional simplicial multi-indexer");
  ::boba::SimplicialMultiindexer<2> smid2(n);
  std::cout << "Size per dimension = " << n << ". Total size = " << smid2.total_size() << std::endl;
  ::boba::loop<space, 2>(::boba::filled_array<2>(n),
                         [=] __boba_host_device__(::boba::Array<size_t, 2> mindex)
  {
    if (::boba::is_nonincreasing(mindex))
    {
      size_t gindex = smid2.index(mindex);
      ::boba::Array<size_t, 2> mindex2 = smid2.multiindex(gindex);

      if (verbose)
      {
        printf("Input multi-index = {%lu, %lu} \n", mindex[0], mindex[1]);
        printf("Corresponding global index = %lu \n", gindex);
        printf("Recomputed multi-index = {%lu, %lu} \n", mindex2[0], mindex2[1]);
      }

      boba_assert_lt(gindex, smid2.total_size(), "gindex is too big!");
      boba_assert_ge(gindex, 0, "gindex must be non-negative");
      boba_always_assert_equal(mindex, mindex2, "Triangular multi-index not working");
    }
    else
    {
      return;
    }
  });

  // Permutation testing
  ::boba::Array<size_t, 2> array1{1, 0};
  test_permutations_of_array(array1, smid2, static_cast<size_t>(2));

  ::boba::Array<size_t, 2> array2{0, 0};
  test_permutations_of_array(array2, smid2, static_cast<size_t>(1));

  boba_print("All two-dimensional assertions passed");

  /*** Three-dimensional verification ***/
  boba_print(" ");
  std::cout << "Now testing three-dimensional simplicial multi-indexer" << std::endl;
  ::boba::SimplicialMultiindexer<3> smid3(n);
  std::cout << "Size per dimension = " << smid3.size_per_dimension() << ". Total size = " << smid3.total_size() << std::endl;
  ::boba::loop<space, 3>(::boba::filled_array<3>(n),
                         [=] __boba_host_device__(::boba::Array<size_t, 3> mindex3)
  {
    if (::boba::is_nonincreasing(mindex3))
    {
      size_t gindex = smid3.index(mindex3);
      ::boba::Array<size_t, 3> mindex3_2 = smid3.multiindex(gindex);

      if (verbose)
      {
        printf("Input multi-index = {%lu, %lu, %lu} \n", mindex3[0], mindex3[1], mindex3[2]);
        printf("Corresponding global index = %lu \n", gindex);
        printf("Recomputed multi-index = {%lu, %lu, %lu} \n", mindex3_2[0], mindex3_2[1], mindex3_2[2]);
      }

      boba_assert_lt(gindex, smid3.total_size(), "gindex is too big!");
      boba_assert_ge(gindex, 0, "gindex must be non-negative");
      boba_always_assert_equal(mindex3, mindex3_2, "Triangular multi-index not working");
    }
    else
    {
      return;
    }
  });

  // Permutation testing for 3D
  ::boba::Array<size_t, 3> array1_3d{2, 1, 0};
  test_permutations_of_array(array1_3d, smid3, static_cast<size_t>(6));

  ::boba::Array<size_t, 3> array2_3d{2, 1, 1};
  test_permutations_of_array(array2_3d, smid3, static_cast<size_t>(3));

  ::boba::Array<size_t, 3> array3_3d{2, 2, 0};
  test_permutations_of_array(array3_3d, smid3, static_cast<size_t>(3));

  ::boba::Array<size_t, 3> array4_3d{1, 1, 1};
  test_permutations_of_array(array4_3d, smid3, static_cast<size_t>(1));

  boba_print("All three-dimensional assertions passed");

  /*** Check higher-dimensional simplicial multi-index functionality ***/
  boba_print(" ");
  std::cout << "Finally, testing in " << big_dim << " dimensions." << std::endl;
  ::boba::SimplicialMultiindexer<big_dim> smid_big(n);
  std::cout << "Size per dimension = " << smid_big.size_per_dimension() << ". Total size = " << smid_big.total_size() << std::endl;
  // Here we loop over the global index to avoid high-dimensional loops
  ::boba::loop<space, 1>(smid_big.total_size(),
                         [=] __boba_host_device__(size_t gindex_big)
  {
    auto mind_big = smid_big.multiindex(gindex_big);
    auto gindex_big2 = smid_big.index(mind_big);

    if (verbose)
    {
      printf("gindex_big = %lu\n", gindex_big);
      printf("gindex_big2 = %lu\n", gindex_big2);
    }

    boba_always_assert_equal(gindex_big, gindex_big2, "High-dimensional simplicial multi-index not working")
  });

  ::boba::Array<size_t, big_dim> array1_big;
  for (size_t i = 0; i < big_dim; i++)
  {
    array1_big[i] = big_dim - 1 - i;
  }
  test_permutations_of_array(array1_big, smid_big, static_cast<size_t>(boba::factorial(big_dim)));

  std::cout << "All " << big_dim << "-dimensional assertions passed" << std::endl;

  checkpoint();

  boba::finalize();

  return final_check(check);
}
