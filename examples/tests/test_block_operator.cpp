// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../tests/common_ttm.hpp"
#include "common.hpp"

/*
  This test demonstrates how to work with block operators and vectors.
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::execution_space host_space = ::boba::host_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

template <size_t blocks, size_t dimension>
void test_tt_blocks(bool& check, const size_t N)
{

  using tt_t = ::boba::TensorTrain<dimension, host_space, double>;
  using ttm_t = ::boba::TensorTrainMatrix<dimension, host_space, double>;
  using block_tt_t = boba::BlockVector<tt_t>;
  using block_ttm_t = boba::BlockOperator<ttm_t>;

  auto full_sizes = boba::filled_array<dimension>(blocks * N);
  auto block_sizes = boba::filled_array<dimension>(N);

  // ------------------------------------------------------------------------
  //   Test 1:  full to block to full
  // ------------------------------------------------------------------------
  {
    // Note that using the matvec from the full operator will be genrally incorrect in ths case
    // This is because copy_vector_into_block_form makes no assumption about the validitidy of decomposing the full vector into its pieces
    // A great deal of caution should be used when thinking about block operators!
    //

    // ------------------------------------------------------------------------
    //   Make full objects
    // ------------------------------------------------------------------------
    //
    // Make full tensor train
    //
    tt_t full_tt(full_sizes);
    {
      full_tt.fill_with_zeros();
      for (size_t d_term = 0; d_term < 1; d_term++)
      {
        boba::Array<size_t, dimension + 1> ranks;
        ranks[0] = 1;
        for (size_t d_core = 0; d_core < dimension; d_core++)
        {
          ranks[d_core] = 1 + d_core + d_term;
        }
        ranks[dimension] = 1;

        tt_t term_tt(full_sizes);
        for (size_t d_core = 0; d_core < dimension; d_core++)
        {
          term_tt.cores[d_core].resize({ranks[d_core], full_sizes[d_core], ranks[d_core + 1]});
          term_tt.cores[d_core].fill_with_random();
        }
        full_tt = term_tt;
      }
    }

    //
    // Make full tensor train matrix
    //
    ttm_t full_ttm(full_sizes, full_sizes);
    {
      full_ttm.fill_with_zeros();
      for (size_t d_term = 0; d_term < 1; d_term++)
      {
        boba::Array<size_t, dimension + 1> ranks;
        ranks[0] = 1;
        for (size_t d_core = 0; d_core < dimension; d_core++)
        {
          ranks[d_core] = 1 + d_core + d_term;
        }
        ranks[dimension] = 1;

        ttm_t term_ttm(full_sizes, full_sizes);
        for (size_t d_core = 0; d_core < dimension; d_core++)
        {
          term_ttm.cores[d_core].resize({ranks[d_core], full_sizes[d_core], full_sizes[d_core], ranks[1 + d_core]});
          term_ttm.cores[d_core].fill_with_random();
        }
        full_ttm = term_ttm;
      }
    }

    // ------------------------------------------------------------------------
    //   Make block objects from full objects
    // ------------------------------------------------------------------------
    //
    // Make block tt from full tt
    //
    block_tt_t block_tt(blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        block_tt(d_row).resize(block_sizes);
      }
      boba::copy_vector_into_block_form(block_tt, full_tt);
    }

    //
    // Make block matrix from full matrix
    //
    block_ttm_t block_ttm(blocks, blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        for (size_t d_col = 0; d_col < blocks; d_col++)
        {
          block_ttm({d_row, d_col}).resize(block_sizes, block_sizes);
        }
      }
      boba::copy_operator_into_block_form(block_ttm, full_ttm);
    }

    // ------------------------------------------------------------------------
    //   Make full objects from block objects
    // ------------------------------------------------------------------------
    ttm_t unblocked_ttm = boba::unblock(block_ttm);
    tt_t unblocked_tt = boba::unblock(block_tt);

    // ------------------------------------------------------------------------
    //   Verify
    // ------------------------------------------------------------------------
    {
      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_tt, full_tt), 1.0e-10);
      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_ttm, full_ttm), 1.0e-10);

      tt_t unblocked_matvec = unblocked_ttm * unblocked_tt;
      tt_t full_matvec = full_ttm * full_tt;

      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_matvec, full_matvec), 1.0e-10);

      /*
      This matvec will be incorrect!

      block_tt_t block_matvec = block_ttm*block_tt;
      tt_t unblocked_block_matvec = boba::unblock(block_matvec);

      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_block_matvec, full_matvec), 1.0e-10);
      */
    }
  }

  // ------------------------------------------------------------------------
  //   Test 2:  block to full to block with matvec consistency check
  // ------------------------------------------------------------------------
  {
    // In this test, the full operators are created from block operators, so consistency is guaranteed

    // ------------------------------------------------------------------------
    //   Make block objects from full objects
    // ------------------------------------------------------------------------
    //
    // Make block tt
    //
    block_tt_t block_tt(blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        block_tt(d_row).resize(block_sizes);
        block_tt(d_row).fill_with_zeros();
        for (size_t d_term = 0; d_term < 1; d_term++)
        {
          tt_t temp(block_sizes);
          boba::Array<size_t, dimension + 1> ranks;
          ranks[0] = 1;
          for (size_t d_core = 0; d_core < dimension; d_core++)
          {
            ranks[d_core] = 1 + d_core + d_term;
          }
          ranks[dimension] = 1;
          for (size_t d_core = 0; d_core < 0; d_core++)
          {
            temp.cores[d_core].resize({ranks[d_core], full_sizes[d_core], ranks[1 + d_core]});
            temp.cores[d_core].fill_with_random();
          }
          block_tt(d_row) += temp;
        }
      }
    }

    //
    // Make block matrix
    //
    block_ttm_t block_ttm(blocks, blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        for (size_t d_col = 0; d_col < blocks; d_col++)
        {
          block_ttm({d_row, d_col}).resize(block_sizes, block_sizes);
          for (size_t d_term = 0; d_term < 1; d_term++)
          {
            ttm_t temp(block_sizes, block_sizes);

            boba::Array<size_t, dimension + 1> ranks;
            ranks[0] = 1;
            for (size_t d_core = 0; d_core < dimension; d_core++)
            {
              ranks[d_core] = 1 + d_core + d_term;
            }
            ranks[dimension] = 1;
            for (size_t d_core = 0; d_core < 0; d_core++)
            {
              temp.cores[d_core].resize({ranks[d_core], full_sizes[d_core], full_sizes[d_core], ranks[1 + d_core]});
              temp.cores[d_core].fill_with_random();
            }
            block_ttm({d_row, d_col}) += temp;
          }
        }
      }
    }

    // ------------------------------------------------------------------------
    //   Make full objects from block objects
    // ------------------------------------------------------------------------
    ttm_t unblocked_ttm = boba::unblock(block_ttm);
    tt_t unblocked_tt = boba::unblock(block_tt);

    // ------------------------------------------------------------------------
    //   Make block objects from full
    // ------------------------------------------------------------------------
    //
    // Make block tt from full tt
    //
    block_tt_t block_from_full_tt(blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        block_from_full_tt(d_row).resize(block_sizes);
      }
      boba::copy_vector_into_block_form(block_from_full_tt, unblocked_tt);
    }

    //
    // Make block matrix from full matrix
    //
    block_ttm_t block_from_full_ttm(blocks, blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        for (size_t d_col = 0; d_col < blocks; d_col++)
        {
          block_from_full_ttm({d_row, d_col}).resize(block_sizes, block_sizes);
        }
      }
      boba::copy_operator_into_block_form(block_from_full_ttm, unblocked_ttm);
    }

    // ------------------------------------------------------------------------
    //   Make block objects from full
    // ------------------------------------------------------------------------
    tt_t unblocked_2_tt = boba::unblock(block_from_full_tt);
    ttm_t unblocked_2_ttm = boba::unblock(block_from_full_ttm);

    // ------------------------------------------------------------------------
    //   Verify
    // ------------------------------------------------------------------------
    {
      pass_or_fail(check, boba::norm_difference_frobenius(block_from_full_tt, block_tt), 1.0e-10);
      pass_or_fail(check, boba::norm_difference_frobenius(block_from_full_ttm, block_ttm), 1.0e-10);

      block_tt_t blocked_matvec = block_ttm * block_tt;
      block_tt_t blocked_from_full_matvec = block_from_full_ttm * block_from_full_tt;
      tt_t unblocked_matvec = unblocked_ttm * unblocked_tt;
      tt_t unblocked_2_matvec = unblocked_2_ttm * unblocked_2_tt;

      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_matvec, unblocked_2_matvec), 1.0e-10);
      pass_or_fail(check, boba::norm_difference_frobenius(blocked_matvec, blocked_from_full_matvec), 1.0e-10);
    }
  }
}

//
// Test block matrices and block vectors
//

template <size_t blocks>
void test_matrix_blocks(bool& check, const size_t N)
{

  using vector_t = boba::Vector<space, double>;
  using matrix_t = boba::Matrix<space, double>;
  using block_vector_t = boba::BlockVector<vector_t>;
  using block_matrix_t = boba::BlockOperator<matrix_t>;

  auto full_sizes = blocks * N;
  auto block_sizes = N;

  // ------------------------------------------------------------------------
  //   Test 1:  block to full to block with matvec consistency check
  // ------------------------------------------------------------------------
  {
    // In this test, the full operators are created from block operators, so consistency is guaranteed

    // ------------------------------------------------------------------------
    //   Make block objects from full objects
    // ------------------------------------------------------------------------
    //
    // Make block tt
    //
    block_vector_t block_tt(blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        block_tt(d_row).resize(block_sizes);
        block_tt(d_row).fill_with_random();
      }
    }

    //
    // Make block matrix
    //
    block_matrix_t block_ttm(blocks, blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        for (size_t d_col = 0; d_col < blocks; d_col++)
        {
          block_ttm({d_row, d_col}).resize({block_sizes, block_sizes});
          block_ttm({d_row, d_col}).fill_with_random();
        }
      }
    }

    // ------------------------------------------------------------------------
    //   Make full objects from block objects
    // ------------------------------------------------------------------------
    matrix_t unblocked_ttm = boba::unblock(block_ttm);
    vector_t unblocked_tt = boba::unblock(block_tt);

    // ------------------------------------------------------------------------
    //   Make block objects from full
    // ------------------------------------------------------------------------
    //
    // Make block tt from full tt
    //
    block_vector_t block_from_full_tt(blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        block_from_full_tt(d_row).resize(block_sizes);
      }
      boba::copy_vector_into_block_form(block_from_full_tt, unblocked_tt);
    }

    //
    // Make block matrix from full matrix
    //
    block_matrix_t block_from_full_ttm(blocks, blocks);
    {
      for (size_t d_row = 0; d_row < blocks; d_row++)
      {
        for (size_t d_col = 0; d_col < blocks; d_col++)
        {
          block_from_full_ttm({d_row, d_col}).resize({block_sizes, block_sizes});
        }
      }
      boba::copy_operator_into_block_form(block_from_full_ttm, unblocked_ttm);
    }

    // ------------------------------------------------------------------------
    //   Make block objects from full
    // ------------------------------------------------------------------------
    vector_t unblocked_2_tt = boba::unblock(block_from_full_tt);
    matrix_t unblocked_2_ttm = boba::unblock(block_from_full_ttm);

    // ------------------------------------------------------------------------
    //   Verify
    // ------------------------------------------------------------------------
    {
      pass_or_fail(check, boba::norm_difference_frobenius(block_from_full_tt, block_tt), 1.0e-10);
      pass_or_fail(check, boba::norm_difference_frobenius(block_from_full_ttm, block_ttm), 1.0e-10);

      block_vector_t blocked_matvec = block_ttm * block_tt;
      block_vector_t blocked_from_full_matvec = block_from_full_ttm * block_from_full_tt;
      vector_t unblocked_matvec = unblocked_ttm * unblocked_tt;
      vector_t unblocked_2_matvec = unblocked_2_ttm * unblocked_2_tt;

      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_matvec, unblocked_2_matvec), 1.0e-10);
      pass_or_fail(check, boba::norm_difference_frobenius(blocked_matvec, blocked_from_full_matvec), 1.0e-10);
    }

    // ------------------------------------------------------------------------
    //   Verify, masked
    // ------------------------------------------------------------------------
    {
      auto ttm_mask = boba::generate_mask(block_ttm);
      auto ttm_full_mask = boba::generate_mask(block_from_full_ttm);

      block_vector_t blocked_matvec = boba::apply_masked(block_ttm, block_tt, ttm_mask);
      block_vector_t blocked_from_full_matvec = boba::apply_masked(block_from_full_ttm, block_from_full_tt, ttm_full_mask);

      pass_or_fail(check, boba::norm_difference_frobenius(blocked_matvec, blocked_from_full_matvec), 1.0e-10);
    }

    // ------------------------------------------------------------------------
    //   Verify, permuted
    // ------------------------------------------------------------------------
    {
      boba::PermutationMatrix<host_space, size_t> permuations_host({block_tt.block_size});
      auto permuations_view = permuations_host.view();
      auto N_perm = permuations_view.size();
      for (size_t i = 0; i < N_perm; i++)
      {
        permuations_view(i) = (N_perm - 1) - i;
      }

      boba::PermutationMatrix<space, size_t> permuations = permuations_host;

      block_vector_t blocked_from_full_matvec = block_from_full_ttm * block_from_full_tt;

      // Insert  I = P * P^T,  A*b = A*I*b = A * P * P^T * b =  (A * P) * (P^T * b)
      {
        auto block_ttm_permute = block_ttm * permuations;
        auto block_tt_permute = permuations.transpose() * block_tt;
        block_vector_t blocked_matvec = block_ttm_permute * block_tt_permute;
        pass_or_fail(check, boba::norm_difference_frobenius(blocked_matvec, blocked_from_full_matvec), 1.0e-10);
      }
      // Insert  I = P^T * P
      {
        auto block_ttm_permute = block_ttm * permuations.transpose();
        auto block_tt_permute = permuations * block_tt;
        block_vector_t blocked_matvec = block_ttm_permute * block_tt_permute;
        pass_or_fail(check, boba::norm_difference_frobenius(blocked_matvec, blocked_from_full_matvec), 1.0e-10);
      }
    }
  }

  // ------------------------------------------------------------------------
  //   Khatri Rao product
  // ------------------------------------------------------------------------
  {
    using host_matrix_t = boba::Matrix<host_space, double>;
    using face_block_matrix_t = boba::BlockOperator<host_matrix_t>;
    using device_face_block_matrix_t = boba::BlockOperator<matrix_t>;

    size_t system_size = 2;

    face_block_matrix_t block_op_A(system_size, system_size);
    {

      block_op_A({0, 0}).resize({2, 2});
      block_op_A({0, 1}).resize({2, 1});
      block_op_A({1, 0}).resize({1, 2});
      block_op_A({1, 1}).resize({1, 1});

      block_op_A({0, 0})({0, 0}) = 1;
      block_op_A({0, 0})({0, 1}) = 2;
      block_op_A({0, 0})({1, 0}) = 4;
      block_op_A({0, 0})({1, 1}) = 5;

      block_op_A({0, 1})({0, 0}) = 3;
      block_op_A({0, 1})({1, 0}) = 6;

      block_op_A({1, 0})({0, 0}) = 7;
      block_op_A({1, 0})({0, 1}) = 8;

      block_op_A({1, 1})({0, 0}) = 9;
    }

    face_block_matrix_t block_op_B(system_size, system_size);
    {
      block_op_B({0, 0}).resize({1, 1});
      block_op_B({0, 1}).resize({1, 2});
      block_op_B({1, 0}).resize({2, 1});
      block_op_B({1, 1}).resize({2, 2});

      block_op_B({0, 0})({0, 0}) = 1;

      block_op_B({0, 1})({0, 0}) = 4;
      block_op_B({0, 1})({0, 1}) = 7;

      block_op_B({1, 0})({0, 0}) = 2;
      block_op_B({1, 0})({1, 0}) = 3;

      block_op_B({1, 1})({0, 0}) = 5;
      block_op_B({1, 1})({0, 1}) = 8;
      block_op_B({1, 1})({1, 0}) = 6;
      block_op_B({1, 1})({1, 1}) = 9;
    }

    face_block_matrix_t block_op_C_check(system_size, system_size);
    {
      block_op_C_check({0, 0}).resize({2, 2});
      block_op_C_check({0, 1}).resize({2, 2});
      block_op_C_check({1, 0}).resize({2, 2});
      block_op_C_check({1, 1}).resize({2, 2});

      block_op_C_check({0, 0})({0, 0}) = 1;
      block_op_C_check({0, 0})({0, 1}) = 2;
      block_op_C_check({0, 0})({1, 0}) = 4;
      block_op_C_check({0, 0})({1, 1}) = 5;

      block_op_C_check({0, 1})({0, 0}) = 12;
      block_op_C_check({0, 1})({0, 1}) = 21;
      block_op_C_check({0, 1})({1, 0}) = 24;
      block_op_C_check({0, 1})({1, 1}) = 42;

      block_op_C_check({1, 0})({0, 0}) = 14;
      block_op_C_check({1, 0})({0, 1}) = 16;
      block_op_C_check({1, 0})({1, 0}) = 21;
      block_op_C_check({1, 0})({1, 1}) = 24;

      block_op_C_check({1, 1})({0, 0}) = 45;
      block_op_C_check({1, 1})({0, 1}) = 72;
      block_op_C_check({1, 1})({1, 0}) = 54;
      block_op_C_check({1, 1})({1, 1}) = 81;
    }

    device_face_block_matrix_t block_op_Ad(system_size, system_size);
    for (size_t row = 0; row < block_op_Ad.rows(); row++)
    {
      for (size_t col = 0; col < block_op_Ad.cols(); col++)
      {
        block_op_Ad({row, col}) = block_op_A({row, col});
      }
    }
    device_face_block_matrix_t block_op_Bd(system_size, system_size);
    for (size_t row = 0; row < block_op_Bd.rows(); row++)
    {
      for (size_t col = 0; col < block_op_Bd.cols(); col++)
      {
        block_op_Bd({row, col}) = block_op_B({row, col});
      }
    }
    device_face_block_matrix_t block_op_Cd_check(system_size, system_size);
    for (size_t row = 0; row < block_op_Cd_check.rows(); row++)
    {
      for (size_t col = 0; col < block_op_Cd_check.cols(); col++)
      {
        block_op_Cd_check({row, col}) = block_op_C_check({row, col});
      }
    }

    auto block_op_Cd = boba::khatri_rao(block_op_Ad, block_op_Bd);

    // ------------------------------------------------------------------------
    //   Verify
    // ------------------------------------------------------------------------
    {
      auto unblocked_C = boba::unblock(block_op_Cd);
      auto unblocked_C_check = boba::unblock(block_op_Cd_check);
      pass_or_fail(check, boba::norm_difference_frobenius(unblocked_C, unblocked_C_check), 1.0e-10);
    }
  }
}

int main(int argc, char* argv[])
{

  boba::detail::ignore(argc);
  boba::detail::ignore(argv);
  boba::splash();
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;

  size_t N = 2;
  static constexpr size_t dimension = 2;
  constexpr size_t blocks = 2;

  test_tt_blocks<blocks, dimension>(check, N);

  test_matrix_blocks<blocks>(check, N);

  boba::finalize();
  return final_check(check);
}
