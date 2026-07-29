// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

/*
  Test BoBa's orthogonalize functions
*/

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

template <typename train_t>
void check_train_orthogonality(bool& check, train_t const& ortho_check)
{
  auto dimension = ortho_check.get_number_cores();
  for (size_t d = 0; d < dimension - 1; d++)
  {
    auto rl2 = ortho_check.get_ranks_left(d);
    auto rr2 = ortho_check.get_ranks_right(d);

    auto rl = boba::sqrt(rl2);
    auto rr = boba::sqrt(rr2);

    boba::Multiindexer<2> left_rank({rl, rl});
    boba::Multiindexer<2> right_rank({rr, rr});

    auto check_view = ortho_check.cores[d].const_view();

    double error = 0.0;
    ::boba::max_reduce<space>(error, 0_z, right_rank.size(), [=] __boba_host_device__(size_t ir, boba::max_reducer_operator<double>& local_error)
    {
      auto value = 0.0;
      for (size_t il = 0; il < left_rank.size(); il++)
      {
        auto [il1, il2] = left_rank.multiindex(il);
        if (il1 == il2)
        {
          value += check_view({il, 0, ir});
        }
      }
      auto [ir1, ir2] = right_rank.multiindex(ir);
      bool is_right_rank_match = ir1 == ir2;
      auto expected_value = (is_right_rank_match) ? 1.0 : 0.0;
      local_error.max(boba::abs(expected_value - value));
    });

    pass_or_fail(check, error, 1.0e-09);
  }
}

template <typename tucker_t>
void check_tucker_orthogonality(bool& check, tucker_t const& ortho_check)
{
  auto dimension = ortho_check.get_number_cores();
  for (size_t d = 0; d < dimension - 1; d++)
  {
    auto r2 = ortho_check.get_ranks(d);
    auto r = boba::sqrt(r2);

    boba::Multiindexer<2> ranks({r, r});

    auto check_view = ortho_check.cores[d].const_view();

    double error = 0.0;
    ::boba::max_reduce<space>(error, 0_z, ranks.size(), [=] __boba_host_device__(size_t i, boba::max_reducer_operator<double>& local_error)
    {
      auto [ir1, ir2] = ranks.multiindex(i);
      auto value = check_view({0, i});
      bool is_rank_match = ir1 == ir2;
      auto expected_value = is_rank_match ? 1.0 : 0.0;
      local_error.max(boba::abs(expected_value - value));
    });

    pass_or_fail(check, error, 1.0e-09);
  }
}

template <typename matrix_t>
double orthogonality_error(matrix_t const& gram)
{
  boba::Matrix<space, double> identity({gram.sizes(0), gram.sizes(1)});
  identity.set_to_identity_matrix();
  return boba::norm_difference_inf(gram, identity);
}

/*
  Tests orthogonalize procedure
*/

int main(int argc, char* argv[])
{
  boba::splash();
  boba::init();
  boba_print("Tests orthogonalize procedure");

  BOBA_CALI_EXTERNAL_MARK

  size_t resolution = 3;

  checkpoint();
  ::boba::argparser args(argc, argv);

  args.add_optional_argument(resolution,
                             "-r",
                             "--resolution",
                             "Tensor sizes.");

  args.parse_check();

  // Common setup for the experiments
  constexpr size_t dimension = 3;
  bool check = true;

  auto sizes = boba::filled_array<dimension>(resolution);
  for (size_t d = 0; d < dimension; d++)
  {
    sizes[d] += d;
  }

  //
  // Tensor train orthogonalize test
  //
  {
    // Generate a tt
    checkpoint();
    boba::TensorTrain<dimension, space, double> tt(sizes);
    auto R1 = 2_z;
    auto R2 = 3_z;

    tt.cores[0].resize({1, sizes[0], R1});
    tt.cores[1].resize({R1, sizes[1], R2});
    tt.cores[2].resize({R2, sizes[2], 1});
    for (size_t d = 0; d < dimension; d++)
    {
      tt.cores[d].fill_with_random();
    }

    // Orthogonalized tt should have the property we are testing

    checkpoint();
    tt.orthogonalize();

    // "Transpose" the orthogonal tt into a ttm
    auto ones = boba::filled_array<3>(1_z);
    boba::TensorTrainMatrix<dimension, space, double> ttm(ones, sizes);

    auto new_R1 = tt.get_ranks_right(0);
    auto new_R2 = tt.get_ranks_right(1);

    ttm.cores[0].resize({1, 1, tt.sizes(0), new_R1});
    ttm.cores[1].resize({new_R1, 1, tt.sizes(1), new_R2});
    ttm.cores[2].resize({new_R2, 1, tt.sizes(2), 1});

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      ttm.cores[d].reshape(tt.cores[d]);
    }

    // tt^T * tt should have some known structure due to orthogonality

    checkpoint();
    auto ortho_check = ttm * tt;

    checkpoint();
    check_train_orthogonality(check, ortho_check);
  }

  //
  // quantized tensor train orthogonalize test
  //
  {
    checkpoint();
    size_t base = 2;
    size_t exponent = 4;
    boba::QuantizedTensorTrain<space, double> qtt(base, exponent);
    auto R1 = 2_z;
    auto R2 = 3_z;
    auto R3 = 2_z;

    qtt.cores[0].resize({1, base, R1});
    qtt.cores[1].resize({R1, base, R2});
    qtt.cores[2].resize({R2, base, R3});
    qtt.cores[3].resize({R3, base, 1});
    for (size_t d = 0; d < exponent; d++)
    {
      qtt.cores[d].fill_with_random();
    }

    checkpoint();
    qtt.orthogonalize();

    boba::QuantizedTensorTrainMatrix<space, double> qttm(1, base, exponent);

    auto new_R1 = qtt.get_ranks_right(0);
    auto new_R2 = qtt.get_ranks_right(1);
    auto new_R3 = qtt.get_ranks_right(2);

    qttm.cores[0].resize({1, 1, base, new_R1});
    qttm.cores[1].resize({new_R1, 1, base, new_R2});
    qttm.cores[2].resize({new_R2, 1, base, new_R3});
    qttm.cores[3].resize({new_R3, 1, base, 1});

    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {
      qttm.cores[d].reshape(qtt.cores[d]);
    }

    checkpoint();
    auto ortho_check = qttm * qtt;

    checkpoint();
    check_train_orthogonality(check, ortho_check);
  }

  //
  // Quantized Tensor Train Matrix Test
  // Since this is only orthogonalization (not rounding), the dense matrix should not change.
  // So we will be testing whether a randomly created QTTM preserves the dense matrix structure
  // under orthogonalization. We will also check if the Frobenius norm of the QTTM is also preserved.
  //
  {
    // Generate a QTTM
    checkpoint();
    size_t base = 2;
    size_t exponent = 4;
    boba::QuantizedTensorTrainMatrix<space, double> QTTM_Example(base, base, exponent);

    // To make the QTTM more interesting, we will give it random ranks.
    auto new_R1 = 3_z;
    auto new_R2 = 4_z;
    auto new_R3 = 3_z;

    QTTM_Example.cores[0].resize({1, base, base, new_R1});
    QTTM_Example.cores[1].resize({new_R1, base, base, new_R2});
    QTTM_Example.cores[2].resize({new_R2, base, base, new_R3});
    QTTM_Example.cores[3].resize({new_R3, base, base, 1});

    // We will now fill the cores of the QTTM with random entries.
    checkpoint();
    for (size_t d = 0; d < exponent; d++)
    {
      QTTM_Example.cores[d].fill_with_random();
    }

    // First we will decompress the random QTTM and store that matrix.
    auto old_Matrix = QTTM_Example.decompress();

    // We will also compute the Frobenius norm of the QTTM before orthogonalization.
    auto old_QTTM_norm = ::boba::norm_frobenius(QTTM_Example);

    // Now, we orthogonalize.
    QTTM_Example.orthogonalize();

    // Next, we decompress the orthogonalized QTTM. This should not change from the old matrix.
    auto new_Matrix = QTTM_Example.decompress();

    // We will also compute the Frobenius norm of the orthogonalized QTTM.
    auto new_QTTM_norm = ::boba::norm_frobenius(QTTM_Example);

    // Here, we compute the relative error of the change in the matrix and Frobenius norm.
    auto MatrixDiff = old_Matrix - new_Matrix;
    auto relError = ::boba::norm_frobenius(MatrixDiff) / ::boba::norm_frobenius(old_Matrix);
    auto normPreserve = boba::abs(old_QTTM_norm - new_QTTM_norm);

    // Lastly, we check if orthogonalization works.
    std::cout << "\nQTT-matrix orthogonalize() test:\n";
    pass_or_fail(check, relError, 1.0e-13);
    pass_or_fail(check, normPreserve, 1.0e-13);
    std::cout << std::endl;

    checkpoint();
  }

  //
  // tucker orthogonalize test
  //
  {
    // Generate a tucker
    checkpoint();
    boba::Tucker<dimension, space, double> tuck(sizes);
    auto R1 = 2_z;
    auto R2 = 3_z;
    auto R3 = 4_z;

    tuck.cores[0].resize({sizes[0], R1});
    tuck.cores[1].resize({sizes[1], R2});
    tuck.cores[2].resize({sizes[2], R3});
    tuck.R_core.resize({R1, R2, R3});
    for (size_t d = 0; d < dimension; d++)
    {
      tuck.cores[d].fill_with_random();
      tuck.R_core.fill_with_random();
    }

    // Orthogonalized tuck should have the property we are testing

    checkpoint();
    tuck.orthogonalize();

    // "Transpose" the orthogonal tuck into a tuck matrix
    auto ones = boba::filled_array<3>(1_z);
    boba::TuckerMatrix<dimension, space, double> tuckmat(ones, sizes);

    auto new_R1 = tuck.get_ranks(0);
    auto new_R2 = tuck.get_ranks(1);
    auto new_R3 = tuck.get_ranks(2);

    tuckmat.cores[0].resize({1, tuck.sizes(0), new_R1});
    tuckmat.cores[1].resize({1, tuck.sizes(1), new_R2});
    tuckmat.cores[2].resize({1, tuck.sizes(2), new_R3});
    tuckmat.R_core.resize({new_R1, new_R2, new_R3});

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      tuckmat.cores[d].reshape(tuck.cores[d]);
    }
    tuckmat.R_core.reshape(tuck.R_core);

    // tuckmat^T * tuck should have some known structure due to orthogonality

    checkpoint();
    auto ortho_check = tuckmat * tuck;

    checkpoint();
    check_tucker_orthogonality(check, ortho_check);
  }

  //
  // HierarchicalTucker orthogonalize test
  //
  {
    constexpr size_t ht_dimension = 4;

    checkpoint();
    auto dim_tree = boba::DimensionTree(boba::BalancedTreeBuilder(ht_dimension));
    auto ht_sizes = boba::filled_array<ht_dimension>(12_z);
    boba::HierarchicalTucker<ht_dimension, space, double> ht(ht_sizes, dim_tree);
    ht.fill_with_random();

    auto dense_before = ht.decompress();
    pass_or_fail_bool(check, !ht.get_is_orthog());

    checkpoint();
    ht.orthogonalize();

    pass_or_fail_bool(check, ht.get_is_orthog());

    auto dense_after = ht.decompress();
    pass_or_fail(check, boba::norm_difference_inf(dense_before, dense_after), 1.0e-09);

    const auto& is_leaf = dim_tree.get_is_leaf();
    for (size_t node = 1; node < ht.get_num_nodes(); ++node)
    {
      if (is_leaf[node])
      {
        auto basis = ht.get_basis_matrix(node);
        auto gram = basis.transpose() * basis;
        pass_or_fail(check, orthogonality_error(gram), 1.0e-09);
      }
      else
      {
        auto transfer = ht.get_transfer_tensor(node);
        auto unfolding = boba::unfold(transfer, std::vector<size_t>{0, 1}, std::vector<size_t>{2});
        auto gram = unfolding.transpose() * unfolding;
        pass_or_fail(check, orthogonality_error(gram), 1.0e-09);
      }
    }
  }

  boba::finalize();
  return final_check(check);
}
