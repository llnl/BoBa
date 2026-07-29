// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off
#include "../tests/common.hpp"
#include <cstdlib>

/*
  Demonstrates some basics regarding canonical polyadic decompositions
  Here we assume you have been introduced to the basics of CP decompositions
*/

constexpr boba::execution_space space = boba::default_execution_space;

int main(int argc, char** argv) {

  boba::splash();
  boba::init();

  bool check = 1;

  const double user_tol = 1.0e-3;
  const size_t hardcoded_cp_rank = 3;

  auto make_hardcoded_tensor = []() {
    ::boba::Array<size_t, 3> tensor_sizes{4, 4, 4};
    boba::Tensor<3, space, double> tensor(tensor_sizes);
    auto tensor_view = tensor.view();

    // This hardcoded tensor is a sum of three separable terms and has low CP rank.
    ::boba::loop<space, 3>(tensor.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
      {
        double x = static_cast<double>(mid[0] + 1);
        double z = static_cast<double>(mid[2] + 1);
        tensor_view(mid) = x*z + x*x*z*z + x*x*x*z*z*z;
      });

    return tensor;
  };

  checkpoint();
  {
    boba_print("Canonical polyadic decomposition from a 3-tensor");

    // Let's fill a 10x10x10 tensor with ones.
    auto sizes = ::boba::filled_array<3, size_t>(10);
    boba::Tensor<3, space, double> tensor_A(sizes);
    tensor_A.rename("ones");
    tensor_A.fill_with(1.0);

    // Instantiate a CP decomposition and compress the dense tensor.
    ::boba::CanonicalPolyadicDecomposition<3, space, double> cp_A(tensor_A.sizes());
    cp_A.compress(tensor_A, 1);

    // The tensor extents are preserved, and the constant tensor is rank one.
    boba_print(tensor_A.sizes());
    boba_print(tensor_A.number_nonzeros());
    boba_print(cp_A.sizes());
    boba_print(cp_A.compression_rate());
    boba_print(cp_A.get_rank());
    boba_print(tensor_A.get_number_elements());
    boba_print(cp_A.get_number_elements());

    boba_print("We also could have used the special fill routine for constants");

    ::boba::CanonicalPolyadicDecomposition<3, space, double> cp_A2(tensor_A.sizes());
    cp_A2.fill_with(1.0);

    boba_print(cp_A2.sizes());
    boba_print(cp_A2.compression_rate());
    boba_print(cp_A2.get_rank());
    boba_print(cp_A2.get_number_elements());

    // The two CP representations should agree.
    auto error_norm = ::boba::norm_difference_frobenius(cp_A.decompress(), cp_A2.decompress());
    pass_or_fail(check, error_norm, 1.0e-4);
  }

  checkpoint();
  {
    boba_print("\n CP decomposition from a hardcoded tensor");

    if(hardcoded_cp_rank == 0)
    {
      boba_error("The hardcoded CP rank must be positive.");
    }

    auto hardcoded_tensor = make_hardcoded_tensor();
    ::boba::CanonicalPolyadicDecomposition<3, space, double> cp_from_tensor(hardcoded_tensor.sizes());
    cp_from_tensor.ALS_tolerance_relative = 0.0;
    cp_from_tensor.compress(hardcoded_tensor, hardcoded_cp_rank);

    auto tensor_uncompressed = cp_from_tensor.decompress();
    auto reconstruction_error_normalized_weights =
      ::boba::norm_difference_inf(tensor_uncompressed, hardcoded_tensor);

    boba_print("hardcoded tensor");
    boba_print("requested CP rank");
    boba_print(hardcoded_cp_rank);
    boba_print(hardcoded_tensor.sizes());
    boba_print(hardcoded_tensor.number_nonzeros());
    boba_print(cp_from_tensor.get_rank());
    boba_print(cp_from_tensor.compression_rate());
    boba_print(cp_from_tensor.weights());
    cp_from_tensor.print();
    boba_print(reconstruction_error_normalized_weights);
  }

  checkpoint();
  {
    boba_print("\n Optimized CP rank search");

    auto tensor_A = make_hardcoded_tensor();

    auto reference_norm = ::boba::norm_frobenius(tensor_A);

    auto compute_cp_error = [&](size_t rank) {
      ::boba::CanonicalPolyadicDecomposition<3, space, double> cp_trial(tensor_A.sizes());
      cp_trial.ALS_tolerance_relative = 0.0;
      cp_trial.compress(tensor_A, rank);

      return ::boba::norm_difference_frobenius(cp_trial.decompress(), tensor_A)/reference_norm;
    };

    size_t lower_rank = 0;
    size_t upper_rank = 1;
    double upper_error = compute_cp_error(upper_rank);
    const size_t max_search_rank = tensor_A.get_number_elements();

    while(upper_error > user_tol && upper_rank < max_search_rank)
    {
      lower_rank = upper_rank;
      upper_rank = (2*upper_rank < max_search_rank) ? 2*upper_rank : max_search_rank;
      upper_error = compute_cp_error(upper_rank);
    }

    if(upper_error <= user_tol)
    {
      while(upper_rank - lower_rank > 1)
      {
        size_t mid_rank = lower_rank + (upper_rank - lower_rank)/2;
        double mid_error = compute_cp_error(mid_rank);
        if(mid_error <= user_tol)
        {
          upper_rank = mid_rank;
          upper_error = mid_error;
        }
        else
        {
          lower_rank = mid_rank;
        }
      }
    }

    boba_print("CP error");
    boba_print(upper_error);
    boba_print("CP tolerance");
    boba_print(user_tol);
    boba_print("CP rank");
    boba_print(upper_rank);
  }

  checkpoint();
  {
    boba_print("\n Sparse vs Low Rank example");

    // Construct the identity matrix as a dense tensor.
    auto sizes = ::boba::filled_array<2, size_t>(10);
    boba::Tensor<2, space, double> tensor_A(sizes);
    auto tensor_A_view = tensor_A.view();

    ::boba::loop<space, 2>(tensor_A.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 2> mid)
      {
        size_t i = mid[0];
        size_t j = mid[1];
        tensor_A_view(mid) = (i == j) ? 1.0 : 0.0;
      });

    boba_print(tensor_A.size());
    boba_print(tensor_A.number_nonzeros());

    ::boba::CanonicalPolyadicDecomposition<2, space, double> cp_A(tensor_A.sizes());
    cp_A.compress(tensor_A, sizes[0]);

    // Sparse structure alone does not imply low CP rank.
    boba_print(tensor_A.sizes());
    boba_print(cp_A.sizes());
    boba_print(tensor_A.get_number_elements());
    boba_print(cp_A.get_number_elements());
    boba_print(cp_A.compression_rate());
    boba_print(cp_A.get_rank());
  }

  checkpoint();
  {
    boba_print("\n Sum of rank-one terms");

    // f(x, y, z) = x*z + x^2*z^2 + x^3*z^3 does not depend on y and is a sum
    // of three separable terms, making it a natural CP example.
    auto sizes = ::boba::filled_array<3, size_t>(10);
    boba::Tensor<3, space, double> tensor_A(sizes);
    auto tensor_A_view = tensor_A.view();

    ::boba::loop<space, 3>(tensor_A.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 3> mid)
      {
        size_t i = mid[0];
        double x = double(i)/double(tensor_A_view.sizes(0));
        size_t k = mid[2];
        double z = double(k)/double(tensor_A_view.sizes(2));
        tensor_A_view(mid) = x*z + x*x*z*z + x*x*x*z*z*z;
      });

    ::boba::CanonicalPolyadicDecomposition<3, space, double> cp_rank3(tensor_A.sizes());
    cp_rank3.ALS_tolerance_relative = 0.0;
    cp_rank3.compress(tensor_A, 3);

    auto relative_error = ::boba::norm_difference_frobenius(cp_rank3.decompress(), tensor_A)/::boba::norm_frobenius(tensor_A);

    boba_print(tensor_A.get_number_elements());
    boba_print(cp_rank3.get_number_elements());
    boba_print(cp_rank3.compression_rate());
    boba_print(cp_rank3.get_rank());
    boba_print("rank-3 relative error");
    boba_print(relative_error);

    pass_or_fail(check, relative_error, 1.0e-4);
  }

  boba::finalize();
  return final_check(check);
}
// clang-format on
