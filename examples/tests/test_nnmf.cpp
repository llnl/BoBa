// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

/*
  Tests BoBa's implementation of a nonnegative matrix factorization
*/

inline void setup_matrices(
  size_t which,
  boba::Matrix<space, double>& test_matrix)
{
  switch (which)
  {
  // ---------------------------------
  case 0:
    // ---------------------------------
    {
      test_matrix.fill_with(1.0);
      test_matrix.rename("ones");
      return;
    }
  // ---------------------------------
  case 1:
    // ---------------------------------
    {
      test_matrix.set_to_identity_matrix();
      test_matrix.rename("identity");
      return;
    }
  // ---------------------------------
  case 2:
    // ---------------------------------
    {
      test_matrix.rename("checkerboard");
      test_matrix.fill_with(-1.0);
      auto test_matrix_view = test_matrix.view();
      ::boba::loop<space, 2>(test_matrix.sizes(),
                             [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
      {
        if (::boba::mod(ij[0] + ij[1], 2) == 0)
        {
          test_matrix_view(ij) = 1.0;
        }
      });
      return;
    }
  // ---------------------------------
  default:
    boba_error("Invalid matrix choice");
    return;
  }
}

//
// Validate NNMF
//
inline void validate_nnmf(
  ::boba::Matrix<space, double>& test_matrix,
  ::boba::Matrix<space, double>& Wpos,
  ::boba::Matrix<space, double>& Hpos,
  ::boba::Matrix<space, double>& Wneg,
  ::boba::Matrix<space, double>& Hneg,
  bool& check)
{
  auto positive_part = Wpos * Hpos;
  auto negative_part = Wneg * Hneg;
  auto reformed_matrix = positive_part - negative_part;

  bool old_check = check;
  double tolerance = 1.0e-7;
  pass_or_fail(check, boba::norm_difference_inf(reformed_matrix, test_matrix), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(positive_part, test_matrix.nonnegative_part()), tolerance);
  pass_or_fail(check, boba::norm_difference_inf(negative_part, test_matrix.nonpositive_part()), tolerance);
  bool new_fail = not(check) and old_check;

  if (new_fail and (test_matrix.size() < 100))
  {
    Wpos.rename("Wpos");
    Hpos.rename("Hpos");
    Wpos.print();
    Hpos.print();
    Wneg.rename("Wneg");
    Hneg.rename("Hneg");
    Wneg.print();
    Hneg.print();
    reformed_matrix.rename("reformed_matrix");
    reformed_matrix.print();
    test_matrix.rename("test_matrix");
    test_matrix.print();
  }
}

//
//
//

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  boba_print("Tests for NNMF implementations");

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  size_t in_rows = 0;
  size_t in_cols = 0;
  size_t kernel = 0;
  size_t nnmf_scheme = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(in_rows,
                             "-r",
                             "--rows",
                             "Matrix rows.");

  args.add_optional_argument(in_cols,
                             "-c",
                             "--cols",
                             "Matrix columns.");

  args.add_optional_argument(kernel,
                             "-w",
                             "--which",
                             "Which kernel.");

  args.add_optional_argument(nnmf_scheme,
                             "-S",
                             "--scheme",
                             "NNMF scheme: 1. multiplicative updates 2. nnls");

  args.parse_check();

  boba_always_assert_equal((in_cols > 0), (in_rows > 0), "Set neither or both");

  auto space_name = ::boba::detail::execution_space_name(space);

  bool check = true;

  size_t num_kernels = 1;
  if (kernel > 0)
  {
    num_kernels = 1;
  }

  std::vector<size_t> rows_list = {5, 10, 15, 20, 50, 100};
  std::vector<size_t> cols_list = {5, 10, 15, 20, 50, 100};

  if (in_cols > 0)
  {
    rows_list.resize(0);
    rows_list.push_back(in_rows);
    cols_list.resize(0);
    cols_list.push_back(in_cols);
  }

  for (auto rows : rows_list)
    for (auto cols : cols_list)
      for (size_t kernel_id = 0; kernel_id < num_kernels; kernel_id++)
      {
        size_t which = (kernel > 0) ? kernel : kernel_id;

        boba::Matrix<space, double> test_matrix({rows, cols});
        setup_matrices(which, test_matrix);
        boba::Matrix<space, double> validation_matrix = test_matrix;

        std::string decomposition_name = "";

        checkpoint();
        ::boba::TicToc<tictoc_units> timer;
        {
          ::boba::NNMF<space, double> nnmf;
          if (nnmf_scheme == 1)
          {
            nnmf.update_scheme = boba::NNMF<space, double>::update_schemes::nonnegative_least_squares;
            decomposition_name = "nnmf, nnls  ";
          }
          else
          {
            decomposition_name = "nnmf, mult  ";
          }

          auto test_matrix_pos = test_matrix.nonnegative_part();
          auto test_matrix_neg = test_matrix.nonpositive_part();

          nnmf(test_matrix_pos);
          auto Wpos = nnmf.W;
          auto Hpos = nnmf.H;

          nnmf(test_matrix_neg);
          auto Wneg = nnmf.W;
          auto Hneg = nnmf.H;

          validate_nnmf(validation_matrix, Wpos, Hpos, Wneg, Hneg, check);
        }
        timer.end();

        if (decomposition_name.length() > 0)
        {
          std::cout << decomposition_name
                    << space_name << " "
                    << test_matrix.name() << " "
                    << rows << " "
                    << cols << " "
                    << which << " "
                    << timer.duration << " "
                    << timer.units_string << std::endl;
        }
        else
        {
          std::cout << "no_data" << std::endl;
        }
      }

  boba::finalize();
  return final_check(check);
}
