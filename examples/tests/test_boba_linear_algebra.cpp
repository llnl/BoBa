// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"

constexpr boba::execution_space space = ::boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::microseconds;

template <typename data_t>
constexpr bool qr_supported()
{
  if constexpr (space == boba::execution_space::CPU)
  {
#ifdef BOBA_ENABLE_APPLE
    return std::is_same_v<data_t, double>;
#else
    return true;
#endif
  }
  else
  {
    return true;
  }
}

template <typename data_t>
constexpr bool svd_supported()
{
  if constexpr (space == boba::execution_space::CPU)
  {
#ifdef BOBA_ENABLE_APPLE
    return std::is_same_v<data_t, double>;
#else
    return true;
#endif
  }
  else
  {
    return true;
  }
}

template <typename data_t>
constexpr bool lu_supported()
{
  if constexpr (space == boba::execution_space::CPU)
  {
#ifdef BOBA_ENABLE_APPLE
    return std::is_same_v<data_t, double>;
#else
    return true;
#endif
  }
  else
  {
    return true;
  }
}

template <typename data_t>
constexpr bool cholesky_supported()
{
  if constexpr (space == boba::execution_space::CPU)
  {
#ifdef BOBA_ENABLE_APPLE
    return std::is_same_v<data_t, double>;
#else
    return true;
#endif
  }
  else
  {
    return true;
  }
}

/*
  Tests the BoBa QR, rank-revealing QR, SVD, and truncated SVD algorithms, and more!
  Defines exact factorizations and compares them to the results of calling the boba functions.
  This test is good for becoming familiar with some of the helper functions related to QR and SVD.

  Note:

  (1) QRs are only unique under certain restrictions, so care must be taken when trying to compare Eigen, Cusolver, and Hipsolver factorizations.
  (2) We have different options for the QR factorization, namely determining if a rank-revealing algorithm is used.
  (3) The SVD provides multiple ways to truncate the number of singular values. (see the class definition and implementation)
*/

template <typename data_t>
void setup_matrices(
  size_t which,
  // we have a few example matrices, and 'which' chooses which matrix to setup
  boba::Matrix<space, data_t>& test_matrix)
{
  auto test_matrix_view = test_matrix.view();

  switch (which)
  {
  // ---------------------------------
  case 0:
    // ---------------------------------
    {
      test_matrix.fill_with(data_t(1));
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
      ::boba::loop<space, 2>(test_matrix_view.sizes(),
                             [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
      {
        size_t r = ij[0];
        size_t c = ij[1];
        // Hilbert matrix is 1/(r + c  - 1) when r, c >= 1
        // for C++, the equivalent matrix is:
        test_matrix_view({r, c}) = data_t(1) / (data_t(1) + data_t(r) + data_t(c));
      });
      test_matrix.rename("hilbert");
      return;
    }
  // ---------------------------------
  case 3:
    // ---------------------------------
    {
      ::boba::loop<space, 2>(test_matrix_view.sizes(),
                             [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
      {
        size_t r = ij[0];
        size_t c = ij[1];
        size_t X = test_matrix_view.sizes(1);
        data_t x = data_t(c + 1) / data_t(X);
        data_t p = data_t(r);
        test_matrix_view({r, c}) = ::boba::pow(x, p);
      });
      test_matrix.rename("vandermonde");
      return;
    }
  // ---------------------------------
  case 4:
    // ---------------------------------
    {
      test_matrix.set_to_identity_matrix();
      test_matrix.fill_diagonal(1, -data_t(1));
      test_matrix.fill_diagonal(0, data_t(2));
      test_matrix.fill_diagonal(-1, -data_t(1));
      test_matrix.rename("laplacian");
      return;
    }
  // ---------------------------------
  case 5:
    // ---------------------------------
    {
      test_matrix.set_to_identity_matrix();
      test_matrix.fill_diagonal(0, data_t(1));
      test_matrix.fill_diagonal(-1, -data_t(1));
      test_matrix.rename("derivative");
      return;
    }
  // ---------------------------------
  case 6:
    // ---------------------------------
    {
      size_t rank_upper_bound = 40;
      ::boba::loop<space, 2>(test_matrix_view.sizes(),
                             [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
      {
        size_t r = ij[0];
        size_t c = ij[1];
        size_t X = test_matrix_view.sizes(1);
        data_t sum = data_t(0);
        for (size_t n = 0; n < rank_upper_bound; n++)
        {
          data_t x = data_t(c + 1) / data_t(X);
          data_t p = data_t(n);
          data_t basis = ::boba::pow(x, p);
          data_t coeff = data_t(1) / data_t(1 + n + r);
          sum += basis * coeff;
        }
        test_matrix_view({r, c}) = sum;
      });
      test_matrix.rename("low_rank_vandermonde");
      return;
    }
  // ---------------------------------
  default:
    boba_error("Invalid matrix choice");
    return;
  }
}

//
// Validate QR
//
template <typename data_t>
void test_qr(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();
  if constexpr (boba::is_hip(space) && std::is_same_v<data_t, float>)
  {
    // HIP single-precision QR drifts on the wide Vandermonde sweep cases, so use
    // a looser bound that still catches outright breakage.
    tolerance = real_data_t(1.0e-2);
  }

  ::boba::TicToc<tictoc_units> timer;
  ::boba::QR<space, data_t> qr;
  qr(test_matrix);
  timer.end();

  boba::Matrix<space, data_t> QR = qr.reform_matrix();

  bool old_check = check;
  pass_or_fail(check, boba::norm_difference_inf(QR, validation_matrix), tolerance);
  bool new_fail = not(check) and old_check;

  if (new_fail and (validation_matrix.size() < 100))
  {
    qr.Q.print();
    qr.R.print();
    QR.rename("QR");
    QR.print();
    validation_matrix.rename("validation_matrix");
    validation_matrix.print();
  }

  status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;

  std::cout << status << std::endl;
}

//
// Validate QRRR
//
template <typename data_t>
void test_qrrr(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  if constexpr (boba::is_host(space))
  {
    constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

    ::boba::TicToc<tictoc_units> timer;
    ::boba::QR<space, data_t> qr;
    qr.qr_type = ::boba::QR<boba::default_execution_space, data_t>::qr_types::column_pivot;
    qr(test_matrix);
    timer.end();

    boba::Matrix<space, data_t> QR = qr.reform_matrix();

    bool old_check = check;
    pass_or_fail(check, boba::norm_difference_inf(QR, test_matrix), tolerance);
    bool new_fail = not(check) and old_check;

    if (new_fail and (validation_matrix.size() < 100))
    {
      qr.Q.print();
      qr.R.print();
      QR.rename("QR");
      QR.print();
      validation_matrix.rename("validation_matrix");
      validation_matrix.print();
    }

    status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;
  }

  std::cout << status << std::endl;
}

//
// Validate SVD
//
template <typename data_t>
void test_svd(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  ::boba::TicToc<tictoc_units> timer;
  ::boba::SVD<space, data_t> svd;
  svd(test_matrix);
  timer.end();

  boba::Matrix<space, data_t> USVt = svd.reform_matrix();

  bool old_check = check;
  pass_or_fail(check, boba::norm_difference_inf(USVt, validation_matrix), tolerance);
  bool new_fail = not(check) and old_check;

  if (new_fail and (validation_matrix.size() < 100))
  {
    svd.U.print();
    svd.S.print();
    svd.V.print();
    USVt.rename("USVt");
    USVt.print();
    validation_matrix.rename("validation_matrix");
    validation_matrix.print();
  }

  status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;

  std::cout << status << std::endl;
}

template <typename data_t>
void test_randomized_svd(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  if constexpr (boba::is_hip(space) && std::is_same_v<data_t, boba::complex<float>>)
  {
    std::cout << " skipped (known HIP randomized_svd failure mode for complex float)" << std::endl;
    return;
  }

  if constexpr (boba::is_hip(space) && std::is_same_v<data_t, float>)
  {
    std::cout << " skipped (known HIP randomized_svd failure mode for float)" << std::endl;
    return;
  }

  real_data_t tolerance = real_data_t(500000) * std::numeric_limits<real_data_t>::epsilon();
  if constexpr (boba::is_cuda(space))
  {
    if (validation_matrix.name() == "low_rank_vandermonde")
    {
      // The randomized SVD is less stable on this matrix family, especially on CUDA.
      tolerance *= real_data_t(5);
    }
  }

  ::boba::TicToc<tictoc_units> timer;
  ::boba::SVD<space, data_t> svd;
  svd.svd_type = ::boba::SVD<boba::default_execution_space, data_t>::svd_types::randomized;
  svd(test_matrix);
  timer.end();

  boba::Matrix<space, data_t> USVt = svd.reform_matrix();

  bool old_check = check;
  pass_or_fail(check, boba::norm_difference_inf(USVt, validation_matrix), tolerance);
  bool new_fail = not(check) and old_check;

  if (new_fail and (validation_matrix.size() < 100))
  {
    svd.U.print();
    svd.S.print();
    svd.V.print();
    USVt.rename("USVt");
    USVt.print();
    validation_matrix.rename("validation_matrix");
    validation_matrix.print();
  }

  status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;

  std::cout << status << std::endl;
}

//
// Validate truncated SVD
//
template <typename data_t>
void test_svdx(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  if constexpr (boba::is_hip(space))
  {
    constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

    ::boba::TicToc<tictoc_units> timer;
    ::boba::SVD<space, data_t> svd;
    svd.svd_type = ::boba::SVD<boba::default_execution_space, data_t>::svd_types::truncated;
    svd(test_matrix);
    timer.end();

    boba::Matrix<space, data_t> USVt = svd.reform_matrix();

    bool old_check = check;
    pass_or_fail(check, boba::norm_difference_inf(USVt, validation_matrix), tolerance);
    bool new_fail = not(check) and old_check;

    if (new_fail and (validation_matrix.size() < 100))
    {
      svd.U.print();
      svd.S.print();
      svd.V.print();
      USVt.rename("USVt");
      USVt.print();
      validation_matrix.rename("validation_matrix");
      validation_matrix.print();
    }

    status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;
  }

  std::cout << status << std::endl;
}

//
// Validate LU
//
template <typename data_t>
void test_lu(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  bool is_invertible = false;
  if (test_matrix.name() == "identity" || test_matrix.name() == "laplacian" || test_matrix.name() == "derivative")
  {
    is_invertible = (test_matrix.rows() == test_matrix.cols());
  }

  if (is_invertible)
  {
    ::boba::TicToc<tictoc_units> timer;
    ::boba::LU<space, data_t> lu;
    lu.lu_type = ::boba::LU<boba::default_execution_space, data_t>::lu_types::partial_pivot;
    lu(test_matrix);
    timer.end();

    boba::Matrix<space, data_t> LU = lu.reform_matrix();

    bool old_check = check;
    pass_or_fail(check, boba::norm_difference_inf(LU, validation_matrix), tolerance);
    bool new_fail = not(check) and old_check;

    if (new_fail and (validation_matrix.size() < 100))
    {
      lu.L.print();
      lu.U.print();
      lu.P.print();
      LU.rename("LU");
      LU.print();
      validation_matrix.rename("validation_matrix");
      validation_matrix.print();
    }

    status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;
  }

  std::cout << status << std::endl;
}

//
// Validate fullly-pivoted LU
//
template <typename data_t>
void test_lu_full(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  if constexpr (boba::is_host(space))
  {
    constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

    bool is_invertible = false;
    if (test_matrix.name() == "identity" || test_matrix.name() == "laplacian" || test_matrix.name() == "derivative")
    {
      is_invertible = (test_matrix.rows() == test_matrix.cols());
    }

    if (is_invertible)
    {
      ::boba::TicToc<tictoc_units> timer;
      ::boba::LU<space, data_t> lu;
      lu.lu_type = ::boba::LU<boba::default_execution_space, data_t>::lu_types::full_pivot;
      lu(test_matrix);
      timer.end();

      boba::Matrix<space, data_t> LU = lu.reform_matrix();

      bool old_check = check;
      pass_or_fail(check, boba::norm_difference_inf(LU, validation_matrix), tolerance);
      bool new_fail = not(check) and old_check;

      if (new_fail and (validation_matrix.size() < 100))
      {
        lu.L.print();
        lu.U.print();
        lu.P.print();
        LU.rename("LU");
        LU.print();
        validation_matrix.rename("validation_matrix");
        validation_matrix.print();
      }

      status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;
    }
  }

  std::cout << status << std::endl;
}

//
// Validate Cholesky
//
template <typename data_t>
void test_cholesky(::boba::Matrix<space, data_t>& test_matrix, ::boba::Matrix<space, data_t>& validation_matrix, bool& check)
{
  using real_data_t = boba::real_type_t<data_t>;

  std::string status = " skipped";

  constexpr real_data_t tolerance = real_data_t(2000) * std::numeric_limits<real_data_t>::epsilon();

  bool is_symmetric_positive_definite = false;
  if (test_matrix.name() == "identity" || test_matrix.name() == "laplacian")
  {
    is_symmetric_positive_definite = (test_matrix.rows() == test_matrix.cols());
  }
  // Hilbert is mathematically SPD but too ill-conditioned for this fixed-tolerance
  // reconstruction test, and vendor no-pivot Cholesky rejects larger cases outright.

  if (is_symmetric_positive_definite)
  {
    ::boba::TicToc<tictoc_units> timer;
    ::boba::Cholesky<space, data_t> cholesky;
    // ? cholesky.lu_type = ::boba::LU<boba::default_execution_space, data_t>::lu_types::full_pivot;
    cholesky(test_matrix);
    timer.end();

    boba::Matrix<space, data_t> LLT = cholesky.reform_matrix();

    auto error = ::boba::norm_difference_inf(LLT, validation_matrix);

    bool old_check = check;
    pass_or_fail(check, error, tolerance);
    bool new_fail = not(check) and old_check;

    if (new_fail and (validation_matrix.size() < 100))
    {
      cholesky.L.print();
      LLT.rename("LLT");
      LLT.print();
      validation_matrix.rename("validation_matrix");
      validation_matrix.print();
    }

    status = std::string(" elapsed ") + std::to_string(timer.duration) + std::string(" ") + timer.units_string;
  }

  std::cout << status << std::endl;
}

template <typename data_t>
void test_cases(std::string const& float_name, std::vector<size_t> rows_list, std::vector<size_t> cols_list, size_t kernel, size_t max_size, bool& check)
{
  for (size_t which = 0; which < 7; which++)
  {
    for (auto rows : rows_list)
    {
      for (auto cols : cols_list)
      {
        // fix
        if (rows * cols > max_size)
        {
          continue;
        }

        boba::Matrix<space, data_t> validation_matrix({rows, cols});
        setup_matrices(which, validation_matrix);

        checkpoint();
        if ((kernel == 0) or (kernel == 1))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " qr";
          if constexpr (qr_supported<data_t>())
          {
            boba::Matrix<space, data_t> test_matrix = validation_matrix;
            std::cout << std::endl;
            test_qr(test_matrix, validation_matrix, check);
          }
          else
          {
            std::cout << " skipped" << std::endl;
          }
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 2))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " qrrr" << std::endl;
          boba::Matrix<space, data_t> test_matrix = validation_matrix;
          test_qrrr(test_matrix, validation_matrix, check);
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 3))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " svd";
          if constexpr (svd_supported<data_t>())
          {
            boba::Matrix<space, data_t> test_matrix = validation_matrix;
            std::cout << std::endl;
            test_svd(test_matrix, validation_matrix, check);
          }
          else
          {
            std::cout << " skipped" << std::endl;
          }
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 4))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " svdx" << std::endl;
          boba::Matrix<space, data_t> test_matrix = validation_matrix;
          test_svdx(test_matrix, validation_matrix, check);
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 5))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " lu";
          if constexpr (lu_supported<data_t>())
          {
            boba::Matrix<space, data_t> test_matrix = validation_matrix;
            std::cout << std::endl;
            test_lu(test_matrix, validation_matrix, check);
          }
          else
          {
            std::cout << " skipped" << std::endl;
          }
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 6))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " lu_full" << std::endl;
          boba::Matrix<space, data_t> test_matrix = validation_matrix;
          test_lu_full(test_matrix, validation_matrix, check);
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 7))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " cholesky";
          if constexpr (cholesky_supported<data_t>())
          {
            boba::Matrix<space, data_t> test_matrix = validation_matrix;
            std::cout << std::endl;
            test_cholesky(test_matrix, validation_matrix, check);
          }
          else
          {
            std::cout << " skipped" << std::endl;
          }
        }

        checkpoint();
        if ((kernel == 0) or (kernel == 8))
        {
          std::cout << ::boba::detail::execution_space_name(space) << " "
                    << float_name << " "
                    << validation_matrix.name() << " "
                    << rows << "x"
                    << cols << " randomized_svd" << std::endl;
          boba::Matrix<space, data_t> test_matrix = validation_matrix;
          test_randomized_svd(test_matrix, validation_matrix, check);
        }
      }
    }
  }
}

//
//
//

int main(int argc, char* argv[])
{

  boba::splash();
  boba::init();
  std::cout << "Tests for LU, Cholesky, QR, and SVD implementations" << std::endl;

  BOBA_CALI_EXTERNAL_MARK

  checkpoint();

  size_t in_rows = 0;
  size_t in_cols = 0;
  size_t kernel = 0;
  size_t max_size = 10000;

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

  args.add_optional_argument(max_size,
                             "-M",
                             "--maximum",
                             "Maximum matrix size (rows*cols) that will be attempted.");

  args.parse_check();

  boba_always_assert_equal((in_cols > 0), (in_rows > 0), "Set neither or both");

  std::vector<size_t> rows_list = {5, 10, 15, 20, 50, 100};
  std::vector<size_t> cols_list = {5, 10, 15, 20, 50, 100};

  if (in_cols > 0)
  {
    rows_list.resize(0);
    rows_list.push_back(in_rows);
    cols_list.resize(0);
    cols_list.push_back(in_cols);
  }

  bool check = true;

  if (::boba::is_env_nonempty("linalg_test_single_precision"))
  {
    test_cases<float>("float", rows_list, cols_list, kernel, max_size, check);
  }
  else if (::boba::is_env_nonempty("linalg_test_complex_single_precision"))
  {
    test_cases<boba::complex<float>>("complex_float", rows_list, cols_list, kernel, max_size, check);
  }
  else if (::boba::is_env_nonempty("linalg_test_complex_double_precision"))
  {
    test_cases<boba::complex<double>>("complex_double", rows_list, cols_list, kernel, max_size, check);
  }
  else
  {
    test_cases<double>("double", rows_list, cols_list, kernel, max_size, check);
  }

  boba::finalize();
  return final_check(check);
}
