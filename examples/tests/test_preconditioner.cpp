// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "BOBA/boba.hpp"
#include "common.hpp"

/*
  Test BoBa's Krylov preconditioning
*/

constexpr ::boba::execution_space space = ::boba::default_execution_space;

// not used; keeping around for reference
std::tuple<boba::Matrix<space, double>,
           boba::Matrix<space, double>,
           boba::Vector<space, double>,
           boba::Vector<space, double>>
create_test_system_full(size_t n, double epsilon, double lambda_min, double lambda_max, double rho)
{
  //
  // Test setup from Carson et al (2024), "Towards understanding CG and GMRES through examples"
  // (https://doi.org/10.1016/j.laa.2024.04.003)
  //

  //
  // example parameters
  //   epsilon    = 1.0e-03
  //   lambda_min = 0.1
  //   lambda_max = 100.0
  //   rho        = 1.0
  //

  // construct orthogonal U using QR of random matrix X, where X is small random perturbation of identity
  ::boba::Matrix<space, double> X({n, n});
  X.fill_with_random(); // random matrix with entries in [0, 1]

  auto X_view = X.view();
  ::boba::loop<space, 2>(X.sizes(), [=] __boba_host_device__(::boba::Array<size_t, 2> ij)
  {
    X_view(ij) = epsilon * (2.0 * X_view(ij) - 1.0); // random matrix with entries in [-epsilon, epsilon]
    X_view(ij) += (ij[0] == ij[1]) ? 1.0 : 0.0;      // shift the diagonal entries
  });

  ::boba::QR<space, double> qr;
  qr(X);

  // construct Lambda
  ::boba::Matrix<space, double> Lambda({n, n});
  Lambda.fill_with_zeros();

  auto Lambda_view = Lambda.view();
  if (rho >= 1.0)
  {
    ::boba::loop<space, 1>(n, [=] __boba_host_device__(size_t i)
    {
      Lambda_view({i, i}) = lambda_min + (lambda_max - lambda_min) * boba::pow(i / (n - 1.0), rho);
    });
  }
  else
  {
    ::boba::loop<space, 1>(n, [=] __boba_host_device__(size_t i)
    {
      Lambda_view({i, i}) = lambda_max - (lambda_max - lambda_min) * boba::pow(1.0 - i / (n - 1.0), 1.0 / rho);
    });
  }

  // A = U * Lambda * U'
  ::boba::Matrix<space, double> A = qr.Q * (Lambda * qr.Q.transpose());

  // construct diagonal preconditioner
  ::boba::Matrix<space, double> P({n, n});
  P.fill_with_zeros();

  auto A_view = A.const_view();
  auto P_view = P.view();
  ::boba::loop<space, 1>(n, [=] __boba_host_device__(size_t i)
  {
    P_view({i, i}) = 1.0 / A_view({i, i});
  });

  // create LHS
  ::boba::Vector<space, double> x_exact({n});
  x_exact.fill_with(1.0);

  // create RHS
  auto b = A * x_exact;

  return std::make_tuple(A, P, b, x_exact);
}

void symmetric_eigendecomp(const boba::Matrix<space, double>& matrix,
                           boba::Vector<space, double>& eigenvals,
                           boba::Matrix<space, double>& eigenvecs)
{
  size_t n = matrix.rows();

  boba::Matrix<boba::execution_space::CPU, double> matrix_host = matrix;

  Eigen::MatrixXd matrix_eigen(n, n);
  std::copy(matrix_host.data(), matrix_host.data() + n * n, matrix_eigen.data());

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> factorization(matrix_eigen);

  const Eigen::VectorXd eigenvals_eigen = factorization.eigenvalues();
  const Eigen::MatrixXd eigenvecs_eigen = factorization.eigenvectors();

  boba::Vector<boba::execution_space::CPU, double> eigenvals_host({n});
  std::copy(eigenvals_eigen.data(), eigenvals_eigen.data() + n, eigenvals_host.data());

  boba::Matrix<boba::execution_space::CPU, double> eigenvecs_host({n, n});
  std::copy(eigenvecs_eigen.data(), eigenvecs_eigen.data() + n * n, eigenvecs_host.data());

  eigenvals = eigenvals_host;
  eigenvecs = eigenvecs_host;
}

template <size_t dimension>
std::tuple<boba::TensorTrainMatrix<dimension, space, double>,
           boba::TensorTrainMatrix<dimension, space, double>,
           boba::TensorTrain<dimension, space, double>,
           boba::TensorTrain<dimension, space, double>>
create_test_system_tt(size_t n, size_t q)
{
  //
  // Test setup from Coulaud et al (2022), "A robust GMRES algorithm in tensor train format"
  // (arXiv:2210.14533)
  //

  //
  // higher q supposedly leads to better preconditioner
  // seems to be practically true, only up to certain level
  //

  boba::Matrix<space, double> laplacian_1d({n, n});
  laplacian_1d.fill_with_zeros();
  laplacian_1d.fill_diagonal(0, 2.0);
  laplacian_1d.fill_diagonal(1, -1.0);
  laplacian_1d.fill_diagonal(-1, -1.0);

  boba::Matrix<space, double> identity_1d({n, n});
  identity_1d.set_to_identity_matrix();

  boba::Vector<space, double> eigenvals;
  boba::Matrix<space, double> eigenvecs;
  symmetric_eigendecomp(laplacian_1d, eigenvals, eigenvecs);
  auto eigenvals_view = eigenvals.const_view();

  boba::Array<size_t, dimension> sizes = boba::filled_array<dimension>(n);

  boba::TensorTrainMatrix<dimension, space, double> laplacian(sizes, sizes);
  for (size_t k = 0; k < dimension; ++k)
  {
    boba::Array<boba::Matrix<space, double>, dimension> term;
    for (size_t j = 0; j < dimension; ++j)
    {
      term[j] = (j == k) ? laplacian_1d : identity_1d;
    }
    laplacian += boba::make_ttm_from_matrices(term);
  }
  laplacian.round();

  boba::TensorTrainMatrix<dimension, space, double> preconditioner(sizes, sizes);
  for (int k = -static_cast<int>(q); k <= static_cast<int>(q); ++k)
  {
    double xi = boba::pi / q;
    double tk = boba::exp(k * xi);
    double ck = xi * tk;

    boba::Vector<space, double> dk({n});
    auto dk_view = dk.view();
    boba::loop<space, 1>(n, [=] __boba_host_device__(size_t i)
    {
      dk_view(i) = boba::exp(-tk * eigenvals_view(i));
    });

    boba::Matrix<space, double> Mk_1d = eigenvecs;
    boba::apply_as_diagonal_right_in_place(dk, Mk_1d);
    Mk_1d = Mk_1d * eigenvecs.transpose();

    preconditioner += ck * boba::make_ttm_from_matrices(boba::filled_array<dimension>(Mk_1d));
  }
  preconditioner.svd_tolerance_relative = 1.0e-02;
  preconditioner.round();

  boba::TensorTrain<dimension, space, double> input(sizes);
  input.fill_with(1.0);

  boba::TensorTrain<dimension, space, double> rhs = laplacian * input;
  rhs.round();

  return std::make_tuple(laplacian, preconditioner, rhs, input);
}

template <typename operator_t, typename vector_t>
void test_solve(const operator_t& A, const operator_t& P, const vector_t& b, const vector_t& x_exact, size_t krylov_method, size_t outer_iters, size_t inner_iters, double tolerance, bool& check)
{
  vector_t x0(b.sizes());
  x0.fill_with(0.0);

  auto x_no_prec = x0;
  auto x_left_prec = x0;
  auto x_right_prec = x0;

  boba_print("----------------------------------------");
  boba_print("----------  no preconditioner  ---------");
  boba_print("----------------------------------------");

  // Instantiate solver with your options without preconditioner
  checkpoint();
  {
    boba::Krylov<operator_t, vector_t> solver(outer_iters, inner_iters, tolerance);
    solver.method = krylov_method;
    solver.verbose = true;
    solver.set_matrix(A);

    // Check the initial residual
    auto initial_residual_no_preconditioner = ::boba::norm_frobenius(solver.compute_residual(x0, b));
    boba_print(initial_residual_no_preconditioner);

    // Run solver
    solver.solve(b, x0, x_no_prec);

    // Check the final residual
    auto final_residual_no_preconditioner = ::boba::norm_frobenius(solver.compute_residual(x_no_prec, b));
    boba_print(final_residual_no_preconditioner);
  }

  pass_or_fail(check, ::boba::norm_difference_inf(x_no_prec.decompress(), x_exact.decompress()), 1.0e-07);

  boba_print("----------------------------------------");
  boba_print("------  with left preconditioner  ------");
  boba_print("----------------------------------------");

  // Instantiate solver with your options where we use left preconditioner
  checkpoint();
  {
    boba::Krylov<operator_t, vector_t> solver(outer_iters, inner_iters, tolerance);
    solver.method = krylov_method;
    solver.verbose = true;
    solver.set_matrix(A);
    solver.set_left_preconditioner(P);

    // Check the initial residual
    auto initial_residual = ::boba::norm_frobenius(solver.compute_unpreconditioned_residual(x0, b));
    boba_print(initial_residual);

    // Run solver
    solver.solve(b, x0, x_left_prec);

    // Check the final residual
    auto final_residual = ::boba::norm_frobenius(solver.compute_unpreconditioned_residual(x_left_prec, b));
    boba_print(final_residual);
  }

  pass_or_fail(check, ::boba::norm_difference_inf(x_left_prec.decompress(), x_exact.decompress()), 1.0e-07);

  if (krylov_method != boba::KrylovMethods::conjugate_gradient)
  {
    // left and right preconditioning for CG should be the same

    boba_print("----------------------------------------");
    boba_print("------  with right preconditioner  -----");
    boba_print("----------------------------------------");

    // Instantiate solver with your options where we use left preconditioner
    checkpoint();
    {
      boba::Krylov<operator_t, vector_t> solver(outer_iters, inner_iters, tolerance);
      solver.method = krylov_method;
      solver.verbose = true;
      solver.set_matrix(A);
      solver.set_right_preconditioner(P);

      // Check the initial residual
      auto initial_residual = ::boba::norm_frobenius(solver.compute_unpreconditioned_residual(x0, b));
      boba_print(initial_residual);

      // Run solver
      solver.solve(b, x0, x_right_prec);

      // Check the final residual
      auto final_residual = ::boba::norm_frobenius(solver.compute_unpreconditioned_residual(x_right_prec, b));
      boba_print(final_residual);
    }

    pass_or_fail(check, ::boba::norm_difference_inf(x_right_prec.decompress(), x_exact.decompress()), 1.0e-07);
  }
}

template <size_t dimension>
void test_preconditioned_solve_tt(size_t n, size_t q, size_t krylov_method, size_t outer_iters, size_t inner_iters, double tolerance, bool& check)
{
  auto [A, P, b, x_exact] = create_test_system_tt<dimension>(n, q);
  test_solve(A, P, b, x_exact, krylov_method, outer_iters, inner_iters, tolerance, check);
}

int main(int argc, char* argv[])
{

  BOBA_CALI_EXTERNAL_MARK
  boba::splash();
  boba::init();

  size_t dimension = 3;
  size_t resolution = 10;
  size_t quality = 1;
  size_t krylov_method = ::boba::KrylovMethods::conjugate_gradient;
  size_t outer_iters = 1;
  size_t inner_iters = 250;
  double tolerance = 1.0e-12;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(dimension,
                             "-d",
                             "--dimension",
                             "Dimension of TT vectors/matrices");

  args.add_optional_argument(resolution,
                             "-n",
                             "--resolution",
                             "Resolution in each dimension");

  args.add_optional_argument(quality,
                             "-q",
                             "--quality",
                             "Quality parameter of the preconditioner");

  args.add_optional_argument(krylov_method,
                             "-m",
                             "--method",
                             "Choice of krylov_method");

  args.add_optional_argument(outer_iters,
                             "-oi",
                             "--outer_iters",
                             "Number of outer iterations");

  args.add_optional_argument(inner_iters,
                             "-ii",
                             "--inner_iters",
                             "Number of inner iterations");

  args.add_optional_argument(tolerance,
                             "-t",
                             "--tolerance",
                             "Convergence tolerance");

  args.parse_check();

  boba_always_assert(quality > 0, "preconditioner quality parameter must be positive integer");

  bool check = true;

  if (dimension == 1)
  {
    test_preconditioned_solve_tt<1>(resolution, quality, krylov_method, outer_iters, inner_iters, tolerance, check);
  }
  else if (dimension == 2)
  {
    test_preconditioned_solve_tt<2>(resolution, quality, krylov_method, outer_iters, inner_iters, tolerance, check);
  }
  else if (dimension == 3)
  {
    test_preconditioned_solve_tt<3>(resolution, quality, krylov_method, outer_iters, inner_iters, tolerance, check);
  }
  else if (dimension == 4)
  {
    test_preconditioned_solve_tt<4>(resolution, quality, krylov_method, outer_iters, inner_iters, tolerance, check);
  }
  else
  {
    boba_error("dimension is not supported");
  }

  return final_check(check);
}
