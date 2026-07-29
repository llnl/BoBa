// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "../tests/common.hpp"

/*
  This tutorial shows how to perform basic matrix/vector manipulation and
  some linear algebra concepts
*/

constexpr ::boba::execution_space host_space = ::boba::execution_space::CPU;
constexpr ::boba::execution_space device_space = ::boba::default_execution_space;

int main() {

  boba::splash();
  boba::init();

  bool check = true;

  checkpoint();
  {
    boba_print("Basic manipulation");

    // Let's make a small identity matrix
    ::boba::Matrix<device_space, double> example_matrix({3, 3});
    example_matrix.set_to_identity_matrix();

    // Different ways to print it:
    example_matrix.print();

    boba_print(example_matrix);

    auto example_matrix_view = example_matrix.const_view();

    ::boba::loop<device_space, 2>(example_matrix_view.sizes(),
        [=]__boba_host_device__(::boba::Array<size_t, 2> mid)
      {
        auto [i, j] = mid;
        auto matrix_entry = example_matrix_view(mid);
        printf("Matrix index and value {%lu, %lu, %lf} \n", i, j, matrix_entry);
      });

    // We can check various norms
    boba_print(::boba::norm_frobenius(example_matrix));
    boba_print(example_matrix.matrix_one_norm());
    boba_print(example_matrix.matrix_inf_norm());

    // Let's make a small vector
    ::boba::Vector<device_space, double> example_vector({3});

    example_vector.print();

    // Let's do a matvec
    const auto vector_2 = example_matrix * example_vector;

    // We can check various norms
    boba_print(::boba::norm_frobenius(vector_2));
    boba_print(vector_2.vector_one_norm());
    boba_print(::boba::norm_inf(vector_2));

    // How do we check that  vector_2 = example_vector ?

    auto difference = vector_2 - example_vector;

    boba_print(difference);

    boba_print(::boba::norm_inf(difference));

    boba_print(norm_difference_inf(vector_2, example_vector));

    auto error = ::boba::norm_inf(difference);

    pass_or_fail(check, error, 1e-09);

    // Let's scale the matrix and the input vector and check equality in a new way
    auto scaled_input = example_vector/2.0;
    auto scaled_matrix = example_matrix*2.0;

    auto vector_3 = scaled_matrix*scaled_input;

    boba_print(::boba::norm_inf(vector_2 - vector_3));

    pass_or_fail(check, ::boba::norm_inf(vector_2 - vector_3), 1e-09);
  }

  checkpoint();
  {
    boba_print("Solving a small system");

    // Say I want to create this system and try to solve it with different Krylov methods
    // | 4       | |x1|   |1|
    // |    9    | |x2| = |2|
    // |    1  11| |x3|   |3|
    ::boba::Matrix<host_space, double> A({3, 3});
    A({0, 0}) = 4;
    A({1, 1}) = 9;
    A({2, 2}) = 11;
    A({2, 1}) = 1;

    // Created a vector for rhs
    ::boba::Vector<host_space, double> b({3});
    b({0}) = 1;
    b({1}) = 2;
    b({2}) = 3;

    // We can't do backslashes in C++, but we have the backsolve function.
    auto c = boba::backsolve(A, b);

    boba_print(c);

    // Validate solution

    auto residual = A*c - b;

    boba_print(residual);

    pass_or_fail(check, ::boba::norm_inf(residual), 1e-09);

    boba_print("Checking consistency of right_backsolve");
    // We also have the right-backsolve function
    {
      ::boba::Matrix<host_space, double> bT({1, b.size()});
      bT.reshape(b);
      auto AT = A.transpose();
      auto cT = boba::right_backsolve(AT, bT);
      pass_or_fail(check, boba::norm_difference_inf(c, ::boba::flatten(cT)), 1.0e-09);
    }

    boba_print("QR solver");

    // Let's solve this another way - using QR
    // Recall that if A = QR
    // then  Ax = QRx = b
    // so Rx = Q^T * b
    // then x = R \ (Q^T * b)

    // Instantiate the QR class
    // Q and R are member variables

    ::boba::QR<host_space, double> qr;
    qr(A);

    boba_print(qr.Q);
    boba_print(qr.R);

    auto Rc2 = qr.Q.transpose() * b;
    auto c2 = boba::backsolve(qr.R, Rc2);

    auto residual2 = A*c2 - b;

    boba_print(residual2);

    pass_or_fail(check, ::boba::norm_inf(residual2), 1e-09);

    // Let's solve this another way - using SVD
    // Rrecall that if A = USV^T
    // Then Ax = USV^Tx = b
    //  SV^Tx = U^T * b
    //  V^Tx = S \ ( U^T * b )
    //  x = V * ( S \ ( U^T * b ))

    // Instantiate the SVD class
    // U, S, V are member variables

    ::boba::SVD<host_space, double> svd;
    svd(A);

    boba_print(svd.U);
    boba_print(svd.S); // note that S is stored as a vector
    boba_print(svd.V);

    auto SVc3 = svd.U.transpose() * b;
    // S needs to be converted into a matrix
    auto S_matrix = boba::diagonalize(svd.S);
    auto Vc3 = boba::backsolve(S_matrix, SVc3);
    auto c3 = svd.V * Vc3;

    auto residual3 = A*c3 - b;

    boba_print(residual3);

    pass_or_fail(check, ::boba::norm_inf(residual3), 1e-09);
  }

  checkpoint();
  {
    boba_print("Make a laplacian");

    // Let's generate this matrix on the host
    ::boba::Matrix<host_space, double> laplacian_host({10, 10});

    // Put ones on the diagonal
    laplacian_host.set_to_identity_matrix();

    // Now they will be twos
    laplacian_host *= 2.0;

    // fill the 1st superdiagonal and subdiagonal with -1.0
    laplacian_host.fill_diagonal( 1, -1.0);
    laplacian_host.fill_diagonal(-1, -1.0);

    // Erase the first and last row
    laplacian_host.fill_row(0, 0.0);
    laplacian_host.fill_row(laplacian_host.rows()-1, 0.0);

    // Set Dirichlet boundary conditions
    auto laplacian_view = laplacian_host.view();

    // This must be done on the host
    // You can't access single points of memory on the device without going into a loop.
    laplacian_view({0, 0}) = 1.0;
    laplacian_view({laplacian_host.rows()-1, laplacian_host.cols()-1}) = 1.0;

    laplacian_host.print();

    // Now copy it to the device
    ::boba::Matrix<device_space, double> laplacian = laplacian_host;

    // Now let's use a krylov solver
    boba::Vector<device_space, double> b({laplacian.cols()});
    b.fill_with(1.0/double(laplacian.cols()));

    // set up guess x = 0 and solution object
    auto initial_guess = b * 0.0;
    auto solution = initial_guess * 0.0;

    size_t outer_iterations = 2;
    size_t inner_iterations = laplacian.rows();
    double tolerance = 1.0e-10;

    // Instantiate solver with your options
    checkpoint();
    using operator_t = boba::Matrix<device_space, double>;
    using vector_t = boba::Vector<device_space, double>;
    boba::Krylov<operator_t, vector_t> solver(
      outer_iterations,
      inner_iterations,
      tolerance);

    solver.method = boba::KrylovMethods::gmres;
    solver.set_matrix(laplacian);

    // Check the initial residual
    boba_print(::boba::norm_frobenius(solver.compute_residual(initial_guess, b)));

    solver.solve(b, initial_guess, solution);

    // Check the final residual
    auto final_residual = ::boba::norm_frobenius(solver.compute_residual(solution, b));
    boba_print(final_residual);

    pass_or_fail(check, final_residual, 1e-09);
  }

  return final_check(check);
}
// clang-format on
