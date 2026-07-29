// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// clang-format off


#include "BOBA/boba.hpp"

//
// Read data generated from MATLAB, operate on it, and save the results to MAT
// files for verification in the next step.
//

int main() {

  boba::init();
  //
  // We chose dimension d = 5 in MATLAB when we generated the MAT files. Now, we
  // have to respect that by specifying the same value.
  //
  constexpr size_t dimension = 5;

  //
  // the TT object is saved as a cell array named x_tt_cores inside the
  // x_tt_cores.mat file. The restriction on the variable name and the file name
  // having to be the same originates from the interface below.
  //
  // TODO<feature>: Allow reading arbitrarily named cell array from a MAT file
  //
  boba::TensorTrain<dimension, boba::host_space, double> x;
  boba::read_from_mat_file(x, "x_tt_cores");

  //
  // Same for the TTM object:
  //
  boba::TensorTrainMatrix<dimension, boba::host_space, double> A;
  boba::read_from_mat_file(A, "A_tt_cores");

  //
  // Let's compute the matrix-vector product and save it to a MAT file. This
  // create a cell array named y_tt_cores inside the y_tt_cores.mat file.
  //
  // TODO<feature>: Writing arbitrarily named cell array to a MAT file.
  //                To allow writing multiple variables to the same file, a
  //                MatFile.save(tensor/tt/ttm) interface might be better.
  //                Same for read.
  //
  auto y = A * x;
  boba::write_to_mat_file(y, "y_tt_cores");

  //
  // Let's try creating the negative Laplacian in TT format and save to a MAT file.
  //
  const boba::Array<size_t, 4> sizes{4, 5, 8, 4};
  const size_t numel = boba::product(sizes);
  const double delta = 1.0 / static_cast<double>(numel + 1);

  boba::Matrix<boba::host_space, double> B({numel, numel});
  B.fill_with_zeros();
  B.fill_diagonal(-1, -1.0);
  B.fill_diagonal( 0,  2.0);
  B.fill_diagonal( 1, -1.0);
  B /= delta * delta;

  boba::TensorTrainMatrix<4, boba::host_space, double> B_tt(sizes, sizes);
  B_tt.compress(B);

  boba::write_to_mat_file(B_tt, "B_tt_cores");

  //
  // Finally, create a full vector by sampling sin(2 * pi * x) on [0, 1].
  //
  boba::Vector<boba::host_space, double> t({numel});
  auto t_view = t.view();
  boba::loop<boba::host_space, 1>(numel, [=]__boba_host_device__(size_t i) {
    t_view(i) = boba::sin(2.0 * boba::pi * (static_cast<double>(i) + 1.0) * delta);
  });

  //
  // Applying the negative Laplacian TT operator to a normal vector will create
  // a new normal vector. We are writing this result "tensor" to a MAT file.
  //
  auto z = B_tt * t;
  boba::write_to_mat_file(z, "z");

  boba::finalize();
}
// clang-format on
