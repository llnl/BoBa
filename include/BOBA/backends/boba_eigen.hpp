// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_ENABLE_EIGEN
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

namespace boba
{

#ifdef BOBA_ENABLE_EIGEN

/**
 * \brief Returns an Eigen-compatible Matrix view.
 * \return Mutable Eigen map over this Matrix's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> get_eigen_map(Matrix<space, data_t>& matrix)
{
  return {matrix.data(), static_cast<Eigen::Index>(matrix.rows()), static_cast<Eigen::Index>(matrix.cols())};
}

/**
 * \brief Returns a const Eigen-compatible Matrix view.
 * \return Const Eigen map over this Matrix's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> get_const_eigen_map(Matrix<space, data_t> const& matrix)
{
  return {matrix.const_data(), static_cast<Eigen::Index>(matrix.rows()), static_cast<Eigen::Index>(matrix.cols())};
}

/**
 * \brief Returns a const Eigen-compatible Matrix view.
 * \return Const Eigen map over this Matrix's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<const Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>> get_eigen_map(Matrix<space, data_t> const& matrix)
{
  return get_const_eigen_map(matrix);
}

/**
 * \brief Compatibility spelling for const Eigen Matrix maps.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
auto get_eigen_const_map(Matrix<space, data_t> const& matrix)
{
  return get_const_eigen_map(matrix);
}

/**
 * \brief Returns an Eigen-compatible Vector view.
 * \return Mutable Eigen map over this Vector's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<Eigen::Matrix<data_t, Eigen::Dynamic, 1>> get_eigen_map(Vector<space, data_t>& vector)
{
  return {vector.data(), static_cast<Eigen::Index>(vector.size())};
}

/**
 * \brief Returns a const Eigen-compatible Vector view.
 * \return Const Eigen map over this Vector's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<const Eigen::Matrix<data_t, Eigen::Dynamic, 1>> get_const_eigen_map(Vector<space, data_t> const& vector)
{
  return {vector.const_data(), static_cast<Eigen::Index>(vector.size())};
}

/**
 * \brief Returns a const Eigen-compatible Vector view.
 * \return Const Eigen map over this Vector's storage.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
Eigen::Map<const Eigen::Matrix<data_t, Eigen::Dynamic, 1>> get_eigen_map(Vector<space, data_t> const& vector)
{
  return get_const_eigen_map(vector);
}

/**
 * \brief Compatibility spelling for const Eigen Vector maps.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
auto get_eigen_const_map(Vector<space, data_t> const& vector)
{
  return get_const_eigen_map(vector);
}

#else

struct UnavailableEigenMap
{
  UnavailableEigenMap()
  {
    boba_error("Eigen map support requires BOBA_ENABLE_EIGEN.");
  }
};

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_map(Matrix<space, data_t>& matrix)
{
  ::boba::detail::ignore(matrix);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_map(Matrix<space, data_t> const& matrix)
{
  ::boba::detail::ignore(matrix);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_const_eigen_map(Matrix<space, data_t> const& matrix)
{
  ::boba::detail::ignore(matrix);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_const_map(Matrix<space, data_t> const& matrix)
{
  return get_const_eigen_map(matrix);
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_map(Vector<space, data_t>& vector)
{
  ::boba::detail::ignore(vector);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_map(Vector<space, data_t> const& vector)
{
  ::boba::detail::ignore(vector);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_const_eigen_map(Vector<space, data_t> const& vector)
{
  ::boba::detail::ignore(vector);
  return {};
}

template <execution_space space, typename data_t>
[[nodiscard]]
UnavailableEigenMap get_eigen_const_map(Vector<space, data_t> const& vector)
{
  return get_const_eigen_map(vector);
}

#endif

namespace detail
{

#ifdef BOBA_ENABLE_EIGEN

/**
 * @brief Solves a linear system with Eigen's restarted GMRES implementation.
 * @tparam space Execution space of the input matrix and vector.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param matrix System matrix.
 * @param input Right-hand side vector.
 * @param tolerance_relative Relative convergence tolerance.
 * @param maximum_iterations Maximum GMRES iteration count.
 * @param sparsity Sparsification threshold passed to Eigen.
 * @return Host-space solution vector.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
boba::Vector<host_space, data_t> eigen_gmres(
  const boba::Matrix<space, data_t>& matrix,
  const boba::Vector<space, data_t>& input,
  data_t tolerance_relative = 1.0e-7,
  size_t maximum_iterations = 30,
  data_t sparsity = 1.0e-3)
{
  BOBA_CALI_MARK
  boba_always_assert_equal(space, host_space, "eigen_gmres only useable for host data.");
  boba::Vector<host_space, data_t> output(input.sizes());
  output.fill_with_zeros();
  auto input_eigen = ::boba::get_eigen_map(input);
  auto output_eigen = ::boba::get_eigen_map(output);

  ::boba::detail::ignore(tolerance_relative);
  ::boba::detail::ignore(maximum_iterations);

  auto matrix_eigen = ::boba::get_eigen_map(matrix).sparseView(sparsity);

  ::Eigen::SparseLU<Eigen::SparseMatrix<data_t>> esolver;
  esolver.analyzePattern(matrix_eigen);
  esolver.factorize(matrix_eigen);
  output_eigen = esolver.solve(input_eigen);
  return output;
}

/**
 * @brief Multiplies two matrices with Eigen and writes the result to `matrix_C`.
 * @tparam A_matrix_t Left operand matrix type.
 * @tparam B_matrix_t Right operand matrix type.
 * @tparam C_matrix_t Output matrix type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_matrix_t, typename B_matrix_t, typename C_matrix_t>
void eigen_gemm(
  A_matrix_t const& matrix_A,
  B_matrix_t const& matrix_B,
  C_matrix_t& matrix_C)
{
  auto A_map = ::boba::get_const_eigen_map(matrix_A);
  auto B_map = ::boba::get_const_eigen_map(matrix_B);
  auto C_map = ::boba::get_eigen_map(matrix_C);
  C_map = A_map * B_map;
}

/**
 * @brief Solves a least-squares system with Eigen's column-pivoted QR.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param matrix_A System matrix.
 * @param matrix_B Right-hand side matrix.
 * @param matrix_C Output solution matrix.
 */
template <typename data_t>
void ls_solve_qr_eigen(Matrix<execution_space::CPU, data_t> const& matrix_A,
                       Matrix<execution_space::CPU, data_t> const& matrix_B,
                       Matrix<execution_space::CPU, data_t>& matrix_C)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  auto A_map = ::boba::get_const_eigen_map(matrix_A);
  auto rhs_map = ::boba::get_const_eigen_map(matrix_B);
  auto output_map = ::boba::get_eigen_map(matrix_C);
  Eigen::ColPivHouseholderQR<EigenMatrix> solver(A_map);
  output_map = solver.solve(rhs_map);
}

/**
 * @brief Solves a minimum-norm rectangular system with Eigen's complete orthogonal decomposition.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param matrix_A System matrix.
 * @param matrix_B Right-hand side matrix.
 * @param matrix_C Output solution matrix.
 */
template <typename data_t>
void ls_solve_cod_eigen(Matrix<execution_space::CPU, data_t> const& matrix_A,
                        Matrix<execution_space::CPU, data_t> const& matrix_B,
                        Matrix<execution_space::CPU, data_t>& matrix_C)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  auto A_map = ::boba::get_const_eigen_map(matrix_A);
  auto rhs_map = ::boba::get_const_eigen_map(matrix_B);
  auto output_map = ::boba::get_eigen_map(matrix_C);
  Eigen::CompleteOrthogonalDecomposition<EigenMatrix> solver(A_map);
  output_map = solver.solve(rhs_map);
}

/**
 * \brief Forms the lower-triangular factor from Eigen's LLT decomposition.
 *
 * The output preserves BoBa's existing storage convention for this code path.
 *
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1LLT.html
 */
template <typename data_t>
void factor_cholesky_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  BOBA_CALI_BEGIN("initialize_cholesky");

  boba_always_assert_equal(input.rows(), input.cols(), "Cholesky requires SPD matrix");

  Eigen::LLT<EigenMatrix> eigen_cholesky;
  BOBA_CALI_SWITCH("initialize_cholesky", "compute_cholesky");
  Eigen::Map<const EigenMatrix> eigen_input = ::boba::get_const_eigen_map(input);
  eigen_cholesky.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_cholesky", "form_L");
  size_t rows = static_cast<size_t>(eigen_input.rows());
  size_t cols = static_cast<size_t>(eigen_input.cols());

  matrix_L.resize({rows, cols});
  matrix_L.fill_with_zeros();

  Eigen::Map<EigenMatrix> Lmap = ::boba::get_eigen_map(matrix_L);
  Lmap.template triangularView<Eigen::Lower>() = eigen_cholesky.matrixL();
  BOBA_CALI_END("form_L");
}

/**
 * \brief Forms a reduced Householder QR factorization with Eigen.
 *
 * On return, \p matrix_Q stores the reduced orthonormal factor, \p matrix_R
 * stores the reduced upper-triangular factor, and \p ranks is set to
 * `min(rows, cols)`.
 *
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1HouseholderQR.html
 */
template <typename data_t>
void factor_qr_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_Q,
  Matrix<execution_space::CPU, data_t>& matrix_R,
  size_t& ranks)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  BOBA_CALI_BEGIN("initialize_qr");

  Eigen::HouseholderQR<EigenMatrix> eigen_qr(
    static_cast<int>(input.rows()),
    static_cast<int>(input.cols()));

  BOBA_CALI_SWITCH("initialize_qr", "compute_QR");
  Eigen::Map<const EigenMatrix> eigen_input = ::boba::get_const_eigen_map(input);
  eigen_qr.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_QR", "form_Q");
  size_t qrows = static_cast<size_t>(eigen_input.rows());
  size_t qcols = static_cast<size_t>(eigen_input.cols());
  ranks = boba::min(qrows, qcols);

  matrix_Q.resize({qrows, ranks});
  Eigen::Map<EigenMatrix> Qmap = ::boba::get_eigen_map(matrix_Q);
  Qmap = EigenMatrix::Identity(
    static_cast<Eigen::Index>(matrix_Q.rows()),
    static_cast<Eigen::Index>(matrix_Q.cols()));
  Qmap = eigen_qr.householderQ() * Qmap;

  EigenMatrix QR = eigen_qr.matrixQR();

  BOBA_CALI_SWITCH("form_Q", "set_R");
  matrix_R.resize({ranks, qcols});
  Eigen::Map<EigenMatrix> Rmap = ::boba::get_eigen_map(matrix_R);
  Rmap = QR.topRows(static_cast<Eigen::Index>(ranks)).rightCols(static_cast<Eigen::Index>(qcols));
  matrix_R.erase_lower_triangular();
  BOBA_CALI_END("set_R");
}

/**
 * \brief Forms a reduced column-pivoted QR factorization with Eigen.
 *
 * Eigen computes `A P = Q R`; this helper applies the column permutation to
 * return `R P^T` so the caller can reconstruct `A = Q (R P^T)` without
 * tracking the permutation separately.
 *
 * \param qr_rank_tolerance Threshold forwarded to Eigen's rank-revealing QR.
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1ColPivHouseholderQR.html
 */
template <typename data_t>
void factor_qrrr_eigen(
  Matrix<execution_space::CPU, data_t>& input,
  Matrix<execution_space::CPU, data_t>& matrix_Q,
  Matrix<execution_space::CPU, data_t>& matrix_R,
  size_t& ranks,
  real_type_t<data_t> qr_rank_tolerance)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  BOBA_CALI_BEGIN("initialize_qr");

  Eigen::ColPivHouseholderQR<EigenMatrix> eigen_qr(
    static_cast<int>(input.rows()),
    static_cast<int>(input.cols()));
  eigen_qr.setThreshold(qr_rank_tolerance);

  BOBA_CALI_SWITCH("initialize_qr", "compute_QR");
  Eigen::Map<EigenMatrix> eigen_input = ::boba::get_eigen_map(input);
  eigen_qr.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_QR", "form_Q");
  size_t qrows = static_cast<size_t>(eigen_input.rows());
  size_t qcols = static_cast<size_t>(eigen_input.cols());
  ranks = static_cast<size_t>(eigen_qr.rank());

  if (ranks == 0)
  {
    matrix_Q.resize({qrows, 1});
    matrix_Q.fill_with_zeros();
    matrix_R.resize({1, qcols});
    matrix_R.fill_with_zeros();
    BOBA_CALI_SWITCH("form_Q", "set_R");
    BOBA_CALI_END("set_R");
    return;
  }

  matrix_Q.resize({qrows, ranks});
  Eigen::Map<EigenMatrix> Qmap = ::boba::get_eigen_map(matrix_Q);
  Qmap = EigenMatrix::Identity(
    static_cast<Eigen::Index>(qrows),
    static_cast<Eigen::Index>(ranks));
  Qmap = eigen_qr.householderQ().setLength(static_cast<int>(ranks)) * Qmap;

  matrix_R.resize({ranks, qcols});
  Eigen::Map<EigenMatrix> Rmap = ::boba::get_eigen_map(matrix_R);
  Rmap = eigen_qr.matrixR().topRows(static_cast<Eigen::Index>(ranks));
  matrix_R.erase_lower_triangular();

  BOBA_CALI_SWITCH("form_Q", "set_R");
  auto perms = eigen_qr.colsPermutation().indices();
  size_t perms_rows = static_cast<size_t>(perms.rows());

  boba::PermutationMatrix<execution_space::CPU, index_t> P({perms_rows});
  auto P_view = P.view();
  for (auto i = 0_z; i < perms_rows; i++)
  {
    P_view(i) = static_cast<size_t>(perms[static_cast<Eigen::Index>(i)]);
  }

  boba_always_assert_equal(perms_rows, matrix_R.cols(), "inconsistent sizes");

  auto Pt = P.transpose();
  boba::Matrix<execution_space::CPU, data_t> Rtemp(matrix_R);
  Rtemp.rename("Rtemp");
  matrix_R = Rtemp * Pt;
  BOBA_CALI_END("set_R");
}

template <typename data_t>
void ls_solve_lu_eigen(Matrix<execution_space::CPU, data_t> const& matrix_A,
                       Matrix<execution_space::CPU, data_t> const& matrix_B,
                       Matrix<execution_space::CPU, data_t>& matrix_C)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  auto A_map = ::boba::get_const_eigen_map(matrix_A);
  auto rhs_map = ::boba::get_const_eigen_map(matrix_B);
  auto output_map = ::boba::get_eigen_map(matrix_C);
  Eigen::PartialPivLU<EigenMatrix> solver(A_map);
  output_map = solver.solve(rhs_map);
}

/**
 * \brief Forms a partial-pivot LU factorization with Eigen.
 *
 * On return, `P * A = L * U`.
 *
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1PartialPivLU.html
 */
template <typename data_t>
void factor_lu_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_P)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  BOBA_CALI_BEGIN("initialize_lu");

  boba_always_assert_equal(input.rows(), input.cols(), "LU requires nonsingular square matrix");

  Eigen::PartialPivLU<EigenMatrix> eigen_lu;
  BOBA_CALI_SWITCH("initialize_lu", "compute_LU");
  Eigen::Map<const EigenMatrix> eigen_input = ::boba::get_const_eigen_map(input);
  eigen_lu.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_LU", "form_L");
  size_t rows = static_cast<size_t>(eigen_input.rows());
  size_t cols = static_cast<size_t>(eigen_input.cols());

  matrix_L.resize({rows, cols});
  matrix_L.fill_with_zeros();

  Eigen::Map<EigenMatrix> Lmap = ::boba::get_eigen_map(matrix_L);
  Lmap.template triangularView<Eigen::StrictlyLower>() = eigen_lu.matrixLU();

  matrix_U.resize({rows, cols});
  matrix_U.erase_lower_triangular();
  Eigen::Map<EigenMatrix> Umap = ::boba::get_eigen_map(matrix_U);
  Umap = eigen_lu.matrixLU().template triangularView<Eigen::Upper>();

  BOBA_CALI_SWITCH("form_L", "form_U");
  auto perms = eigen_lu.permutationP().indices();
  size_t perms_rows = static_cast<size_t>(perms.rows());

  BOBA_CALI_SWITCH("form_U", "form_P");
  matrix_P.resize({perms_rows});
  auto P_view = matrix_P.view();
  auto L_view = matrix_L.view();
  for (auto i = 0_z; i < perms_rows; i++)
  {
    P_view(i) = static_cast<index_t>(perms[static_cast<Eigen::Index>(i)]);
    L_view({i, i}) = 1.0;
  }
  BOBA_CALI_END("form_P");
}

/**
 * \brief Forms a full-pivot LU factorization with Eigen.
 *
 * On return, `P * A * Q = L * U`.
 *
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1FullPivLU.html
 */
template <typename data_t>
void factor_lufull_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_P,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_Q)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  BOBA_CALI_BEGIN("initialize_lu");

  Eigen::FullPivLU<EigenMatrix> eigen_lu;
  BOBA_CALI_SWITCH("initialize_lu", "compute_LU");
  Eigen::Map<const EigenMatrix> eigen_input = ::boba::get_const_eigen_map(input);
  eigen_lu.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_LU", "form_L");
  size_t rows = static_cast<size_t>(eigen_input.rows());
  size_t cols = static_cast<size_t>(eigen_input.cols());
  size_t common = ::boba::min(rows, cols);

  matrix_L.resize({rows, common});
  matrix_L.fill_with_zeros();

  Eigen::Map<EigenMatrix> Lmap = ::boba::get_eigen_map(matrix_L);
  Lmap.template triangularView<Eigen::StrictlyLower>() = eigen_lu.matrixLU();

  matrix_U.resize({rows, cols});
  matrix_U.erase_lower_triangular();
  Eigen::Map<EigenMatrix> Umap = ::boba::get_eigen_map(matrix_U);
  Umap = eigen_lu.matrixLU().template triangularView<Eigen::Upper>();
  matrix_U.resize({common, cols});

  BOBA_CALI_SWITCH("form_L", "form_U");
  auto P_perms = eigen_lu.permutationP().indices();
  size_t P_perms_rows = static_cast<size_t>(P_perms.rows());
  BOBA_CALI_SWITCH("form_U", "form_P");
  matrix_P.resize({P_perms_rows});

  auto P_view = matrix_P.view();
  auto L_view = matrix_L.view();
  for (auto i = 0_z; i < P_perms_rows; i++)
  {
    P_view(i) = static_cast<index_t>(P_perms[static_cast<Eigen::Index>(i)]);
  }

  for (auto i = 0_z; i < common; i++)
  {
    L_view({i, i}) = 1.0;
  }

  auto Q_perms = eigen_lu.permutationQ().indices();
  size_t Q_perms_rows = static_cast<size_t>(Q_perms.rows());
  BOBA_CALI_SWITCH("form_P", "form_Q");
  matrix_Q.resize({Q_perms_rows});

  auto Q_view = matrix_Q.view();
  for (auto i = 0_z; i < Q_perms_rows; i++)
  {
    Q_view(i) = static_cast<index_t>(Q_perms[static_cast<Eigen::Index>(i)]);
  }
  BOBA_CALI_END("form_Q");
}

/**
 * \brief Computes Eigen's economical SVD and returns the nonzero singular values.
 *
 * The helper forms thin `U` and `V` and leaves additional truncation policy to
 * the caller.
 *
 * \see https://eigen.tuxfamily.org/dox/classEigen_1_1BDCSVD.html
 */
template <typename data_t>
void factor_svd_eigen(
  Matrix<execution_space::CPU, data_t>& input,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  Vector<execution_space::CPU, real_type_t<data_t>>& singular_values,
  Matrix<execution_space::CPU, data_t>& matrix_V,
  size_t& number_nonzeros,
  real_type_t<data_t> tolerance_relative)
{
  using EigenMatrix = Eigen::Matrix<data_t, Eigen::Dynamic, Eigen::Dynamic>;
  using EigenVector = Eigen::Matrix<real_type_t<data_t>, Eigen::Dynamic, 1>;
  BOBA_CALI_BEGIN("compute_svd");

  Eigen::BDCSVD<EigenMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV> eigen_svd;
  eigen_svd.setThreshold(tolerance_relative);

  Eigen::Map<EigenMatrix> eigen_input = ::boba::get_eigen_map(input);

  size_t input_rows = static_cast<size_t>(eigen_input.rows());
  size_t input_cols = static_cast<size_t>(eigen_input.cols());

  eigen_svd.compute(eigen_input);

  BOBA_CALI_SWITCH("compute_svd", "set_singular_values");
  number_nonzeros = static_cast<size_t>(eigen_svd.nonzeroSingularValues());
  if (number_nonzeros == 0)
  {
    singular_values.resize({1});
    singular_values.fill_with_zeros();
    matrix_U.resize({input_rows, 1});
    matrix_U.fill_with_zeros();
    matrix_V.resize({input_cols, 1});
    matrix_V.fill_with_zeros();
    BOBA_CALI_END("set_singular_values");
    return;
  }

  singular_values.resize({number_nonzeros});
  Eigen::Map<EigenVector> Smap = ::boba::get_eigen_map(singular_values);
  Smap = eigen_svd.singularValues().head(static_cast<Eigen::Index>(number_nonzeros));

  BOBA_CALI_SWITCH("set_singular_values", "set_U");
  matrix_U.resize({input_rows, number_nonzeros});
  Eigen::Map<EigenMatrix> Umap = ::boba::get_eigen_map(matrix_U);
  Umap = eigen_svd.matrixU().leftCols(static_cast<Eigen::Index>(number_nonzeros));

  BOBA_CALI_SWITCH("set_U", "set_V");
  matrix_V.resize({input_cols, number_nonzeros});
  Eigen::Map<EigenMatrix> Vmap = ::boba::get_eigen_map(matrix_V);
  Vmap = eigen_svd.matrixV().leftCols(static_cast<Eigen::Index>(number_nonzeros));
  BOBA_CALI_END("set_V");
}

#else

/**
 * @brief Reports that Eigen GMRES support is unavailable.
 * @tparam space Execution space of the input matrix and vector.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @param matrix System matrix.
 * @param input Right-hand side vector.
 * @param tolerance_relative Relative convergence tolerance.
 * @param maximum_iterations Maximum GMRES iteration count.
 * @return This function does not return successfully.
 */
template <execution_space space, typename data_t>
[[nodiscard]]
boba::Vector<host_space, data_t> eigen_gmres(
  const boba::Matrix<space, data_t>& matrix,
  const boba::Vector<space, data_t>& input,
  data_t tolerance_relative = 1.0e-7,
  size_t maximum_iterations = 30)
{
  boba::Vector<space, data_t> output;
  detail::ignore(matrix);
  detail::ignore(input);
  detail::ignore(tolerance_relative);
  detail::ignore(maximum_iterations);
  boba_error("You have called eigen's GMRES routine, but BOBA_ENABLE_EIGEN is set to false.");
  return output;
}

/**
 * @brief Reports that Eigen GEMM support is unavailable.
 * @tparam A_matrix_t Left operand matrix type.
 * @tparam B_matrix_t Right operand matrix type.
 * @tparam C_matrix_t Output matrix type.
 * @param matrix_A Left operand matrix.
 * @param matrix_B Right operand matrix.
 * @param matrix_C Output matrix.
 */
template <typename A_matrix_t, typename B_matrix_t, typename C_matrix_t>
void eigen_gemm(
  A_matrix_t const& matrix_A,
  B_matrix_t const& matrix_B,
  C_matrix_t& matrix_C)
{
  detail::ignore(matrix_A);
  detail::ignore(matrix_B);
  detail::ignore(matrix_C);
  boba_error("Build error! You are trying to run eigen_gemm without BOBA_ENABLE_EIGEN");
}

/**
 * @brief Reports that Eigen QR least-squares support is unavailable.
 * @tparam space Execution space of the input matrices.
 * @tparam index_t Index type.
 * @param matrix_A System matrix.
 * @param matrix_B Right-hand side matrix.
 * @param matrix_C Output solution matrix.
 */
template <execution_space space>
void ls_solve_qr_eigen(Matrix<space, double> const& matrix_A,
                       Matrix<space, double> const& matrix_B,
                       Matrix<execution_space::CPU, double>& matrix_C)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  ::boba::detail::ignore(matrix_C);
  boba_error("ls_solve_qr_eigen requires Eigen.");
}

template <execution_space space>
void ls_solve_cod_eigen(Matrix<space, double> const& matrix_A,
                        Matrix<space, double> const& matrix_B,
                        Matrix<execution_space::CPU, double>& matrix_C)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  ::boba::detail::ignore(matrix_C);
  boba_error("ls_solve_cod_eigen requires Eigen.");
}

/**
 * @brief Reports that Eigen LU least-squares support is unavailable.
 * @tparam space Execution space of the input matrices.
 * @tparam index_t Index type.
 * @param matrix_A System matrix.
 * @param matrix_B Right-hand side matrix.
 * @param matrix_C Output solution matrix.
 */
template <execution_space space>
void ls_solve_lu_eigen(Matrix<space, double> const& matrix_A,
                       Matrix<space, double> const& matrix_B,
                       Matrix<execution_space::CPU, double>& matrix_C)
{
  ::boba::detail::ignore(matrix_A);
  ::boba::detail::ignore(matrix_B);
  ::boba::detail::ignore(matrix_C);
  boba_error("ls_solve_lu_eigen requires Eigen.");
}

template <typename data_t>
void factor_cholesky_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_L);
  boba_error("factor_cholesky_eigen requires Eigen.");
}

template <typename data_t>
void factor_qr_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_Q,
  Matrix<execution_space::CPU, data_t>& matrix_R,
  size_t& ranks)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_Q);
  ::boba::detail::ignore(matrix_R);
  ::boba::detail::ignore(ranks);
  boba_error("factor_qr_eigen requires Eigen.");
}

template <typename data_t>
void factor_qrrr_eigen(
  Matrix<execution_space::CPU, data_t>& input,
  Matrix<execution_space::CPU, data_t>& matrix_Q,
  Matrix<execution_space::CPU, data_t>& matrix_R,
  size_t& ranks,
  real_type_t<data_t> qr_rank_tolerance)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_Q);
  ::boba::detail::ignore(matrix_R);
  ::boba::detail::ignore(ranks);
  ::boba::detail::ignore(qr_rank_tolerance);
  boba_error("factor_qrrr_eigen requires Eigen.");
}

template <typename data_t>
void factor_lu_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_P)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_L);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(matrix_P);
  boba_error("factor_lu_eigen requires Eigen.");
}

template <typename data_t>
void factor_lufull_eigen(
  Matrix<execution_space::CPU, data_t> const& input,
  Matrix<execution_space::CPU, data_t>& matrix_L,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_P,
  PermutationMatrix<execution_space::CPU, index_t>& matrix_Q)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_L);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(matrix_P);
  ::boba::detail::ignore(matrix_Q);
  boba_error("factor_lufull_eigen requires Eigen.");
}

template <typename data_t>
void factor_svd_eigen(
  Matrix<execution_space::CPU, data_t>& input,
  Matrix<execution_space::CPU, data_t>& matrix_U,
  Vector<execution_space::CPU, real_type_t<data_t>>& singular_values,
  Matrix<execution_space::CPU, data_t>& matrix_V,
  size_t& number_nonzeros,
  real_type_t<data_t> tolerance_relative)
{
  ::boba::detail::ignore(input);
  ::boba::detail::ignore(matrix_U);
  ::boba::detail::ignore(singular_values);
  ::boba::detail::ignore(matrix_V);
  ::boba::detail::ignore(number_nonzeros);
  ::boba::detail::ignore(tolerance_relative);
  boba_error("factor_svd_eigen requires Eigen.");
}

#endif

} // namespace detail
} // namespace boba
