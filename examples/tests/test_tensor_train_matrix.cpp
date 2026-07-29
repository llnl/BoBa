// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common.hpp"
#include "common_ttm.hpp"

#ifdef BOBA_ENABLE_EIGEN
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/SparseLU"
#endif

//////////////////////////////////////////////////////////////////////
/*
  This test draws on everything we have learned about tensor trains and expands this to tensor train matrices.
  You'll learn:
    -tensor train matrices
      - how to create a tensor train matrix from 1d discretization matrices
      - round
      - how to compute the inverse of a tensor train matrix with compute_inverse

  Creates the discrete Laplacian operator or the 'advection' operator

    Laplacian = \Delta = \grad\cdot\grad

    Derivative = (1,...,1)\cdot \grad

  Uses Dirichlet boundary conditions

  Computes the inverse of the discrete operator

    Laplacian^{-1} or Derivative^{-1}

  Optionally also creates the conventional equivalent of these objects
  for error checking

*/
//////////////////////////////////////////////////////////////////////

constexpr boba::execution_space space = boba::default_execution_space;
constexpr boba::tictoc_units tictoc_units = boba::tictoc_units::milliseconds;

bool default_create_full = true;
bool default_check_error = true;

#if defined(BOBA_FULL_TEST)
bool default_compute_full_inverse = true;
bool default_compute_ttm_inverse_oseledets = false;
bool default_compute_ttm_inverse_alternate = false;
bool default_compute_ttm_split_inverse = false;
#else
bool default_compute_full_inverse = true;
bool default_compute_ttm_inverse_oseledets = boba::is_boba_debug_mode();
bool default_compute_ttm_inverse_alternate = boba::is_boba_debug_mode();
bool default_compute_ttm_split_inverse = boba::is_boba_debug_mode();
#endif

enum inverse_schemes : int
{
  ttm_oseledets = 0,
  ttm_alternate,
  ttm_multistep,
  ttm_split,
  number_of_schemes,
  full_inverse,
  ttm_ipsvd
};

static const ::boba::Array<const char*, inverse_schemes::number_of_schemes> inverse_schemes_strings{
  "oseledets",
  "alternate",
  "multistep",
  "split",
  //  "ipsvd"
};

static const ::boba::Array<const char*, inverse_schemes::number_of_schemes> inverse_schemes_names{
  "Oseledets quadratic newton solve",
  "Oseledets non-quadratic newton solve",
  "Fast Iteration",
  "Split matrices non-quadratic newton solve",
  //  "Split matrices with SVD of inner-product term"
};

template <size_t dimension>
struct run
{

  // runs the problem with a specified set of parameters
  run(
    size_t number_elements_1d,
    ::boba::Array<double, inverse_schemes::number_of_schemes>& times,
    ::boba::Array<double, inverse_schemes::number_of_schemes>& errors,
    ::boba::Array<double, inverse_schemes::number_of_schemes>& final_sizes,
    bool& check)
  {
    bool verbose_tests = boba::is_env_nonempty("VERBOSE");

    if (verbose_tests)
    {
      std::cout << "%---------------------------------------------------------" << std::endl;
      std::cout << "% Running test " << std::endl;
      std::cout << "% number_elements_1d = " << number_elements_1d << std::endl;
      std::cout << "% dimension = " << dimension << std::endl;
      std::cout << "%---------------------------------------------------------" << std::endl;
    }

    size_t N = number_elements_1d;
    size_t N_full = boba::pow(N, dimension);

    using tensor_type = ::boba::Tensor<dimension, space, double>;
    using view_type = typename tensor_type::view_type;

    auto sizes = ::boba::filled_array<dimension>(N);
    auto strides = view_type::precompute_strides(sizes);

    bool create_full = default_create_full;
    bool compute_full_inverse = default_compute_full_inverse;
    bool check_error = default_check_error;

    bool verbose_ttm_inverse = boba::is_env_nonempty("VERBOSE_INVERSE");

    // Use Laplacian or derivative matrix
    bool laplacian_matrix = boba::is_env_empty("ADVECTION_MATRIX");

    if (N_full > 46340)
    {
      // int overflow on full size = N_full*N_full
      create_full = false;
      if (verbose_tests)
      {
        std::cout << "int overflow " << std::endl;
        boba_print(N_full);
      }
    }
    if (N_full > 44000)
    {
      // bad alloc ahead
      if (verbose_tests)
      {
        std::cout << "bad alloc ahead " << std::endl;
        boba_print(N_full);
      }
    }

    if (verbose_tests)
    {
      std::cout << "% Create tensor-train Discretization matrix from 1d matrices" << std::endl;
    }
    checkpoint();
    boba::TicToc<tictoc_units> tictoc_ttm_from_1d;

    // call helper function to generate discretizations
    common_ttm<dimension, space, double> operators(N, 1.0, false);

    checkpoint();
    double dx = operators.dx;
    double dx2 = operators.dx2;

    checkpoint();
    if (verbose_tests && create_full)
    {
      std::cout << "% Create full Laplacian/derivative matrix " << std::endl;
    }

    checkpoint();
    boba::Matrix<boba::execution_space::CPU, double> full_host({1, 1});
    boba::Vector<boba::execution_space::CPU, double> full_rhs({1});
    full_host.rename("full_host_matrix");
    size_t full_sparse_nnz = 0;
    checkpoint();
#ifdef BOBA_ENABLE_EIGEN
    auto eigen_N_full = static_cast<Eigen::Index>(N_full);
    if (create_full)
    {
      checkpoint();
      Eigen::SparseMatrix<double> full_sparse(eigen_N_full, eigen_N_full);
      checkpoint();
      full_rhs.resize(N_full);
      for (size_t row = 0; row < N_full; row++)
      {
        auto eigen_row = static_cast<Eigen::Index>(row);
        auto indices = view_type::multiindex(sizes, row);
        bool is_boundary = false;
        for (size_t d = 0; d < dimension; d++)
        {
          if ((indices[d] == 0) || (indices[d] == N - 1))
          {
            is_boundary = true;
          }
        }

        if (is_boundary)
        {
          // Fix the boundary terms to prescribed values of 1.0
          double value = 1;
          full_sparse.insert(eigen_row, eigen_row) = value;
          full_rhs({row}) = 1;
        }
        else
        {
          full_rhs({row}) = (laplacian_matrix ? 1. * dx2 : 1. * dx);
          // For non-boundary terms, add 1s to all 'neighboring' dofs
          full_sparse.insert(eigen_row, eigen_row) = (laplacian_matrix ? -2. * double(dimension) : 1. * double(dimension));
          for (size_t d = 0; d < dimension; d++)
          {
            auto index = indices;
            {
              index[d] = indices[d] - 1;
              size_t col = view_type::index(strides, index);
              auto eigen_col = static_cast<Eigen::Index>(col);
              full_sparse.insert(eigen_row, eigen_col) = (laplacian_matrix ? 1.0 : -1.0);
            }
            {
              index[d] = indices[d] + 1;
              size_t col = view_type::index(strides, index);
              auto eigen_col = static_cast<Eigen::Index>(col);
              full_sparse.insert(eigen_row, eigen_col) = (laplacian_matrix ? 1.0 : 0.0);
            }
          }
        }
      }
      checkpoint();
      full_sparse.makeCompressed();
      checkpoint();
      full_host.resize({N_full, N_full});
      auto full_host_map = ::boba::get_eigen_map(full_host);
      checkpoint();
      full_host_map = full_sparse;
      checkpoint();
      full_sparse_nnz = size_t(full_sparse.nonZeros());
      if (boba::is_env_nonempty("BOBA_VERBOSE"))
      {
        full_host.print();
      }
    }
#else
    if (create_full)
    {
      boba_error("Eigen not enabled.");
    }
#endif

    checkpoint();
    boba::TensorTrainMatrix<dimension, space, double> ttm_from_1d(sizes, sizes);
    ttm_from_1d.fill_with_zeros();
    if (laplacian_matrix)
    {
      ttm_from_1d.TensorTrainMatrix_add(operators.laplacian_interior);
    }
    else
    {
      ttm_from_1d.TensorTrainMatrix_add(operators.advection_forward_interior);
    }
    ttm_from_1d.TensorTrainMatrix_add(operators.boundaries);

    auto time_ttm_from_1d = tictoc_ttm_from_1d.timing();

    if (verbose_tests)
    {
      std::cout << boba::write_indent(1)
                << "Generate TTM from 1D matrices, timing "
                << tictoc_ttm_from_1d.units_string << ": "
                << time_ttm_from_1d << std::endl;
    }

    checkpoint();
    if (boba::is_env_nonempty("BOBA_VERBOSE"))
    {
      ttm_from_1d.decompress().print();
    }
    if (create_full && check_error)
    {
      if (verbose_tests)
      {
        std::cout << boba::write_indent(1)
                  << "error_1d_host = ";
      }
      ::boba::Matrix<space, double> full_space(full_host);
      double error_1d_host = ::boba::norm_difference_inf(ttm_from_1d.decompress(), full_space);
      if (verbose_tests)
      {
        std::cout << error_1d_host << std::endl;
      }
      pass_or_fail(check, error_1d_host, 1.0e-10);
    }

    if (verbose_tests)
    {
      std::cout << "% round TTM" << std::endl;
    }

    boba::TensorTrainMatrix<dimension, space, double> ttm_from_1d_round(sizes, sizes);
    ttm_from_1d_round = ttm_from_1d;

    checkpoint();
    ttm_from_1d_round.rename("ttm_from_1d_round");
    boba::TicToc<tictoc_units> tictoc_ttm_round;
    ttm_from_1d_round.round();
    auto time_ttm_round = tictoc_ttm_round.timing();

    checkpoint();
    if (verbose_tests)
    {
      std::cout << " Tensor Train Matrix Representation " << std::endl;
      std::cout << "   rounding time " << tictoc_ttm_round.units_string << ": " << time_ttm_round << std::endl;
      std::cout << "   Matrix nonzeros        : " << full_sparse_nnz << std::endl;
      std::cout << "   Boba size              : " << ttm_from_1d_round.get_number_elements() << std::endl;
      std::cout << "   Ratio                  : " << double(full_sparse_nnz) / double(ttm_from_1d_round.get_number_elements()) << std::endl;
      std::cout << "   Compression rate       : " << ttm_from_1d_round.compression_rate() << "x" << std::endl;
    }
    if (create_full && check_error)
    {
      if (verbose_tests)
      {
        std::cout << boba::write_indent(1)
                  << "error_1d_host_round = ";
      }
      ::boba::Matrix<space, double> full_space(full_host);
      double error_1d_host_round = ::boba::norm_difference_inf(ttm_from_1d_round.decompress(), full_space);
      if (verbose_tests)
      {
        std::cout << error_1d_host_round << std::endl;
      }
      pass_or_fail(check, error_1d_host_round, 1.0e-10);
    }

    checkpoint();
    boba::TensorTrainSplitMatrix<dimension, space, double> ttm_split_round(ttm_from_1d);
    checkpoint();
    ttm_split_round.rename("ttm_split_round");
    ttm_split_round.round();

    checkpoint();
    if (create_full)
    {
      ::boba::Matrix<space, double> full_space(full_host);
      double error_1d_host_split_round = ::boba::norm_difference_inf(ttm_split_round.decompress(), full_space);

      if (verbose_tests)
      {
        std::cout << " Tensor Train Split Matrix Representation " << std::endl;
        std::cout << "   Matrix size            : " << full_host.size() << std::endl;
        std::cout << "   Boba size              : " << ttm_split_round.get_number_elements() << std::endl;
        std::cout << "   Ratio                  : " << double(full_host.size()) / double(ttm_from_1d_round.get_number_elements()) << std::endl;
        std::cout << "   Compression rate       : " << ttm_split_round.compression_rate() << "x" << std::endl;
      }
      if (verbose_tests)
      {
        std::cout << boba::write_indent(1)
                  << "error_1d_host_split_round = " << error_1d_host_split_round
                  << std::endl;
      }
    }

    checkpoint();
    boba::Matrix<boba::execution_space::CPU, double> full_inverse({1, 1});
    full_inverse.rename("full_inverse");
    boba::Tensor<dimension, boba::execution_space::CPU, double> full_inverse_soln;
    checkpoint();
    double full_time = 0;
    size_t full_sparse_inverse_nnz = 0;
#ifdef BOBA_ENABLE_EIGEN
    if (create_full)
    {
      full_inverse_soln.resize(sizes);
      if (compute_full_inverse)
      {
        Eigen::SparseMatrix<double> full_sparse(eigen_N_full, eigen_N_full);

        for (size_t row = 0; row < N_full; row++)
        {
          auto eigen_row = static_cast<Eigen::Index>(row);
          auto indices = view_type::multiindex(sizes, row);
          bool is_boundary = false;
          for (size_t d = 0; d < dimension; d++)
          {
            if ((indices[d] == 0_z) || (indices[d] == N - 1_z))
            {
              is_boundary = true;
            }
          }

          if (is_boundary)
          {
            // Fix the boundary terms to prescribed values of 1.0
            double value = 1;
            full_sparse.insert(eigen_row, eigen_row) = value;
          }
          else
          {
            // For non-boundary terms, add 1s to all 'neighboring' dofs
            full_sparse.insert(eigen_row, eigen_row) = (laplacian_matrix ? -2. * double(dimension) : 1. * double(dimension));
            for (size_t d = 0; d < dimension; d++)
            {
              auto index = indices;
              {
                index[d] = indices[d] - 1;
                size_t col = view_type::index(strides, index);
                auto eigen_col = static_cast<Eigen::Index>(col);

                full_sparse.insert(eigen_row, eigen_col) = (laplacian_matrix ? 1.0 : -1.0);
              }
              {
                index[d] = indices[d] + 1;
                size_t col = view_type::index(strides, index);
                auto eigen_col = static_cast<Eigen::Index>(col);

                full_sparse.insert(eigen_row, eigen_col) = (laplacian_matrix ? 1.0 : 0.0);
              }
            }
          }
        }
        checkpoint();
        full_sparse.makeCompressed();

        Eigen::SparseMatrix<double> full_sparse_inverse(eigen_N_full, eigen_N_full);
        if (verbose_tests)
        {
          std::cout << "% Full Matrix inverse " << std::endl;
        }
        Eigen::VectorXd output(N_full);

        Eigen::VectorXd rhs(N_full);
        rhs = ::boba::get_eigen_map(full_rhs);

        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        checkpoint();
        boba::TicToc<tictoc_units> tictoc_inverse_full;
        if constexpr (not(::boba::is_device_context()))
        {
          solver.analyzePattern(full_sparse);
          solver.factorize(full_sparse);
          output = solver.solve(rhs);
        }
        full_time = tictoc_inverse_full.timing();
        auto full_inverse_soln_data = full_inverse_soln.data();
        for (size_t i = 0; i < static_cast<size_t>(rhs.size()); ++i)
        {
          full_inverse_soln_data[i] = output(static_cast<Eigen::Index>(i));
        }
        if (verbose_tests)
        {
          std::cout << "   Timing " << tictoc_inverse_full.units_string << "            : " << full_time << std::endl;
        }
      }
    }
#else
    if (create_full)
    {
      boba_error("Eigen not enabled.");
    }
#endif

    //
    // Compute the right-hand side
    //
    boba::TensorTrain<dimension, space, double> rhs(sizes);
    rhs.rename("right_hand_side");
    rhs.fill_with_zeros();
    {
      boba::TensorTrain<dimension, space, double> rhs_temp(sizes);
      rhs_temp.rename("rhs_temp");
      rhs_temp.fill_with(1.0);
      boba::TensorTrain<dimension, space, double> rhs_temp_out(sizes);
      rhs_temp_out.rename("rhs_temp_out");
      rhs_temp_out.fill_with(1.0);

      //   filter the boundary out
      rhs_temp_out = operators.identity_interior * rhs_temp;
      rhs_temp_out *= (laplacian_matrix ? 1. * dx2 : 1. * dx);
      rhs += rhs_temp_out;

      //   filter the domain out
      rhs_temp_out = operators.boundaries * rhs_temp;
      rhs += rhs_temp_out;
    }

    //
    // Compute the inverse
    //
    double tolerance = 1.0e-10;
    size_t maximum_iterations = 50;
    double alpha = 0.0001;
    std::string scheme_env = boba::get_env("SCHEME");

    for (size_t scheme = 0; scheme < inverse_schemes::number_of_schemes; scheme++)
    {
      bool scheme_defined = boba::is_env_nonempty("SCHEME");
      bool scheme_match = boba::env_match("SCHEME", inverse_schemes_strings[scheme]);
      if (scheme_defined && !scheme_match)
      {
        continue;
      }

      double residual = -1.0;

      if (verbose_tests)
      {
        std::cout << "% Tensor Train Matrix inverse - " << inverse_schemes_names[scheme] << std::endl;
      }
      {
        checkpoint();
        boba::TensorTrain<dimension, space, double> output(sizes);

        // Initialize with alpha*A^T
        boba::TensorTrainMatrix<dimension, space, double> ttm_inverse(sizes, sizes);
        ttm_inverse.rename("ttm_inverse");
        checkpoint();
        ttm_inverse = ttm_from_1d.transpose();
        checkpoint();
        ttm_inverse *= alpha;
        double time_scheme = -1.0;
        bool ttm_success = false;
        if (scheme < inverse_schemes::ttm_ipsvd)
        {
          boba::TicToc<tictoc_units> tictoc_inverse_ttm;
          ttm_success = ttm_from_1d.compute_inverse(
            ttm_inverse,
            residual,
            tolerance,
            maximum_iterations,
            scheme,
            verbose_ttm_inverse);
          checkpoint();
          output = ttm_inverse * rhs;
          time_scheme = tictoc_inverse_ttm.timing();
          checkpoint();
          if (ttm_success)
          {
            if (check_error)
            {
              ::boba::Tensor<dimension, space, double> full_inverse_space(full_inverse_soln);
              double error_1d_host_inverse = ::boba::norm_difference_inf(output.decompress(), full_inverse_space);
              errors[scheme] = error_1d_host_inverse;
            }
            times[scheme] = time_scheme;
            final_sizes[scheme] = ttm_inverse.get_number_elements();
          }
        }
        else if (scheme >= inverse_schemes::ttm_ipsvd)
        {
          boba::TensorTrainSplitMatrix<dimension, space, double> ttm_split_inverse(sizes, sizes);
          ttm_split_inverse.rename("ttm_split_inverse");
          ttm_inverse = ttm_from_1d_round.transpose();
          ttm_split_inverse.from_ttm(ttm_inverse);
          ttm_split_inverse *= alpha;
          ttm_split_inverse.round();
          boba::TicToc<tictoc_units> tictoc_inverse_ttm;
          ttm_success = ttm_split_round.compute_inverse(
            ttm_split_inverse,
            residual,
            tolerance,
            maximum_iterations,
            (scheme == inverse_schemes::ttm_ipsvd),
            verbose_ttm_inverse);
          checkpoint();
          output = ttm_split_inverse * rhs;
          time_scheme = tictoc_inverse_ttm.timing();
          checkpoint();
          if (ttm_success)
          {
            if (check_error)
            {
              ::boba::Tensor<dimension, space, double> full_inverse_space(full_inverse_soln);
              double error_1d_host_inverse = ::boba::norm_difference_inf(output.decompress(), full_inverse_space);
              errors[scheme] = error_1d_host_inverse;
            }
            times[scheme] = time_scheme;
            final_sizes[scheme] = ttm_inverse.get_number_elements();
          }
        }
        if (ttm_success)
        {
          if (verbose_tests)
          {
            std::cout << "%   Inverse Timing "
                      << tictoc_ttm_round.units_string
                      << "            : "
                      << time_scheme << std::endl;
            std::cout << "%   Residual : " << residual << std::endl;
            std::cout << "%   Boba size              : " << ttm_inverse.get_number_elements() << std::endl;
            std::cout << "%   Full size              : " << full_sparse_inverse_nnz << std::endl;
            std::cout << "%   Ratio                  : " << full_sparse_inverse_nnz / ttm_inverse.get_number_elements() << std::endl;
            std::cout << "%   Compression rate       : " << ttm_inverse.compression_rate() << "x" << std::endl;
          }
          checkpoint();
          if (create_full && compute_full_inverse)
          {
            if (verbose_tests)
            {
              if (time_scheme < full_time)
              {
                std::cout << "%   Speedup                : " << divide_string(full_time, time_scheme) << "x" << std::endl;
              }
              else
              {
                std::cout << "%   Slowdown               : " << divide_string(time_scheme, full_time) << "x" << std::endl;
              }
              if (check_error)
              {
                std::cout << boba::write_indent(1)
                          << "%error_1d_host_inverse = "
                          << errors[scheme] << std::endl;
                pass_or_fail(check, errors[scheme], 1.0e-3);
              }
            }
          }
        }
        else
        {
          times[scheme] = -6;
          if (verbose_tests)
          {
            std::cout << "   Tensor train matrix inverse failed" << std::endl;
          }
        }
      }
    }
  }
};

template <size_t dimension>
struct run_wrapper
{

  // runs the problem several times and averages the results

  run_wrapper(size_t number_elements, bool& check, size_t averages = 1)
  {
    size_t number_elements_1d = number_elements;
    auto total_times = ::boba::filled_array<inverse_schemes::number_of_schemes>(static_cast<double>(0));
    auto errors = ::boba::filled_array<inverse_schemes::number_of_schemes>(static_cast<double>(-1));
    auto final_sizes = ::boba::filled_array<inverse_schemes::number_of_schemes>(static_cast<double>(-1));
    for (size_t t = 0; t < inverse_schemes::number_of_schemes; t++)
    {
      total_times[t] = 0;
      errors[t] = 0;
      final_sizes[t] = 0;
    }
    for (size_t r = 0; r < averages; r++)
    {
      auto times = ::boba::filled_array<inverse_schemes::number_of_schemes>(static_cast<double>(-1));
      for (size_t t = 0; t < inverse_schemes::number_of_schemes; t++)
      {
        times[t] = 0;
      }

      run<dimension> runner(
        number_elements_1d,
        times,
        errors,
        final_sizes,
        check);

      for (size_t t = 0; t < inverse_schemes::number_of_schemes; t++)
      {
        bool no_failure = (times[t] >= 0);
        bool no_previous_failures = (total_times[t] >= 0);
        if (no_failure && no_previous_failures)
        {
          total_times[t] += times[t];
        }
        else if (no_previous_failures)
        {
          total_times[t] = times[t];
        }
      }
    }
    for (size_t t = 0; t < inverse_schemes::number_of_schemes; t++)
    {
      bool no_failures = (total_times[t] >= 0);
      if (no_failures)
      {
        total_times[t] /= double(averages);
      }
    }
    if (boba::is_env_empty("VERBOSE"))
    {
      // Non-verbose output
      std::string type = boba::is_env_empty("ADVECTION_MATRIX") ? "laplacian" : "derivative";
      std::cout << type << " "
                << ::boba::default_execution_space_name() << " "
                << dimension << " "
                << number_elements_1d << " ";
      // -1 means test was skipped
      // -6 means test failed
      for (size_t t = 0; t < inverse_schemes::number_of_schemes; t++)
      {
        std::cout << total_times[t] << " ";
        std::cout << errors[t] << " ";
        std::cout << final_sizes[t] << " ";
      }
      std::cout << std::endl;
    }
  }
};

int main(int argc, char* argv[])
{

  if (boba::is_env_nonempty("VERBOSE"))
  {
    boba::splash();
    std::cout << "Tests for boba tensor train implementation" << std::endl;
  }
  boba::init();

  BOBA_CALI_EXTERNAL_MARK

  bool check = true;
  checkpoint();

  size_t number_elements = 3;
  size_t dimension = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(number_elements,
                             "-n",
                             "--resolution",
                             "Number of elements in each dimension.");

  args.add_optional_argument(dimension,
                             "-d",
                             "--dimensions",
                             "Dimensions in which to run test.");

  args.parse_check();

  if (boba::is_env_empty("VERBOSE"))
  {
    std::string type = boba::is_env_empty("ADVECTION_MATRIX") ? "laplacian" : "derivative";
    // TODO<bugfix> clean this up with a contiguous_name enum and property enum
    std::cout << "matrix_type "
              << "execution_space "
              << "use_umpire "
              << "dimension "
              << "elements_1d "
              << "full_timing "
              << "full_error "
              << "full_size "
              << "oseledets_timing "
              << "oseledets_error "
              << "oseledets_size "
              << "alternate_timing "
              << "alternate_error "
              << "alternate_size "
              << "split_timing "
              << "split_error "
              << "split_size "
              << "split_fused_timing "
              << "split_fused_error "
              << "split_fused_size "
              << std::endl;
  }

  size_t averages = 1;

  if (dimension == 0)
  {
    run_wrapper<1> runner1(number_elements, check, averages);
    run_wrapper<2> runner2(number_elements, check, averages);
    run_wrapper<3> runner3(number_elements, check, averages);
  }
  if (dimension == 1)
  {
    run_wrapper<1> runner(number_elements, check, averages);
  }
  if (dimension == 2)
  {
    run_wrapper<2> runner(number_elements, check, averages);
  }
  if (dimension == 3)
  {
    run_wrapper<3> runner(number_elements, check, averages);
  }
  if (dimension == 4)
  {
    run_wrapper<4> runner(number_elements, check, averages);
  }
  if (dimension == 5)
  {
    run_wrapper<5> runner(number_elements, check, averages);
  }
  if (dimension == 6)
  {
    run_wrapper<6> runner(number_elements, check, averages);
  }
  if (dimension == 9)
  {
    run_wrapper<9> runner(number_elements, check, averages);
  }
  if (dimension == 12)
  {
    run_wrapper<12> runner(number_elements, check, averages);
  }

  checkpoint();
  boba::finalize();
  return final_check(check);
}
