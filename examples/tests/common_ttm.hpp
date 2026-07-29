// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "common.hpp"

/*
  Creates the discrete Laplacian operator, gradient operator, and advection operator

    Laplacian = \Delta = \grad\cdot\grad

    Derivative = (1,...,1)\cdot \grad

    [Derivative]_interior => defined in interior of domain only
    [Derivative]_periodic => sets periodic BCs

    Other types of BCs must be treated separately by projecting to

    identity_boundary = projection to boundary
    identity_interior = projection to interior

*/

template <size_t dimension, ::boba::execution_space space, typename data_t>
struct common_ttm
{

  size_t size_1d;
  double dx;
  double dx2;
  double odx;
  double odx2;

  boba::Tensor<4, space, double> laplacian_1d_interior;
  boba::Tensor<4, space, double> laplacian_1d_periodic;

  boba::Tensor<4, space, double> derivative_1d_forward_interior;
  boba::Tensor<4, space, double> derivative_1d_forward_periodic;

  boba::Tensor<4, space, double> derivative_1d_backward_interior;
  boba::Tensor<4, space, double> derivative_1d_backward_periodic;

  boba::Tensor<4, space, double> identity_1d;
  boba::Tensor<4, space, double> identity_1d_interior;
  boba::Tensor<4, space, double> identity_1d_boundary;

  boba::TensorTrainMatrix<dimension, space, double> laplacian_interior;
  boba::TensorTrainMatrix<dimension, space, double> laplacian_periodic;

  boba::TensorTrainMatrix<dimension, space, double> gradient_forward_interior[dimension];
  boba::TensorTrainMatrix<dimension, space, double> gradient_forward_periodic[dimension];

  boba::TensorTrainMatrix<dimension, space, double> gradient_backward_interior[dimension];
  boba::TensorTrainMatrix<dimension, space, double> gradient_backward_periodic[dimension];

  boba::TensorTrainMatrix<dimension, space, double> advection_forward_interior;
  boba::TensorTrainMatrix<dimension, space, double> advection_forward_periodic;

  boba::TensorTrainMatrix<dimension, space, double> advection_backward_interior;
  boba::TensorTrainMatrix<dimension, space, double> advection_backward_periodic;

  boba::TensorTrainMatrix<dimension, space, double> identity;
  boba::TensorTrainMatrix<dimension, space, double> identity_interior;
  boba::TensorTrainMatrix<dimension, space, double> identity_boundary;

  // ones on the boundary, zero elsewhere
  boba::TensorTrainMatrix<dimension, space, double> boundaries;

  // ones everywhere except the boundary
  boba::TensorTrainMatrix<dimension, space, double> domain;

  // integration = {dx/2, dx, dx, ..., dx/2}
  boba::TensorTrainMatrix<dimension, space, double> ttm_integration;
  boba::TensorTrainMatrix<dimension, space, double> ttm_integration_domain;
  boba::TensorTrain<dimension, space, double> tt_integration;

  common_ttm()
  {
  }

  common_ttm(
    size_t N_input,
    double domain_size,
    bool scale_by_dx_factor)
  {
    this->generate(
      N_input,
      domain_size,
      scale_by_dx_factor);
  }

  void generate(
    size_t N_input,
    double domain_size,
    bool scale_by_dx_factor)
  {
    // scale_by_dx_factor
    // if true, D = [1 -1]/dx
    // if false, D = [1 -1]

    size_t N = N_input;
    size_1d = N;

    dx = domain_size / double(N - 1);
    dx2 = dx * dx;
    odx = 1.0 / dx;
    odx2 = 1.0 / dx2;

    auto sizes = ::boba::filled_array<dimension>(N);
    auto sizes_one = ::boba::filled_array<dimension>(1_z);

    checkpoint();
    laplacian_1d_interior.resize({1, N, N, 1});
    laplacian_1d_periodic.resize({1, N, N, 1});

    laplacian_1d_interior.fill_with_zeros();
    laplacian_1d_periodic.fill_with_zeros();

    set_to_rank_one_scalar_core(laplacian_1d_periodic, 0.0);

    derivative_1d_forward_interior.resize({1, N, N, 1});
    derivative_1d_forward_periodic.resize({1, N, N, 1});

    derivative_1d_forward_interior.fill_with_zeros();
    derivative_1d_forward_periodic.fill_with_zeros();

    set_to_rank_one_scalar_core(derivative_1d_forward_interior, 0.0);
    set_to_rank_one_scalar_core(derivative_1d_forward_periodic, 0.0);

    derivative_1d_backward_interior.resize({1, N, N, 1});
    derivative_1d_backward_periodic.resize({1, N, N, 1});

    derivative_1d_backward_interior.fill_with_zeros();
    derivative_1d_backward_periodic.fill_with_zeros();

    set_to_rank_one_scalar_core(derivative_1d_backward_interior, 0.0);
    set_to_rank_one_scalar_core(derivative_1d_backward_periodic, 0.0);

    identity_1d.resize({1, N, N, 1});
    identity_1d_boundary.resize({1, N, N, 1});
    identity_1d_interior.resize({1, N, N, 1});

    identity_1d.fill_with_zeros();
    identity_1d_boundary.fill_with_zeros();
    identity_1d_interior.fill_with_zeros();

    set_to_rank_one_scalar_core(identity_1d, 0.0);
    set_to_rank_one_scalar_core(identity_1d_boundary, 0.0);
    set_to_rank_one_scalar_core(identity_1d_interior, 0.0);

    auto laplacian_1d_interior_view = laplacian_1d_interior.view();
    auto laplacian_1d_periodic_view = laplacian_1d_periodic.view();

    auto derivative_1d_forward_interior_view = derivative_1d_forward_interior.view();
    auto derivative_1d_forward_periodic_view = derivative_1d_forward_periodic.view();

    auto derivative_1d_backward_interior_view = derivative_1d_backward_interior.view();
    auto derivative_1d_backward_periodic_view = derivative_1d_backward_periodic.view();

    auto identity_1d_view = identity_1d.view();
    auto identity_1d_boundary_view = identity_1d_boundary.view();
    auto identity_1d_interior_view = identity_1d_interior.view();

    boba_always_assert_gt(N, 2_z, "Invalid size");

    checkpoint();
    ::boba::detail::loop<space>(0, N, [=] __boba_host_device__(size_t row)
    {
      bool is_first_row = (row == 0);
      bool is_last_row = (row == N - 1);
      // bool is_edge = is_first_row || is_last_row;

      // Matrix indices
      ::boba::Array<size_t, 4> ii{0, row, row, 0};

      if (is_first_row)
      {

        ::boba::Array<size_t, 4> iip1{0, row, row + 1, 0};
        ::boba::Array<size_t, 4> iip2{0, row, row + 2, 0};

        identity_1d_view(ii) = 1.0;
        identity_1d_boundary_view(ii) = 1.0;

        ::boba::Array<size_t, 4> iim1roll{0, 0, N - 1, 0};

        laplacian_1d_periodic_view(iim1roll) = 1.0;
        laplacian_1d_periodic_view(ii) = -2.0;
        laplacian_1d_periodic_view(iip1) = 1.0;

        derivative_1d_forward_periodic_view(iim1roll) = -1.0;
        derivative_1d_forward_periodic_view(ii) = 1.0;

        derivative_1d_backward_periodic_view(ii) = -1.0;
        derivative_1d_backward_periodic_view(iip1) = 1.0;
      }
      else if (is_last_row)
      {
        ::boba::Array<size_t, 4> iim1{0, row, row - 1, 0};
        ::boba::Array<size_t, 4> iim2{0, row, row - 2, 0};

        identity_1d_view(ii) = 1.0;
        identity_1d_boundary_view(ii) = 1.0;

        ::boba::Array<size_t, 4> iip1roll{0, N - 1, 0, 0};

        laplacian_1d_periodic_view(iim1) = 1.0;
        laplacian_1d_periodic_view(ii) = -2.0;
        laplacian_1d_periodic_view(iip1roll) = 1.0;

        derivative_1d_forward_periodic_view(iim1) = -1.0;
        derivative_1d_forward_periodic_view(ii) = 1.0;

        derivative_1d_backward_periodic_view(ii) = -1.0;
        derivative_1d_backward_periodic_view(iip1roll) = 1.0;
      }
      else
      {
        ::boba::Array<size_t, 4> iim1{0, row, row - 1, 0};
        ::boba::Array<size_t, 4> iip1{0, row, row + 1, 0};

        identity_1d_view(ii) = 1.0;
        identity_1d_interior_view(ii) = 1.0;

        laplacian_1d_interior_view(iim1) = 1.0;
        laplacian_1d_interior_view(ii) = -2.0;
        laplacian_1d_interior_view(iip1) = 1.0;

        laplacian_1d_periodic_view(iim1) = 1.0;
        laplacian_1d_periodic_view(ii) = -2.0;
        laplacian_1d_periodic_view(iip1) = 1.0;

        derivative_1d_forward_interior_view(iim1) = -1.0;
        derivative_1d_forward_interior_view(ii) = 1.0;

        derivative_1d_forward_periodic_view(iim1) = -1.0;
        derivative_1d_forward_periodic_view(ii) = 1.0;

        derivative_1d_backward_interior_view(ii) = -1.0;
        derivative_1d_backward_interior_view(iip1) = 1.0;

        derivative_1d_backward_periodic_view(ii) = -1.0;
        derivative_1d_backward_periodic_view(iip1) = 1.0;
      }
    });

    checkpoint();
    laplacian_interior.resize(sizes, sizes);
    laplacian_interior.rename("laplacian_interior");
    laplacian_periodic.resize(sizes, sizes);
    laplacian_periodic.rename("laplacian_periodic");

    advection_forward_interior.resize(sizes, sizes);
    advection_forward_interior.rename("advection_forward_interior");
    advection_forward_periodic.resize(sizes, sizes);
    advection_forward_periodic.rename("advection_forward_periodic");

    advection_backward_interior.resize(sizes, sizes);
    advection_backward_interior.rename("advection_backward_interior");
    advection_backward_periodic.resize(sizes, sizes);
    advection_backward_periodic.rename("advection_backward_periodic");

    identity.resize(sizes, sizes);
    identity.rename("identity");
    identity_interior.resize(sizes, sizes);
    identity_interior.rename("identity_interior");
    identity_boundary.resize(sizes, sizes);
    identity_boundary.rename("identity_boundary");

    boundaries.resize(sizes, sizes);
    boundaries.rename("boundaries");
    domain.resize(sizes, sizes);
    domain.rename("domain");

    checkpoint();
    for (size_t d = 0; d < dimension; d++)
    {
      gradient_forward_interior[d].resize(sizes, sizes);
      gradient_forward_periodic[d].resize(sizes, sizes);

      gradient_forward_interior[d].fill_with_zeros();
      gradient_forward_periodic[d].fill_with_zeros();

      gradient_forward_interior[d].rename("gradient_forward_interior_dx" + std::to_string(d));
      gradient_forward_periodic[d].rename("gradient_forward_periodic_dx" + std::to_string(d));

      gradient_backward_interior[d].resize(sizes, sizes);
      gradient_backward_periodic[d].resize(sizes, sizes);

      gradient_backward_interior[d].fill_with_zeros();
      gradient_backward_periodic[d].fill_with_zeros();

      gradient_backward_interior[d].rename("gradient_backward_interior_dx" + std::to_string(d));
      gradient_backward_periodic[d].rename("gradient_backward_periodic_dx" + std::to_string(d));
    }

    checkpoint();
    // ttm_from_1d = D x I x I x ... + I x D x I x ...
    for (size_t d = 0; d < dimension; d++)
    {
      identity.cores[d] = identity_1d;
      identity_interior.cores[d] = identity_1d_interior;
      identity_boundary.cores[d] = identity_1d_boundary;

      boba::TensorTrainMatrix<dimension, space, double> laplacian_interior_temp(sizes, sizes);
      boba::TensorTrainMatrix<dimension, space, double> laplacian_periodic_temp(sizes, sizes);

      boba::TensorTrainMatrix<dimension, space, double> advection_forward_interior_temp(sizes, sizes);
      boba::TensorTrainMatrix<dimension, space, double> advection_forward_periodic_temp(sizes, sizes);

      boba::TensorTrainMatrix<dimension, space, double> advection_backward_interior_temp(sizes, sizes);
      boba::TensorTrainMatrix<dimension, space, double> advection_backward_periodic_temp(sizes, sizes);

      laplacian_interior_temp.fill_with_zeros();
      laplacian_periodic_temp.fill_with_zeros();

      advection_forward_interior_temp.fill_with_zeros();
      advection_forward_periodic_temp.fill_with_zeros();

      advection_backward_interior_temp.fill_with_zeros();
      advection_backward_periodic_temp.fill_with_zeros();

      for (size_t di = 0; di < dimension; di++)
      {
        if (d == di)
        {
          laplacian_interior_temp.cores[di] = laplacian_1d_interior;
          laplacian_periodic_temp.cores[di] = laplacian_1d_periodic;

          advection_forward_interior_temp.cores[di] = derivative_1d_forward_interior;
          advection_forward_periodic_temp.cores[di] = derivative_1d_forward_periodic;

          advection_backward_interior_temp.cores[di] = derivative_1d_backward_interior;
          advection_backward_periodic_temp.cores[di] = derivative_1d_backward_periodic;

          gradient_forward_interior[d].cores[di] = derivative_1d_forward_interior;
          gradient_forward_periodic[d].cores[di] = derivative_1d_forward_periodic;

          gradient_backward_interior[d].cores[di] = derivative_1d_backward_interior;
          gradient_backward_periodic[d].cores[di] = derivative_1d_backward_periodic;
        }
        else
        {
          checkpoint();
          laplacian_interior_temp.cores[di] = identity_1d_interior;
          laplacian_periodic_temp.cores[di] = identity_1d;

          checkpoint();
          advection_forward_interior_temp.cores[di] = identity_1d_interior;
          advection_forward_periodic_temp.cores[di] = identity_1d;

          advection_backward_interior_temp.cores[di] = identity_1d_interior;
          advection_backward_periodic_temp.cores[di] = identity_1d;

          checkpoint();
          gradient_forward_interior[d].cores[di] = identity_1d_interior;
          gradient_forward_periodic[d].cores[di] = identity_1d;

          gradient_backward_interior[d].cores[di] = identity_1d_interior;
          gradient_backward_periodic[d].cores[di] = identity_1d;
        }
      }

      laplacian_interior.TensorTrainMatrix_add(laplacian_interior_temp);
      laplacian_periodic.TensorTrainMatrix_add(laplacian_periodic_temp);

      advection_forward_interior.TensorTrainMatrix_add(advection_forward_interior_temp);
      advection_forward_periodic.TensorTrainMatrix_add(advection_forward_periodic_temp);

      advection_backward_interior.TensorTrainMatrix_add(advection_backward_interior_temp);
      advection_backward_periodic.TensorTrainMatrix_add(advection_backward_periodic_temp);
    }

    checkpoint();
    if (scale_by_dx_factor)
    {
      laplacian_interior *= odx2;
      laplacian_periodic *= odx2;

      advection_forward_interior *= odx;
      advection_forward_periodic *= odx;

      advection_backward_interior *= odx;
      advection_backward_periodic *= odx;

      for (size_t d = 0; d < dimension; d++)
      {
        gradient_forward_interior[d] *= odx;
        gradient_forward_periodic[d] *= odx;

        gradient_backward_interior[d] *= odx;
        gradient_backward_periodic[d] *= odx;
      }
    }

    checkpoint();
    // Compute boundaries and domain 'filter'
    {
      boba::TensorTrainMatrix<dimension, space, double> boundaries_temp[2];
      boundaries_temp[0].rename("boundaries_I");
      boundaries_temp[0].resize(sizes, sizes);
      boundaries_temp[0].fill_with_zeros();
      boundaries_temp[1].rename("boundaries_I_interior");
      boundaries_temp[1].resize(sizes, sizes);
      boundaries_temp[1].fill_with_zeros();
      for (size_t di = 0; di < dimension; di++)
      {
        boundaries_temp[0].cores[di] = identity_1d;
        boundaries_temp[1].cores[di] = identity_1d_interior;
      }
      boundaries_temp[1] *= -1.0;

      boundaries += boundaries_temp[0];
      boundaries += boundaries_temp[1];

      for (size_t di = 0; di < dimension; di++)
      {
        domain.cores[di] = identity_1d_interior;
      }
    }

    checkpoint();

    //
    // Integration
    //
    ttm_integration_domain.resize(sizes_one, sizes);
    ttm_integration_domain.rename("ttm_integration");
    ttm_integration_domain.fill_with_zeros();
    ttm_integration.resize(sizes_one, sizes);
    ttm_integration.rename("ttm_integration_bc");
    ttm_integration.fill_with_zeros();
    tt_integration.resize(sizes);
    tt_integration.rename("tt_integration");
    tt_integration.fill_with_zeros();
    double dx_local = dx;
    for (size_t di = 0; di < dimension; di++)
    {
      auto integration_m_view = ttm_integration_domain.cores[di].view();
      auto integration_m_bc_view = ttm_integration.cores[di].view();
      auto integration_view = tt_integration.cores[di].view();
      checkpoint();
      ::boba::detail::loop<space>(0_z, N, [=] __boba_host_device__(size_t i)
      {
        double integration_value = dx_local;
        double integration_value_bc = dx_local;
        if ((i == 0_z) || (i == N - 1))
        {
          integration_value = 0.0;
          integration_value_bc = 0.5 * dx_local;
        }

        integration_view({0, i, 0}) = integration_value;
        integration_m_view({0, 0, i, 0}) = integration_value;
        integration_m_bc_view({0, 0, i, 0}) = integration_value_bc;
      });
    }

    checkpoint();
  }
};
