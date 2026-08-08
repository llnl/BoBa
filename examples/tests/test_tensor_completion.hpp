#pragma once

#include "BOBA/boba.hpp"

#include <cstddef>
#include <vector>

template<std::size_t dimension>
using coo_index = boba::Array<boba::index_t, dimension>;

constexpr boba::execution_space tensor_completion_space = boba::execution_space::CPU;

template<std::size_t dimension>
struct TensorCOO {
    static constexpr std::size_t order = dimension;

    coo_index<dimension> dims = boba::filled_array<dimension>(boba::index_t(0));
    boba::Matrix<tensor_completion_space, boba::index_t> indices{
      {0, static_cast<boba::index_t>(dimension)}};
    boba::Vector<tensor_completion_space, double> values{{0}};

    [[nodiscard]]
    __boba_host_device__
    constexpr auto get_indices_const_view() const noexcept
    {
      return indices.const_view();
    }
};

struct GaussianHistogramConfig {
    std::vector<int> dims;
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;
    unsigned seed = 12345;
};

struct CPAPRParams {
    int kmax = 200;
    int ellmax = 10;
    double tau = 1e-4;
    double kappa = 1e-2;
    double kappa_tol = 1e-10;
    double epsilon = 1e-10;
    unsigned seed = 12345;
};

template<std::size_t dimension>
using CPAPRModel = boba::CanonicalPolyadicDecomposition<dimension, tensor_completion_space, double>;

constexpr double kEstimateTolerance = 0.5;
