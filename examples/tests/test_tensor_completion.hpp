#pragma once

#include "BOBA/boba.hpp"

#include <cstddef>
#include <vector>

template <std::size_t dimension>
using coo_index = boba::Array<boba::index_t, dimension>;

constexpr boba::execution_space tensor_completion_space = boba::execution_space::CPU;

struct GaussianHistogramConfig
{
  std::vector<int> dims;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  unsigned seed = 12345;
};

struct CPAPRParams
{
  int kmax = 200;
  int ellmax = 10;
  double tau = 1e-4;
  double kappa = 1e-2;
  double kappa_tol = 1e-10;
  double epsilon = 1e-10;
  unsigned seed = 12345;
};

template <std::size_t dimension>
using CPAPRModel = boba::CanonicalPolyadicDecomposition<dimension, tensor_completion_space, double>;

constexpr double kEstimateTolerance = 0.5;
