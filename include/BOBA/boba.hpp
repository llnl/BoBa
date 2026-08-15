// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#if __has_include("BOBA/config.hpp")
#include "BOBA/config.hpp"
#endif

// clang-format off
// Common
#include "BOBA/abstractions/common.hpp"
#include "BOBA/abstractions/types.hpp"
#include "BOBA/abstractions/assert.hpp"
#include "BOBA/abstractions/math.hpp"
#include "BOBA/objects/Complex.hpp"
#include "BOBA/objects/Array.hpp"
#include "BOBA/objects/StaticArray.hpp"
#include "BOBA/objects/Multiindexer.hpp"
#include "BOBA/objects/SimplicialMultiindexer.hpp"
#include "BOBA/objects/PermutationMultiindexer.hpp"
#include "BOBA/abstractions/umpire.hpp"

// Backends
#include "BOBA/backends/boba_cuda.hpp"
#include "BOBA/backends/boba_hip.hpp"
#include "BOBA/backends/boba_sequential.hpp"

// Abstractions
#include "BOBA/abstractions/environment.hpp"
#include "BOBA/abstractions/numerics.hpp"
#include "BOBA/abstractions/debug.hpp"
#include "BOBA/abstractions/profiling.hpp"
#include "BOBA/abstractions/memory.hpp"
#include "BOBA/abstractions/atomics.hpp"
#include "BOBA/abstractions/loops.hpp"
#include "BOBA/abstractions/reductions.hpp"
#include "BOBA/abstractions/tictoc.hpp"

// Dynamic objects
#include "BOBA/tensors/TensorView.hpp"
#include "BOBA/tensors/Tensor.hpp"
#include "BOBA/tensors/SparseTensorView.hpp"
#include "BOBA/tensors/SubtensorView.hpp"
#include "BOBA/backends/boba_mat_files.hpp"
#include "BOBA/backends/boba_hdf5.hpp"
#include "BOBA/tensors/Tensor_io.hpp"
#include "BOBA/tensors/Vector.hpp"
#include "BOBA/tensors/SparseTensor.hpp"
#include "BOBA/abstractions/argparser.hpp"
#include "BOBA/tensors/Matrix.hpp"
#include "BOBA/tensors/PermutationMatrix.hpp"

// Linear algebra
#include "BOBA/backends/boba_cublas.hpp"
#include "BOBA/backends/boba_cusolver.hpp"
#include "BOBA/backends/boba_hipblas.hpp"
#include "BOBA/backends/boba_hipsolver.hpp"

// Static objects
#include "BOBA/StaticTensors/StaticTensorView.hpp"
#include "BOBA/StaticTensors/StaticTensor.hpp"
#include "BOBA/StaticTensors/StaticMatrix.hpp"
#include "BOBA/StaticTensors/StaticVector.hpp"

// Tensor functions
#include "BOBA/backends/boba_eigen.hpp"
#include "BOBA/backends/boba_eigen_tensor.hpp"
#include "BOBA/backends/boba_cutensor.hpp"
#include "BOBA/backends/boba_hiptensor.hpp"
#include "BOBA/backends/boba_metal.hpp"
#include "BOBA/tensors/Matrix_functions.hpp"
#include "BOBA/tensors/Tensor_functions.hpp"

// Fast Fourier Transform
#include "BOBA/backends/boba_eigen_fft.hpp"
#include "BOBA/backends/boba_cufft.hpp"
#include "BOBA/backends/boba_hipfft.hpp"
#include "BOBA/fft/fft.hpp"

// Linear Algebra
#include "BOBA/linear_algebra/LU.hpp"
#include "BOBA/linear_algebra/QR.hpp"
#include "BOBA/linear_algebra/SVD.hpp"
#include "BOBA/linear_algebra/CUR.hpp"
#include "BOBA/linear_algebra/Cholesky.hpp"
#include "BOBA/linear_algebra/nnls.hpp"
#include "BOBA/linear_algebra/NNMF.hpp"

// Tensor Trains
#include "BOBA/TensorTrain/TensorTrain.hpp"
#include "BOBA/QuantizedTensorTrain/QuantizedTensorTrain.hpp"
#include "BOBA/TensorTrain/StaticTensorTrainView.hpp"
#include "BOBA/TensorTrain/DynamicTensorTrainView.hpp"
#include "BOBA/TensorTrain/TensorTrainMatrix.hpp"
#include "BOBA/TensorTrain/TensorTrainSplitMatrix.hpp"
#include "BOBA/TensorTrain/TensorTrain_functions.hpp"
#include "BOBA/TensorTrain/TensorTrainMatrix_functions.hpp"
#include "BOBA/TensorTrain/TensorTrain_utilities.hpp"
#include "BOBA/QuantizedTensorTrain/QuantizedTensorTrainMatrix.hpp"
#include "BOBA/QuantizedTensorTrain/QuantizedTensorTrain_functions.hpp"
#include "BOBA/QuantizedTensorTrain/QuantizedTensorTrainMatrix_functions.hpp"
#include "BOBA/QuantizedTensorTrain/QuantizedTensorTrain_utilities.hpp"
#include "BOBA/TensorTrain/TensorTrainMatrix_utilities.hpp"

// Tucker
#include "BOBA/Tucker/Tucker.hpp"
#include "BOBA/Tucker/Tucker_functions.hpp"
#include "BOBA/Tucker/TuckerMatrix.hpp"
#include "BOBA/Tucker/Tucker_utilities.hpp"
#include "BOBA/Tucker/TuckerMatrix_utilities.hpp"
#include "BOBA/Tucker/StaticTuckerView.hpp"

// HierarchicalTucker
#include "BOBA/HTucker/TreeBuilder.hpp"
#include "BOBA/HTucker/DimensionTree.hpp"
#include "BOBA/HTucker/HTucker.hpp"
#include "BOBA/HTucker/HTucker_functions.hpp"

// CPD (canonial polyadic decomposition)
#include "BOBA/CanonicalPolyadic/CanonicalPolyadicDecomposition.hpp"
#include "BOBA/CanonicalPolyadic/CanonicalPolyadicDecomposition_functions.hpp"
#include "BOBA/CanonicalPolyadic/CanonicalPolyadicDecomposition_utilities.hpp"

// Block structures
#include "BOBA/blocks/BlockVector.hpp"
#include "BOBA/blocks/BlockOperator.hpp"

// Solvers
#include "BOBA/Krylov/Krylov.hpp"
#include "BOBA/TensorTrain/DMRGCross.hpp"
#include "BOBA/TensorTrain/TensorTrainSolver_utilities.hpp"
#include "BOBA/TensorTrain/TensorTrainAMEN.hpp"
#include "BOBA/TensorTrain/TensorTrainAMENBlock.hpp"
#include "BOBA/TensorTrain/TensorTrainDMRG.hpp"
// clang-format on
