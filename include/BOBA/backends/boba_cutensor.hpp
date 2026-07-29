// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_CUTENSOR
#include <cutensor.h>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_CUTENSOR

extern cutensorHandle_t cutensor_handle;

#define cutensor_assert(a) cutensor_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a cuTENSOR error and terminates.
 * @param error cuTENSOR status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void cutensor_assert_(
  cutensorStatus_t error,
  const std::string& call,
  int line,
  const std::string& function,
  const std::string& file)
{
  if (error == CUTENSOR_STATUS_SUCCESS)
  {
    return;
  }

  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;
  std::cout << "Error: " << cutensorGetErrorString(error) << std::endl;
  exit(1);
}

/**
 * @brief Maps a BoBa scalar type to a cuTENSOR data-type tag.
 * @tparam data_t Scalar type.
 * @return Corresponding cuTENSOR data-type enum.
 */
template <typename data_t>
[[nodiscard]]
constexpr cutensorDataType_t get_cuDataType()
{
  if constexpr (std::is_same<data_t, double>::value)
  {
    return CUTENSOR_R_64F;
  }
  if constexpr (std::is_same<data_t, float>::value)
  {
    return CUTENSOR_R_32F;
  }
  if constexpr (std::is_same<data_t, complex<double>>::value)
  {
    return CUTENSOR_C_64F;
  }
  if constexpr (std::is_same<data_t, complex<float>>::value)
  {
    return CUTENSOR_C_32F;
  }

  boba_error("Unkown cudatatype");
  return static_cast<cutensorDataType_t>(0);
}

/**
 * @brief Maps a BoBa scalar type to a cuTENSOR compute descriptor.
 * @tparam data_t Scalar type.
 * @return Corresponding cuTENSOR compute descriptor enum.
 */
template <typename data_t>
[[nodiscard]]
constexpr cutensorComputeDescriptor_t get_cutensorComputeType()
{
  if constexpr (std::is_same<data_t, double>::value)
  {
    return CUTENSOR_COMPUTE_DESC_64F;
  }
  if constexpr (std::is_same<data_t, float>::value)
  {
    return CUTENSOR_COMPUTE_DESC_32F;
  }
  if constexpr (std::is_same<data_t, complex<double>>::value)
  {
    return CUTENSOR_COMPUTE_DESC_64F;
  }
  if constexpr (std::is_same<data_t, complex<float>>::value)
  {
    return CUTENSOR_COMPUTE_DESC_32F;
  }

  boba_error("Unkown cutensorComputeType");
  return static_cast<cutensorComputeDescriptor_t>(0);
}

/**
 * @brief Converts a const subtensor view to an equivalent dense tensor view for cuTENSOR.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param tensor Subtensor view.
 * @return Dense tensor view over the subtensor storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
TensorView<DefaultAccessor<data_t const>, dimension>
cutensor_subtensor_view(
  SubtensorView<DefaultAccessor<data_t const>, dimension> tensor)
{
  return {tensor.const_data() + tensor.offset(), tensor.sizes(), tensor.strides()};
}

/**
 * @brief Converts a mutable subtensor view to an equivalent dense tensor view for cuTENSOR.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param tensor Subtensor view.
 * @return Dense tensor view over the subtensor storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
TensorView<DefaultAccessor<data_t>, dimension>
cutensor_subtensor_view(
  SubtensorView<DefaultAccessor<data_t>, dimension> tensor)
{
  return {tensor.data() + tensor.offset(), tensor.sizes(), tensor.strides()};
}

//
// Permute
//
template <size_t dimension, typename data_t>
void cutensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  BOBA_CALI_MARK
  data_t alpha = PotentiallyComplex<data_t>::value(1.0);

  //
  // C_{c,w,h,n} = alpha * A_{w,h,c,n}
  //
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/elementwise_permute.cu

  std::vector<int> mode_new;
  std::vector<int> mode_old;
  for (size_t d = 0; d < dimension; d++)
  {
    mode_old.push_back(d);
    mode_new.push_back(static_cast<int>(permutations[d]));
  }

  std::unordered_map<int, int64_t> extent;
  for (size_t d = 0; d < dimension; d++)
  {
    extent[d] = static_cast<int>(old_tensor.sizes(d));
  }

  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(old_tensor.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_permutation.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(old_tensor.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_permutation.strides()));

  //
  // Allocating data
  //
  uint64_t const kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(old_tensor.data()), kAlignment, "Workspace must be aligned to some byte-boundary.");

  //
  // Create Tensor Descriptors
  //
  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      old_tensor.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_permutation.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  //
  // Create Permutation Descriptor
  //

  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreatePermutation(
      detail::cutensor_handle,
      &desc,
      descA,
      mode_old.data(),
      CUTENSOR_OP_IDENTITY,
      descC,
      mode_new.data(),
      get_cutensorComputeType<data_t>()));

  //
  // Optional (but recommended): ensure that the scalar type is correct.
  //

  cutensorDataType_t scalarType;
  boba::detail::cutensor_assert(
    cutensorOperationDescriptorGetAttribute(
      detail::cutensor_handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, (void*)&scalarType, sizeof(scalarType)));

  assert(scalarType == get_cuDataType<data_t>());

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  //
  // Create Plan
  //
  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      0 // workspaceSizeLimit
      ));

  //
  // Run
  //
  boba::detail::cutensor_assert(
    cutensorPermute(
      detail::cutensor_handle,
      plan,
      &alpha,
      old_tensor.data(),
      tensor_permutation.data(),
      nullptr // stream
      ));

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyPlanPreference(planPref));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
}

//
// Contract
//
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_A,
  size_t contraction_dimension_B)
{
  BOBA_CALI_MARK
  // See:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/contraction.cu

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  // Create vector of modes and extents
  std::vector<int32_t> modeA, modeB, modeC;
  std::unordered_map<int32_t, int64_t> extent;

  // Contraction index is 0
  int32_t running_index = 1;
  for (size_t i = 0; i < dimension_A; i++)
  {
    if (i == contraction_dimension_A)
    {
      extent[0] = static_cast<int32_t>(tensor_A.sizes(i));
      modeA.push_back(0);
    }
    else
    {
      extent[running_index] = static_cast<int32_t>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }
  for (size_t i = 0; i < dimension_B; i++)
  {
    if (i == contraction_dimension_B)
    {
      boba_always_assert_equal(extent[0], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(0);
    }
    else
    {
      extent[running_index] = static_cast<int32_t>(tensor_B.sizes(i));
      modeB.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentB = make_std_vector(cast<int64_t>(tensor_B.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesB = make_std_vector(cast<int64_t>(tensor_B.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  // Initialize cuTENSOR library
  const uint64_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(tensor_A.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_B.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_C.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descB;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descB,
      tensor_B.get_dimension(),
      extentB.data(),
      stridesB.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  // Create the Contraction Descriptor
  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreateContraction(
      detail::cutensor_handle,
      &desc,
      descA,
      modeA.data(),
      /* unary operator A*/ CUTENSOR_OP_IDENTITY,
      descB,
      modeB.data(),
      /* unary operator B*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      /* unary operator C*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      get_cutensorComputeType<data_t>()));

  cutensorDataType_t scalarType;
  boba::detail::cutensor_assert(
    cutensorOperationDescriptorGetAttribute(
      detail::cutensor_handle,
      desc,
      CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
      (void*)&scalarType,
      sizeof(scalarType)));

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  boba::detail::cutensor_assert(
    cutensorEstimateWorkspaceSize(
      detail::cutensor_handle,
      desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

  //
  // Create Contraction Plan
  //

  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));

  //
  // Optional: Query information about the created plan
  //

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  boba::detail::cutensor_assert(
    cutensorPlanGetAttribute(
      detail::cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  actualWorkspaceSize *= 2;

  // Allocate workspace
  ::boba::Vector<execution_space::CUDA, unsigned char> work({static_cast<size_t>(actualWorkspaceSize)});

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  boba::detail::cutensor_assert(
    cutensorContract(
      detail::cutensor_handle,
      plan,
      (void*)&alpha,
      tensor_A.data(),
      tensor_B.data(),
      (void*)&beta,
      tensor_C.data(),
      tensor_C.data(),
      work.data(),
      actualWorkspaceSize,
      stream));

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descB));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
  boba::detail::cuda_assert(cudaStreamDestroy(stream));
}

template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_A,
  size_t contraction_dimension_B)
{
  cutensor_contract(
    cutensor_subtensor_view(tensor_A),
    cutensor_subtensor_view(tensor_B),
    tensor_C,
    contraction_dimension_A,
    contraction_dimension_B);
}

//
// Contract
//
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B)
{
  BOBA_CALI_BEGIN("setup");

  // See:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/contraction.cu

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  //
  // Computing:
  //

  // Create vector of modes and extents
  std::vector<int32_t> modeA, modeB, modeC;
  std::unordered_map<int32_t, int64_t> extent;

  // Contraction indices are 0, 1
  int32_t running_index = 2;
  for (size_t i = 0; i < dimension_A; i++)
  {
    if (i == contraction_1_dimension_A)
    {
      extent[0] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(0);
    }
    else if (i == contraction_2_dimension_A)
    {
      extent[1] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(1);
    }
    else
    {
      extent[running_index] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }
  for (size_t i = 0; i < dimension_B; i++)
  {
    if (i == contraction_1_dimension_B)
    {
      boba_always_assert_equal(extent[0], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(0);
    }
    else if (i == contraction_2_dimension_B)
    {
      boba_always_assert_equal(extent[1], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(1);
    }
    else
    {
      extent[running_index] = static_cast<int64_t>(tensor_B.sizes(i));
      modeB.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentB = make_std_vector(cast<int64_t>(tensor_B.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesB = make_std_vector(cast<int64_t>(tensor_B.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  // Initialize cuTENSOR library

  const uint64_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(tensor_A.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_B.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_C.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descB;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descB,
      tensor_B.get_dimension(),
      extentB.data(),
      stridesB.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  // Create the Contraction Descriptor
  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreateContraction(
      detail::cutensor_handle,
      &desc,
      descA,
      modeA.data(),
      /* unary operator A*/ CUTENSOR_OP_IDENTITY,
      descB,
      modeB.data(),
      /* unary operator B*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      /* unary operator C*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      get_cutensorComputeType<data_t>()));

  cutensorDataType_t scalarType;
  boba::detail::cutensor_assert(
    cutensorOperationDescriptorGetAttribute(
      detail::cutensor_handle,
      desc,
      CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
      (void*)&scalarType,
      sizeof(scalarType)));

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  BOBA_CALI_SWITCH("setup", "cutensorEstimateWorkspaceSize");

  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  boba::detail::cutensor_assert(
    cutensorEstimateWorkspaceSize(
      detail::cutensor_handle,
      desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

  //
  // Create Contraction Plan
  //
  BOBA_CALI_SWITCH("cutensorEstimateWorkspaceSize", "cutensorCreatePlan");

  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));

  //
  // Optional: Query information about the created plan
  //
  BOBA_CALI_SWITCH("cutensorCreatePlan", "cutensorPlanGetAttribute");

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  boba::detail::cutensor_assert(
    cutensorPlanGetAttribute(
      detail::cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  actualWorkspaceSize *= 2;

  BOBA_CALI_SWITCH("cutensorPlanGetAttribute", "allocate_work");

  // Allocate workspace
  ::boba::Vector<execution_space::CUDA, unsigned char> work({static_cast<size_t>(actualWorkspaceSize)});

  BOBA_CALI_SWITCH("allocate_work", "cudaStreamCreate");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  BOBA_CALI_SWITCH("cudaStreamCreate", "cutensorContract");

  boba::detail::cutensor_assert(
    cutensorContract(
      detail::cutensor_handle,
      plan,
      (void*)&alpha,
      tensor_A.data(),
      tensor_B.data(),
      (void*)&beta,
      tensor_C.data(),
      tensor_C.data(),
      work.data(),
      actualWorkspaceSize,
      stream));

  BOBA_CALI_SWITCH("cutensorContract", "destroy");

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descB));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
  boba::detail::cuda_assert(cudaStreamDestroy(stream));

  BOBA_CALI_END("destroy");
}

template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B)
{
  cutensor_contract(
    cutensor_subtensor_view(tensor_A),
    cutensor_subtensor_view(tensor_B),
    tensor_C,
    contraction_1_dimension_A,
    contraction_2_dimension_A,
    contraction_1_dimension_B,
    contraction_2_dimension_B);
}

//
// Contract
//
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_3_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B,
  size_t contraction_3_dimension_B)
{
  BOBA_CALI_BEGIN("setup");

  // See:
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/contraction.cu

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  //
  // Computing:
  //

  // Create vector of modes and extents
  std::vector<int32_t> modeA, modeB, modeC;
  std::unordered_map<int32_t, int64_t> extent;

  // Contraction indices are 0, 1, 2
  int32_t running_index = 3;
  for (size_t i = 0; i < dimension_A; i++)
  {
    if (i == contraction_1_dimension_A)
    {
      extent[0] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(0);
    }
    else if (i == contraction_2_dimension_A)
    {
      extent[1] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(1);
    }
    else if (i == contraction_3_dimension_A)
    {
      extent[2] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(2);
    }
    else
    {
      extent[running_index] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }
  for (size_t i = 0; i < dimension_B; i++)
  {
    if (i == contraction_1_dimension_B)
    {
      boba_always_assert_equal(extent[0], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(0);
    }
    else if (i == contraction_2_dimension_B)
    {
      boba_always_assert_equal(extent[1], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(1);
    }
    else if (i == contraction_3_dimension_B)
    {
      boba_always_assert_equal(extent[2], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
      modeB.push_back(2);
    }
    else
    {
      extent[running_index] = static_cast<int64_t>(tensor_B.sizes(i));
      modeB.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentB = make_std_vector(cast<int64_t>(tensor_B.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesB = make_std_vector(cast<int64_t>(tensor_B.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  // Initialize cuTENSOR library

  const uint64_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(tensor_A.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_B.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");
  // boba_always_assert_modulo(uintptr_t(tensor_C.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary.");

  // Create Tensor Descriptors
  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descB;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descB,
      tensor_B.get_dimension(),
      extentB.data(),
      stridesB.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  // Create the Contraction Descriptor
  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreateContraction(
      detail::cutensor_handle,
      &desc,
      descA,
      modeA.data(),
      /* unary operator A*/ CUTENSOR_OP_IDENTITY,
      descB,
      modeB.data(),
      /* unary operator B*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      /* unary operator C*/ CUTENSOR_OP_IDENTITY,
      descC,
      modeC.data(),
      get_cutensorComputeType<data_t>()));

  cutensorDataType_t scalarType;
  boba::detail::cutensor_assert(
    cutensorOperationDescriptorGetAttribute(
      detail::cutensor_handle,
      desc,
      CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
      (void*)&scalarType,
      sizeof(scalarType)));

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  BOBA_CALI_SWITCH("setup", "cutensorEstimateWorkspaceSize");

  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  boba::detail::cutensor_assert(
    cutensorEstimateWorkspaceSize(
      detail::cutensor_handle,
      desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

  //
  // Create Contraction Plan
  //
  BOBA_CALI_SWITCH("cutensorEstimateWorkspaceSize", "cutensorCreatePlan");

  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));

  //
  // Optional: Query information about the created plan
  //
  BOBA_CALI_SWITCH("cutensorCreatePlan", "cutensorPlanGetAttribute");

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  boba::detail::cutensor_assert(
    cutensorPlanGetAttribute(
      detail::cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  actualWorkspaceSize *= 2;

  BOBA_CALI_SWITCH("cutensorPlanGetAttribute", "allocate_work");

  // Allocate workspace
  ::boba::Vector<execution_space::CUDA, unsigned char> work({static_cast<size_t>(actualWorkspaceSize)});

  BOBA_CALI_SWITCH("allocate_work", "cudaStreamCreate");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  BOBA_CALI_SWITCH("cudaStreamCreate", "cutensorContract");

  boba::detail::cutensor_assert(
    cutensorContract(
      detail::cutensor_handle,
      plan,
      (void*)&alpha,
      tensor_A.data(),
      tensor_B.data(),
      (void*)&beta,
      tensor_C.data(),
      tensor_C.data(),
      work.data(),
      actualWorkspaceSize,
      stream));

  BOBA_CALI_SWITCH("cutensorContract", "destroy");

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descB));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
  boba::detail::cuda_assert(cudaStreamDestroy(stream));

  BOBA_CALI_END("destroy");
}

template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_3_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B,
  size_t contraction_3_dimension_B)
{
  cutensor_contract(
    cutensor_subtensor_view(tensor_A),
    cutensor_subtensor_view(tensor_B),
    tensor_C,
    contraction_1_dimension_A,
    contraction_2_dimension_A,
    contraction_3_dimension_A,
    contraction_1_dimension_B,
    contraction_2_dimension_B,
    contraction_3_dimension_B);
}

//
// Reduce
//
template <size_t dimension_A, size_t dimension_C, typename data_t>
void cutensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_1,
  size_t contraction_dimension_2)
{
  BOBA_CALI_MARK
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/reduction.cu

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  //
  // Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
  //

  // Create vector of modes and extents
  std::vector<int> modeA, modeC;
  std::unordered_map<int, int64_t> extent;

  // Indices to be contracted are marked by 0 and 1
  int running_index = 2;
  for (size_t i = 0; i < dimension_A; i++)
  {
    if (i == contraction_dimension_1)
    {
      extent[0] = static_cast<int>(tensor_A.sizes(i));
      modeA.push_back(0);
    }
    else if (i == contraction_dimension_2)
    {
      extent[1] = static_cast<int>(tensor_A.sizes(i));
      modeA.push_back(1);
    }
    else
    {
      extent[running_index] = static_cast<int>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  //
  // Create Tensor Descriptors
  //
  const uint64_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(tensor_A.data()) , kAlignment, "Workspace must be aligned to some byte-boundary?");
  // boba_always_assert_modulo(uintptr_t(tensor_C.data()) , kAlignment, "Workspace must be aligned to some byte-boundary?");

  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  //
  // Create Reduction Descriptor
  //
  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreateReduction(
      detail::cutensor_handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(), CUTENSOR_OP_ADD, get_cutensorComputeType<data_t>()));

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  //
  // Query workspace estimate
  //
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  boba::detail::cutensor_assert(
    cutensorEstimateWorkspaceSize(
      detail::cutensor_handle,
      desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

  //
  // Create Contraction Plan
  //
  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));

  //
  // Optional: Query information about the created plan
  //

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  boba::detail::cutensor_assert(
    cutensorPlanGetAttribute(
      detail::cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  // At this point the user knows exactly how much memory is need by the operation and
  // only the smaller actual workspace needs to be allocated
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  // Allocate workspace
  ::boba::Vector<execution_space::CUDA, unsigned char> work({static_cast<size_t>(actualWorkspaceSize)});

  if (actualWorkspaceSize > 0)
  {
    // boba_always_assert_modulo(uintptr_t(work.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary?");
  }

  //
  // Run
  //
  cudaStream_t stream;
  boba::detail::cuda_assert(cudaStreamCreate(&stream));

  boba::detail::cutensor_assert(
    cutensorReduce(
      detail::cutensor_handle, plan, (const void*)&alpha, tensor_A.const_data(), (const void*)&beta, tensor_C.data(), tensor_C.data(), work.data(), actualWorkspaceSize, stream));

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
  boba::detail::cuda_assert(cudaStreamDestroy(stream));
}

//
// Reduce
//
template <size_t dimension_A, size_t dimension_C, typename data_t>
void cutensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_1)
{
  BOBA_CALI_MARK

  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/reduction.cu

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  //
  // Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
  //

  // Create vector of modes and extents
  std::vector<int> modeA, modeC;
  std::unordered_map<int, int64_t> extent;

  // Indices to be contracted are marked by 0
  int running_index = 1;
  for (size_t i = 0; i < dimension_A; i++)
  {
    if (i == contraction_dimension_1)
    {
      extent[0] = static_cast<int>(tensor_A.sizes(i));
      modeA.push_back(0);
    }
    else
    {
      extent[running_index] = static_cast<int>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  //
  // Create Tensor Descriptors
  //
  const uint64_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
  // boba_always_assert_modulo(uintptr_t(tensor_A.data()) , kAlignment, "Workspace must be aligned to some byte-boundary?");
  // boba_always_assert_modulo(uintptr_t(tensor_C.data()) , kAlignment, "Workspace must be aligned to some byte-boundary?");

  cutensorTensorDescriptor_t descA;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descA,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  cutensorTensorDescriptor_t descC;
  boba::detail::cutensor_assert(
    cutensorCreateTensorDescriptor(
      detail::cutensor_handle,
      &descC,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_cuDataType<data_t>(),
      kAlignment));

  //
  // Create Reduction Descriptor
  //
  cutensorOperationDescriptor_t desc;
  boba::detail::cutensor_assert(
    cutensorCreateReduction(
      detail::cutensor_handle, &desc, descA, modeA.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(), CUTENSOR_OP_IDENTITY, descC, modeC.data(), CUTENSOR_OP_ADD, get_cutensorComputeType<data_t>()));

  //
  // Set the algorithm to use
  //
  const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

  cutensorPlanPreference_t planPref;
  boba::detail::cutensor_assert(
    cutensorCreatePlanPreference(
      detail::cutensor_handle,
      &planPref,
      algo,
      CUTENSOR_JIT_MODE_NONE));

  //
  // Query workspace estimate
  //
  uint64_t workspaceSizeEstimate = 0;
  const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
  boba::detail::cutensor_assert(
    cutensorEstimateWorkspaceSize(
      detail::cutensor_handle,
      desc,
      planPref,
      workspacePref,
      &workspaceSizeEstimate));

  //
  // Create Contraction Plan
  //
  cutensorPlan_t plan;
  boba::detail::cutensor_assert(
    cutensorCreatePlan(
      detail::cutensor_handle,
      &plan,
      desc,
      planPref,
      workspaceSizeEstimate));

  //
  // Optional: Query information about the created plan
  //

  // query actually used workspace
  uint64_t actualWorkspaceSize = 0;
  boba::detail::cutensor_assert(
    cutensorPlanGetAttribute(
      detail::cutensor_handle,
      plan,
      CUTENSOR_PLAN_REQUIRED_WORKSPACE,
      &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));

  // At this point the user knows exactly how much memory is need by the operation and
  // only the smaller actual workspace needs to be allocated
  assert(actualWorkspaceSize <= workspaceSizeEstimate);

  // Allocate workspace
  ::boba::Vector<execution_space::CUDA, unsigned char> work({static_cast<size_t>(actualWorkspaceSize)});

  if (actualWorkspaceSize > 0)
  {
    // boba_always_assert_modulo(uintptr_t(work.data()), kAlignment, "Workspace must be aligned to 128 byte-boundary?");
  }

  //
  // Run
  //
  cudaStream_t stream;
  boba::detail::cuda_assert(cudaStreamCreate(&stream));

  boba::detail::cutensor_assert(
    cutensorReduce(
      detail::cutensor_handle, plan, (const void*)&alpha, tensor_A.const_data(), (const void*)&beta, tensor_C.data(), tensor_C.data(), work.data(), actualWorkspaceSize, stream));

  boba::detail::cutensor_assert(cutensorDestroyPlan(plan));
  boba::detail::cutensor_assert(cutensorDestroyOperationDescriptor(desc));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descA));
  boba::detail::cutensor_assert(cutensorDestroyTensorDescriptor(descC));
  boba::detail::cuda_assert(cudaStreamDestroy(stream));
}

#else

/**
 * @brief Reports that cuTENSOR permutation support is unavailable.
 */
template <size_t dimension, typename data_t>
void cutensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  ::boba::detail::ignore(old_tensor);
  ::boba::detail::ignore(tensor_permutation);
  ::boba::detail::ignore(permutations);
  boba_error("cutensor_permute requires a CUDA build.");
}

/**
 * @brief Reports that cuTENSOR single-index contraction support is unavailable.
 */
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_A,
  size_t contraction_dimension_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimension_A);
  ::boba::detail::ignore(contraction_dimension_B);
  boba_error("cutensor_contract requires a CUDA build.");
}

/**
 * @brief Reports that cuTENSOR double-index contraction support is unavailable.
 */
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_1_dimension_A);
  ::boba::detail::ignore(contraction_2_dimension_A);
  ::boba::detail::ignore(contraction_1_dimension_B);
  ::boba::detail::ignore(contraction_2_dimension_B);
  boba_error("cutensor_contract requires a CUDA build.");
}

/**
 * @brief Reports that cuTENSOR triple-index contraction support is unavailable.
 */
template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void cutensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_1_dimension_A,
  size_t contraction_2_dimension_A,
  size_t contraction_3_dimension_A,
  size_t contraction_1_dimension_B,
  size_t contraction_2_dimension_B,
  size_t contraction_3_dimension_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_1_dimension_A);
  ::boba::detail::ignore(contraction_2_dimension_A);
  ::boba::detail::ignore(contraction_3_dimension_A);
  ::boba::detail::ignore(contraction_1_dimension_B);
  ::boba::detail::ignore(contraction_2_dimension_B);
  ::boba::detail::ignore(contraction_3_dimension_B);
  boba_error("cutensor_contract requires a CUDA build.");
}

/**
 * @brief Reports that cuTENSOR single-axis reduction support is unavailable.
 */
template <size_t dimension_A, size_t dimension_C, typename data_t>
void cutensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_1)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimension_1);
  boba_error("cutensor_reduce requires a CUDA build.");
}

/**
 * @brief Reports that cuTENSOR double-axis reduction support is unavailable.
 */
template <size_t dimension_A, size_t dimension_C, typename data_t>
void cutensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  size_t contraction_dimension_1,
  size_t contraction_dimension_2)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimension_1);
  ::boba::detail::ignore(contraction_dimension_2);
  boba_error("cutensor_reduce requires a CUDA build.");
}

#endif

} // namespace detail
} // namespace boba
