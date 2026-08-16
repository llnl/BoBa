// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_HIPTENSOR
#include <hiptensor/hiptensor.hpp>
#include <hiptensor/hiptensor_types.hpp>
#include <hiptensor/internal/hiptensor_utility.hpp>
#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_HIPTENSOR

extern hiptensorHandle_t hiptensor_handle;

#define hiptensor_assert(a) hiptensor_assert_(a, #a, __LINE__, __FUNCTION__, __FILE__);

/**
 * @brief Reports a hipTensor error and terminates.
 * @param error hipTensor status code to check.
 * @param call Source expression that produced the error.
 * @param line Source line number.
 * @param function Source function name.
 * @param file Source file name.
 */
inline void hiptensor_assert_(
  hiptensorStatus_t error,
  const std::string& call,
  int line,
  const std::string& function,
  const std::string& file)
{
  if (error == HIPTENSOR_STATUS_SUCCESS)
  {
    return;
  }
  std::cout << "Error in " << function << ", " << file << ":" << line
            << std::endl;
  std::cout << call << std::endl;

  std::cout << hiptensorGetErrorString(error) << std::endl;
  exit(1);
}

/**
 * @brief Maps a BoBa scalar type to a HIP runtime data-type tag.
 * @tparam data_t Scalar type.
 * @return Corresponding HIP data-type enum.
 */
template <typename data_t>
[[nodiscard]]
constexpr hipDataType get_hipDataType()
{
  if constexpr (std::is_same<data_t, double>::value)
  {
    return HIP_R_64F;
  }
  if constexpr (std::is_same<data_t, float>::value)
  {
    return HIP_R_32F;
  }
  if constexpr (std::is_same<data_t, complex<double>>::value)
  {
    return HIP_C_64F;
  }
  if constexpr (std::is_same<data_t, complex<float>>::value)
  {
    return HIP_C_32F;
  }

  boba_error("Unknown hipdatatype");
  return static_cast<hipDataType>(0);
}

/**
 * @brief Maps a BoBa scalar type to a hipTensor data-type tag.
 * @tparam data_t Scalar type.
 * @return Corresponding hipTensor data-type enum.
 */
template <typename data_t>
[[nodiscard]]
constexpr hiptensorDataType_t get_hiptensorComputeType()
{
  if constexpr (std::is_same<data_t, double>::value)
  {
    return HIPTENSOR_R_64F;
  }
  if constexpr (std::is_same<data_t, float>::value)
  {
    return HIPTENSOR_R_32F;
  }
  if constexpr (std::is_same<data_t, complex<double>>::value)
  {
    return HIPTENSOR_C_64F;
  }
  if constexpr (std::is_same<data_t, complex<float>>::value)
  {
    return HIPTENSOR_C_32F;
  }

  boba_error("Unknown hiptensorComputeType");
  return static_cast<hiptensorDataType_t>(0);
}

/**
 * @brief Maps a BoBa scalar type to a hipTensor compute descriptor.
 * @tparam data_t Scalar type.
 * @return Corresponding hipTensor compute descriptor enum.
 */
template <typename data_t>
[[nodiscard]]
constexpr hiptensorComputeDescriptor_t get_hiptensorComputeDescriptorType()
{
  if constexpr (std::is_same<data_t, double>::value)
  {
    return HIPTENSOR_COMPUTE_DESC_64F;
  }
  if constexpr (std::is_same<data_t, float>::value)
  {
    return HIPTENSOR_COMPUTE_DESC_32F;
  }
  if constexpr (std::is_same<data_t, complex<double>>::value)
  {
    return HIPTENSOR_COMPUTE_DESC_C64F;
  }
  if constexpr (std::is_same<data_t, complex<float>>::value)
  {
    return HIPTENSOR_COMPUTE_DESC_C32F;
  }

  boba_error("Unknown hiptensorComputeDescriptorType");
  return static_cast<hiptensorComputeDescriptor_t>(0);
}

/**
 * @brief Converts a const subtensor view to an equivalent dense tensor view for hipTensor.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param tensor Subtensor view.
 * @return Dense tensor view over the subtensor storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
TensorView<DefaultAccessor<data_t const>, dimension>
hiptensor_subtensor_view(
  SubtensorView<DefaultAccessor<data_t const>, dimension> tensor)
{
  return {tensor.const_data() + tensor.offset(), tensor.sizes(), tensor.strides()};
}

/**
 * @brief Converts a mutable subtensor view to an equivalent dense tensor view for hipTensor.
 * @tparam data_t Scalar type.
 * @tparam index_t Index type.
 * @tparam dimension Tensor rank.
 * @param tensor Subtensor view.
 * @return Dense tensor view over the subtensor storage.
 */
template <typename data_t, size_t dimension>
[[nodiscard]]
TensorView<DefaultAccessor<data_t>, dimension>
hiptensor_subtensor_view(
  SubtensorView<DefaultAccessor<data_t>, dimension> tensor)
{
  return {tensor.data() + tensor.offset(), tensor.sizes(), tensor.strides()};
}

//
// Permute
//
template <size_t dimension, typename data_t>
void hiptensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  // TODO<external> double precision permute not yet supported
  // boba_always_assert_equal(get_hipDataType<data_t>(), HIP_R_32F, "Only float supported as of rocm/6.3");

  // https://github.com/ROCm/hipTensor/blob/develop/samples/02_permutation/permutation.cpp

  std::vector<int> mode_new, mode_old;
  std::unordered_map<int, int64_t> extent;
  for (size_t d = 0; d < dimension; d++)
  {
    mode_old.push_back(static_cast<int>(d));
    mode_new.push_back(static_cast<int>(permutations[d]));
    extent[d] = static_cast<int64_t>(old_tensor.sizes(d));
  }

  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(old_tensor.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_permutation.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(old_tensor.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_permutation.strides()));

  BOBA_CALI_BEGIN("hiptensorCreateTensorDescriptor");
  //
  // Allocating data
  //
  hiptensorTensorDescriptor_t descA;
  hiptensor_assert(hiptensorCreateTensorDescriptor(
    detail::hiptensor_handle,
    &descA,
    old_tensor.get_dimension(),
    extentA.data(),
    stridesA.data(),
    get_hiptensorComputeType<data_t>(),
    0));

  hiptensorTensorDescriptor_t descC;
  hiptensor_assert(hiptensorCreateTensorDescriptor(
    detail::hiptensor_handle,
    &descC,
    tensor_permutation.get_dimension(),
    extentC.data(),
    stridesC.data(),
    get_hiptensorComputeType<data_t>(),
    0));

  auto A_d = old_tensor.data();
  auto C_d = tensor_permutation.data();
  data_t alpha = PotentiallyComplex<data_t>::value(1.0);

  /*******************************
   * Create Permutation Descriptor
   *******************************/
  BOBA_CALI_SWITCH("hiptensorCreateTensorDescriptor", "hiptensorCreatePermutation");

  hiptensorOperationDescriptor_t desc;
  hiptensor_assert(
    hiptensorCreatePermutation(
      hiptensor_handle,
      &desc,
      descA,
      mode_old.data(),
      HIPTENSOR_OP_IDENTITY,
      descC,
      mode_new.data(),
      get_hiptensorComputeDescriptorType<data_t>()));

  /**************************
   * Set the algorithm to use
   ***************************/
  BOBA_CALI_SWITCH("hiptensorCreatePermutation", "hiptensorCreatePlanPreference");

  const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

  hiptensorPlanPreference_t planPref;
  hiptensor_assert(
    hiptensorCreatePlanPreference(hiptensor_handle, &planPref, algo, HIPTENSOR_JIT_MODE_NONE));

  /**************************
   * Create Plan
   **************************/

  BOBA_CALI_SWITCH("hiptensorCreatePlanPreference", "hiptensorCreatePlan");

  hiptensorPlan_t plan;
  hiptensor_assert(
    hiptensorCreatePlan(hiptensor_handle, &plan, desc, planPref, 0 /* workspaceSizeLimit */));

  /**********************
   * Run
   **********************/

  BOBA_CALI_SWITCH("hiptensorCreatePlan", "hiptensorPermute");

  hiptensor_assert(hiptensorPermute(hiptensor_handle, plan, &alpha, A_d, C_d, nullptr /* stream */))

    BOBA_CALI_END("hiptensorPermute");
}

//
// Common contraction interface
//

template <size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void contract_common(
  TensorView<DefaultAccessor<data_t const>, dimension_A>& tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B>& tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C>& tensor_C,
  std::vector<int>& modeA,
  std::vector<int>& modeB,
  std::vector<int>& modeC)
{
  // See:
  // https://github.com/ROCm/hipTensor/blob/develop/samples/01_contraction/simple_scale_contraction.hpp

  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);
  checkpoint();

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentB = make_std_vector(cast<int64_t>(tensor_B.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesB = make_std_vector(cast<int64_t>(tensor_B.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  BOBA_CALI_BEGIN("hiptensorCreateTensorDescriptor");

  //
  // Initialize tensors with the input lengths
  //
  hiptensorTensorDescriptor_t a_ms_ks;
  hiptensor_assert(
    hiptensorCreateTensorDescriptor(
      detail::hiptensor_handle,
      &a_ms_ks,
      tensor_A.get_dimension(),
      extentA.data(),
      stridesA.data(),
      get_hiptensorComputeType<data_t>(),
      HIPTENSOR_OP_IDENTITY));

  checkpoint();

  hiptensorTensorDescriptor_t b_ns_ks;
  hiptensor_assert(
    hiptensorCreateTensorDescriptor(
      detail::hiptensor_handle,
      &b_ns_ks,
      tensor_B.get_dimension(),
      extentB.data(),
      stridesB.data(),
      get_hiptensorComputeType<data_t>(),
      HIPTENSOR_OP_IDENTITY));

  checkpoint();

  hiptensorTensorDescriptor_t c_ms_ns;
  hiptensor_assert(
    hiptensorCreateTensorDescriptor(
      detail::hiptensor_handle,
      &c_ms_ns,
      tensor_C.get_dimension(),
      extentC.data(),
      stridesC.data(),
      get_hiptensorComputeType<data_t>(),
      HIPTENSOR_OP_IDENTITY));

  //
  // Retrieve the memory alignment for each tensor
  //
  checkpoint();

  BOBA_CALI_SWITCH("hiptensorCreateTensorDescriptor", "hiptensorCreatePlanPreference");

  auto A_d = tensor_A.data();
  auto B_d = tensor_B.data();
  auto C_d = tensor_C.data();

  /*******************************
   * Create Contraction Descriptor
   *******************************/

  hiptensorOperationDescriptor_t desc;
  hiptensor_assert(
    hiptensorCreateContraction(
      detail::hiptensor_handle,
      &desc,
      a_ms_ks,
      modeA.data(),
      HIPTENSOR_OP_IDENTITY,
      b_ns_ks,
      modeB.data(),
      HIPTENSOR_OP_IDENTITY,
      c_ms_ns,
      modeC.data(),
      HIPTENSOR_OP_IDENTITY,
      c_ms_ns,
      modeC.data(),
      get_hiptensorComputeDescriptorType<data_t>()));

  /**************************
   * Set the algorithm to use
   ***************************/
  hiptensorPlanPreference_t planPref;
  hiptensor_assert(
    hiptensorCreatePlanPreference(
      detail::hiptensor_handle,
      &planPref,
      HIPTENSOR_ALGO_ACTOR_CRITIC,
      HIPTENSOR_JIT_MODE_NONE));

  /**********************
   * Query workspace
   **********************/
  BOBA_CALI_SWITCH("hiptensorCreatePlanPreference", "hiptensorEstimateWorkspaceSize");

  uint64_t worksize = 0;
  hiptensor_assert(hiptensorEstimateWorkspaceSize(
    detail::hiptensor_handle,
    desc,
    planPref,
    HIPTENSOR_WORKSPACE_DEFAULT,
    &worksize));

  /**************************
   * Create Contraction Plan
   **************************/
  BOBA_CALI_SWITCH("hiptensorEstimateWorkspaceSize", "hiptensorCreatePlan");

  hiptensorPlan_t plan;
  hiptensor_assert(
    hiptensorCreatePlan(
      detail::hiptensor_handle,
      &plan,
      desc,
      planPref,
      worksize));

  ::boba::Vector<execution_space::HIP, data_t> workspace({static_cast<size_t>(worksize)});

  BOBA_CALI_SWITCH("hiptensorCreatePlan", "hiptensorContract");

  hiptensor_assert(hiptensorContract(
    detail::hiptensor_handle,
    plan,
    static_cast<const void*>(&alpha),
    A_d,
    B_d,
    static_cast<const void*>(&beta),
    C_d,
    C_d,
    workspace.data(),
    worksize,
    nullptr));

  BOBA_CALI_END("hiptensorContract");
}

//
// Contract
//
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void hiptensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  BOBA_CALI_MARK

  // Create vector of modes and extents
  std::vector<int> modeA, modeB, modeC;
  std::unordered_map<int, int64_t> extent;

  int running_index = static_cast<int>(contractions);
  for (size_t i = 0; i < dimension_A; i++)
  {
    bool is_contraction_dimension = false;
    for (size_t c = 0; c < contractions; c++)
    {
      if (i == contraction_dimensions_A[c])
      {
        extent[static_cast<int>(c)] = static_cast<int64_t>(tensor_A.sizes(i));
        modeA.push_back(static_cast<int>(c));
        is_contraction_dimension = true;
        break;
      }
    }

    if (not(is_contraction_dimension))
    {
      extent[running_index] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }
  for (size_t i = 0; i < dimension_B; i++)
  {
    bool is_contraction_dimension = false;
    for (size_t c = 0; c < contractions; c++)
    {
      if (i == contraction_dimensions_B[c])
      {
        boba_always_assert_equal(extent[static_cast<int>(c)], static_cast<int64_t>(tensor_B.sizes(i)), "Unexpected extents");
        modeB.push_back(static_cast<int>(c));
        is_contraction_dimension = true;
        break;
      }
    }

    if (not(is_contraction_dimension))
    {
      extent[running_index] = static_cast<int64_t>(tensor_B.sizes(i));
      modeB.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  contract_common(tensor_A, tensor_B, tensor_C, modeA, modeB, modeC);
}

template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void hiptensor_contract(
  SubtensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  SubtensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  hiptensor_contract(
    hiptensor_subtensor_view(tensor_A),
    hiptensor_subtensor_view(tensor_B),
    tensor_C,
    contraction_dimensions_A,
    contraction_dimensions_B);
}

template <size_t dimension_A, size_t dimension_C, typename data_t>
void reduce_common(
  TensorView<DefaultAccessor<data_t const>, dimension_A>& tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C>& tensor_C,
  std::vector<int>& modeA,
  std::vector<int>& modeC)
{
  BOBA_CALI_MARK

  // Create a vector of extents for each tensor
  std::vector<int64_t> extentA = make_std_vector(cast<int64_t>(tensor_A.sizes()));
  std::vector<int64_t> extentC = make_std_vector(cast<int64_t>(tensor_C.sizes()));

  std::vector<int64_t> stridesA = make_std_vector(cast<int64_t>(tensor_A.strides()));
  std::vector<int64_t> stridesC = make_std_vector(cast<int64_t>(tensor_C.strides()));

  auto A_d = tensor_A.data();
  auto C_d = tensor_C.data();
  data_t alpha = PotentiallyComplex<data_t>::value(1.0);
  data_t beta = PotentiallyComplex<data_t>::value(0.0);

  /**********************
   * Create Tensor Descriptors
   **********************/

  hiptensorTensorDescriptor_t descA = nullptr;
  hiptensor_assert(
    hiptensorCreateTensorDescriptor(
      hiptensor_handle, &descA, tensor_A.get_dimension(), extentA.data(), stridesA.data(), get_hiptensorComputeType<data_t>(), 0));

  hiptensorTensorDescriptor_t descC = nullptr;
  hiptensor_assert(
    hiptensorCreateTensorDescriptor(
      hiptensor_handle, &descC, tensor_C.get_dimension(), extentC.data(), stridesC.data(), get_hiptensorComputeType<data_t>(), 0));

  const hiptensorOperator_t opReduce = HIPTENSOR_OP_ADD;

  /*******************************
   * Create Reduction Descriptor
   *******************************/

  hiptensorOperationDescriptor_t desc;
  hiptensor_assert(
    hiptensorCreateReduction(
      hiptensor_handle, &desc, descA, modeA.data(), HIPTENSOR_OP_IDENTITY, descC, modeC.data(), HIPTENSOR_OP_IDENTITY, descC, modeC.data(), opReduce, get_hiptensorComputeDescriptorType<data_t>()));

  /**************************
   * Set the algorithm to use
   ***************************/

  const hiptensorAlgo_t algo = HIPTENSOR_ALGO_DEFAULT;

  hiptensorPlanPreference_t planPref;
  hiptensor_assert(
    hiptensorCreatePlanPreference(
      hiptensor_handle,
      &planPref,
      algo,
      HIPTENSOR_JIT_MODE_NONE));

  /**********************
   * Query workspace estimate
   **********************/

  uint64_t worksize = 0;
  const hiptensorWorksizePreference_t workspacePref = HIPTENSOR_WORKSPACE_DEFAULT;
  hiptensor_assert(
    hiptensorEstimateWorkspaceSize(
      hiptensor_handle,
      desc,
      planPref,
      workspacePref,
      &worksize));

  ::boba::Vector<execution_space::HIP, data_t> workspace({static_cast<size_t>(worksize)});

  /**************************
   * Create Plan
   **************************/

  hiptensorPlan_t plan;
  hiptensor_assert(
    hiptensorCreatePlan(
      hiptensor_handle,
      &plan,
      desc,
      planPref,
      worksize));

  /**********************
   * Run
   **********************/

  hiptensor_assert(
    hiptensorReduce(
      hiptensor_handle,
      plan,
      (const void*)&alpha,
      A_d,
      (const void*)&beta,
      C_d,
      C_d,
      workspace.data(),
      worksize,
      nullptr));
}

//
// Reduce
//
template <size_t reductions, size_t dimension_A, size_t dimension_C, typename data_t>
void hiptensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  BOBA_CALI_MARK
  // Create vector of modes and extents
  std::vector<int> modeA, modeC;
  std::unordered_map<int, int64_t> extent;

  // Contraction indices are 0, 1, ...
  int running_index = static_cast<int>(reductions);
  for (size_t i = 0; i < dimension_A; i++)
  {
    bool is_contraction_dimension = false;
    for (size_t r = 0; r < reductions; r++)
    {
      if (i == contraction_dimensions[r])
      {
        extent[static_cast<int>(r)] = static_cast<int64_t>(tensor_A.sizes(i));
        modeA.push_back(static_cast<int>(r));
        is_contraction_dimension = true;
        break;
      }
    }

    if (not(is_contraction_dimension))
    {
      extent[running_index] = static_cast<int64_t>(tensor_A.sizes(i));
      modeA.push_back(running_index);
      modeC.push_back(running_index);
      running_index++;
    }
  }

  reduce_common(tensor_A, tensor_C, modeA, modeC);
}

#else

/**
 * @brief Reports that hipTensor permutation support is unavailable.
 */
template <size_t dimension, typename data_t>
void hiptensor_permute(
  TensorView<DefaultAccessor<data_t const>, dimension> old_tensor,
  TensorView<DefaultAccessor<data_t>, dimension> tensor_permutation,
  Array<size_t, dimension> permutations)
{
  ::boba::detail::ignore(old_tensor);
  ::boba::detail::ignore(tensor_permutation);
  ::boba::detail::ignore(permutations);
  boba_error("hiptensor_permute requires a HIP build.");
}

/**
 * @brief Reports that hipTensor contraction support is unavailable.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename data_t>
void hiptensor_contract(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t const>, dimension_B> tensor_B,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions_A);
  ::boba::detail::ignore(contraction_dimensions_B);
  boba_error("hiptensor_contract requires a HIP build.");
}

/**
 * @brief Reports that hipTensor single-axis reduction support is unavailable.
 */
template <size_t reductions, size_t dimension_A, size_t dimension_C, typename data_t>
void hiptensor_reduce(
  TensorView<DefaultAccessor<data_t const>, dimension_A> tensor_A,
  TensorView<DefaultAccessor<data_t>, dimension_C> tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions);
  boba_error("hiptensor_reduce requires a HIP build.");
}

#endif

} // namespace detail
} // namespace boba
