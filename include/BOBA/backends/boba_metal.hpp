// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#ifdef BOBA_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

// #include "metal_wrap/wrapper.hpp"
#include <Metal/Metal.hpp>
#include <functional>
#include <iostream> // TODO
#include <memory>

namespace metal
{

extern NS::Error* _error;
extern MTL::Device* _device;
extern MTL::CommandQueue* _queue;
extern MTL::CommandBuffer* _command_buffer;

/**
 * @brief Returns the shared Metal device.
 * @return Metal device handle.
 */
MTL::Device* device();
/**
 * @brief Returns the shared Metal command queue.
 * @return Metal command queue handle.
 */
MTL::CommandQueue* queue();
/**
 * @brief Returns the active Metal command buffer.
 * @return Metal command buffer handle.
 */
MTL::CommandBuffer* command_buffer();

/**
 * @brief Synchronizes the current Metal device queue.
 */
void device_synchronize();

struct dim3
{
  uint32_t x;
  uint32_t y;
  uint32_t z;

  /**
   * @brief Constructs a three-dimensional launch shape.
   * @param nx Size in the x dimension.
   * @param ny Size in the y dimension.
   * @param nz Size in the z dimension.
   */
  dim3(uint32_t nx, uint32_t ny = 1, uint32_t nz = 1)
      : x(nx),
        y(ny),
        z(ny)
  {
  }
};

template <typename T>
struct array
{
  /**
   * @brief Constructs a shared-memory Metal buffer wrapper.
   * @param size Number of elements to allocate.
   */
  array(size_t size)
  {
    sz = size;
    buffer = device()->newBuffer(size * sizeof(T), MTL::ResourceStorageModeShared);
    ptr = (T*)buffer->contents();
  }

  /**
   * @brief destructor
   */
  ~array()
  {
    buffer->release();
    sz = 0;
    ptr = nullptr;
  }

  /**
   * @brief Accesses a mutable buffer element.
   * @param i Element index.
   * @return Reference to the selected element.
   */
  T& operator[](uint32_t i)
  {
    return ptr[i];
  }
  /**
   * @brief Accesses a const buffer element.
   * @param i Element index.
   * @return Reference to the selected element.
   */
  const T& operator[](uint32_t i) const
  {
    return ptr[i];
  }

  T* ptr;
  MTL::Buffer* buffer;
  size_t sz;
};

template <typename T>
struct KernelArgumentType
{
  static constexpr bool is_valid = false;
};

#define REGISTER_VALID_TYPE(T)             \
  template <>                              \
  struct KernelArgumentType<T>             \
  {                                        \
    static constexpr bool is_valid = true; \
  };

REGISTER_VALID_TYPE(int32_t);
REGISTER_VALID_TYPE(uint32_t);
REGISTER_VALID_TYPE(float);

REGISTER_VALID_TYPE(array<int>&);
REGISTER_VALID_TYPE(array<uint32_t>&);
REGISTER_VALID_TYPE(array<float>&);

REGISTER_VALID_TYPE(const array<int32_t>&);
REGISTER_VALID_TYPE(const array<uint32_t>&);
REGISTER_VALID_TYPE(const array<float>&);

#undef VALID_TYPE

template <typename T>
struct FunctionSignature
{
  static constexpr bool is_valid_kernel_signature = false;
};

template <typename return_type, typename... arg_types>
struct FunctionSignature<return_type(arg_types...)>
{
  static constexpr bool is_valid_kernel_signature =
    std::is_same<return_type, void>::value && (KernelArgumentType<arg_types>::is_valid && ...);
};

namespace impl
{

template <typename T>
struct is_scalar_type : std::false_type
{
};
template <>
struct is_scalar_type<int32_t> : std::true_type
{
};
template <>
struct is_scalar_type<uint32_t> : std::true_type
{
};
template <>
struct is_scalar_type<float> : std::true_type
{
};

template <typename T>
struct is_array_type : std::false_type
{
};
template <typename T>
struct is_array_type<metal::array<T>&> : std::true_type
{
};
template <typename T>
struct is_array_type<const metal::array<T>&> : std::true_type
{
};

/**
 * @brief Encodes a scalar or array argument into a Metal compute encoder.
 * @tparam arg_type Encoded argument type.
 * @param encoder Compute encoder receiving the argument.
 * @param arg Argument value.
 * @param arg_index Shader argument slot.
 */
template <typename arg_type>
void encode_argument(MTL::ComputeCommandEncoder* encoder, arg_type arg, uint32_t arg_index)
{
  if constexpr (is_scalar_type<arg_type>::value)
  {
    encoder->setBytes(&arg, sizeof(arg_type), arg_index);
    // std::cout << "encoding uniform at index " << arg_index << std::endl;
    // std::cout << "arg value: " << arg << std::endl;
  }

  if constexpr (is_array_type<arg_type>::value)
  {
    encoder->setBuffer(arg.buffer, 0, arg_index);
    // std::cout << "encoding buffer at index " << arg_index << std::endl;
    // std::cout << "buffer values: " << arg[0] << " " << arg[1] << " " << arg[2] << std::endl;
  }
}

/**
 * @brief Builds a callable Metal kernel launcher from shader source.
 * @tparam arg_types Kernel argument types.
 * @param shader_src Metal shader source.
 * @param kernel_name Entry-point function name.
 * @return Callable launcher that dispatches the compiled kernel.
 */
template <typename... arg_types>
auto create_kernel(FunctionSignature<void(arg_types...)>, std::string shader_src, std::string kernel_name)
{

  NS::Error* error = nullptr;

  MTL::Library* compute_library = device()->newLibrary(NS::String::string(shader_src.c_str(), NS::UTF8StringEncoding), nullptr, &error);
  if (!compute_library)
  {
    printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }

  MTL::Function* kernel_fn = compute_library->newFunction(NS::String::string(kernel_name.c_str(), NS::UTF8StringEncoding));

  MTL::ComputePipelineState* compute_pso = device()->newComputePipelineState(kernel_fn, &error);
  if (!compute_pso)
  {
    printf("%s", error->localizedDescription()->utf8String());
    assert(false);
  }

  auto deleter = [](auto* ptr)
  {
    ptr->release();
  };

  return std::function<void(dim3, dim3, arg_types...)>([_compute_library = std::shared_ptr<MTL::Library>(compute_library, deleter),
                                                        _kernel_fn = std::shared_ptr<MTL::Function>(kernel_fn, deleter),
                                                        _compute_pso = std::shared_ptr<MTL::ComputePipelineState>(compute_pso, deleter)](dim3 grid, dim3 threadgroup, arg_types... args)
  {
    // Start a compute pass.
    MTL::ComputeCommandEncoder* encoder = command_buffer()->computeCommandEncoder();
    encoder->setComputePipelineState(_compute_pso.get());

    // Encode the compute shader arguments
    uint32_t index = 0;
    (encode_argument<arg_types>(encoder, args, index++), ...);

    // Encode the compute command.
    encoder->dispatchThreads(
      MTL::Size(grid.x, grid.y, grid.z),
      MTL::Size(threadgroup.x, threadgroup.y, threadgroup.z));

    // End the compute pass.
    encoder->endEncoding();

    // Execute the command.
    command_buffer()->commit();
  });
};

} // namespace impl

/**
 * @brief Builds a typed Metal kernel launcher.
 * @tparam T Kernel signature type.
 * @param shader_src Metal shader source.
 * @param kernel_name Entry-point function name.
 * @return Callable launcher that dispatches the compiled kernel.
 */
template <typename T>
auto create_kernel(std::string shader_src, std::string kernel_name)
{
  using signature = FunctionSignature<T>;
  static_assert(signature::is_valid_kernel_signature, "invalid kernel signature");
  return impl::create_kernel(signature{}, shader_src, kernel_name);
};
} // namespace metal

#endif

// -----------------------------------------------------
// Definitions
// -----------------------------------------------------
namespace boba
{

namespace detail
{

#ifdef BOBA_METAL

/**
 * @brief Copies tensor data into a temporary Metal array.
 * @tparam tensor_like_t Tensor-like view type.
 * @param tensor_view Source tensor view.
 * @return Metal array containing the copied tensor data.
 */
template <typename tensor_like_t>
[[nodiscard]]
metal::array<float> make_metal_array_from_tensor(tensor_like_t& tensor_view)
{
  auto size = tensor_view.size();
  metal::array<float> metal_array(static_cast<size_t>(size));

  for (int i = 0; i < size; i++)
  {
    metal_array[i] = tensor_view(i);
  }
  return metal_array;
}

/**
 * @brief Copies data from a Metal array back into a tensor view.
 * @tparam tensor_like_t Tensor-like view type.
 * @param tensor_view Destination tensor view.
 * @param metal_array Source Metal array.
 */
template <typename tensor_like_t>
void write_to_tensor_from_metal(tensor_like_t& tensor_view, metal::array<float>& metal_array)
{
  auto size = tensor_view.size();

  for (int i = 0; i < size; i++)
  {
    tensor_view(i) = metal_array[i];
  }
}

/**
 * @brief Copies a fixed-size BoBa array into a Metal array.
 * @tparam dimension Array length.
 * @param array Source array.
 * @return Metal array containing the copied values.
 */
template <size_t dimension>
[[nodiscard]]
metal::array<float> make_metal_array_from_array(Array<size_t, dimension>& array)
{
  metal::array<float> metal_array(dimension);
  for (int i = 0; i < dimension; i++)
  {
    metal_array[i] = array[i];
  }
  return metal_array;
}

/**
 * @brief Formats an array initialization block for generated Metal source.
 * @tparam dimension Array length.
 * @param name Variable name in generated source.
 * @param array Source values.
 * @param indent Number of leading spaces to insert on assignment lines.
 * @return Generated Metal source snippet.
 */
template <size_t dimension>
[[nodiscard]]
std::string make_metal_string(std::string name, const Array<size_t, dimension>& array, size_t indent = 0)
{
  // Make string from array
  std::string array_str = "uint " + name + "[" + std::to_string(dimension) + "];\n";
  for (int i = 0; i < dimension; i++)
  {
    array_str += std::string(indent, ' ') + name + "[" + std::to_string(i) + "] = " + std::to_string(array[i]) + ";\n";
  }
  return array_str;
}

/**
 * @brief Replaces all occurrences of a substring in a Metal source string.
 * @param raw_string String to modify in place.
 * @param old_text Substring to replace.
 * @param new_text Replacement substring.
 */
void replace_all_string(std::string& raw_string, std::string old_text, std::string new_text)
{
  // Replace in string
  std::size_t position = raw_string.find(old_text);
  while (position < std::string::npos)
  {
    raw_string.replace(position, old_text.length(), new_text);
    // Get next position
    position = raw_string.find(old_text);
  }
}

//
// Permute
//
template <typename accessor, size_t dimension>
void metal_permute(
  TensorView<accessor, dimension>& old_tensor,
  TensorView<accessor, dimension>& tensor_permutation,
  Array<size_t, dimension>& permutations)
{
  //
  // Create algorithm
  //
  std::string shader = R"(
    #include <metal_stdlib>

    using namespace metal;

    kernel void permute(
                      device const float * old_tensor,
                      device float * tensor_permutation,
                      uint i [[thread_position_in_grid]]) {

        // Use regex to replace these with tensor sizes
        x_old_tensor_sizes_x
        x_old_tensor_strides_x
        x_tensor_permutation_strides_x
        x_permutations_x

        // auto multiindex_old = old_view.multiindex(i);
        uint multiindex_old[dimension];
        {
          uint index = i;
          for (uint d = 0; d < dimension - 1; d++) {
            multiindex_old[d] = index % old_tensor_sizes[d];
            index /= old_tensor_sizes[d];
          }
          multiindex_old[ dimension - 1] = index;
        }

        // auto multiindex_new = permute(multiindex_old, local_permutations);
        uint multiindex_new[dimension];
        for(int i = 0; i < dimension; i++)
        {
          multiindex_new[i] = multiindex_old[permutations[i]];
        }
 
        // auto value = old_view(multiindex_old);
        uint id_old = 0;
        id_old += multiindex_old[0];
        for(uint d = 1; d < dimension; ++d) {
          id_old += old_tensor_strides[d]*multiindex_old[d];
        }
        float value = old_tensor[id_old];

        // new_view(multiindex_new) = value;
        uint id_new = 0;
        id_new += multiindex_new[0];
        for(uint d = 1; d < dimension; ++d) {
          id_new += tensor_permutation_strides[d]*multiindex_new[d];
        }
        tensor_permutation[id_new] = value;
    })";

  replace_all_string(shader, "x_old_tensor_sizes_x", make_metal_string("old_tensor_sizes", old_tensor.sizes(), 8));
  replace_all_string(shader, "x_old_tensor_strides_x", make_metal_string("old_tensor_strides", old_tensor.strides(), 8));
  replace_all_string(shader, "x_tensor_permutation_strides_x", make_metal_string("tensor_permutation_strides", tensor_permutation.strides(), 8));
  replace_all_string(shader, "x_permutations_x", make_metal_string("permutations", permutations, 8));
  replace_all_string(shader, "dimension", std::to_string(dimension));

  //
  // Initialize pipeline
  //
  MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
  NS::Error* pError = nullptr;

  MTL::Library* pComputeLibrary = pDevice->newLibrary(NS::String::string(shader.c_str(), NS::UTF8StringEncoding), nullptr, &pError);
  if (!pComputeLibrary)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  MTL::Function* mtl_fcn = pComputeLibrary->newFunction(NS::String::string("permute", NS::UTF8StringEncoding));

  MTL::ComputePipelineState* mtl_pipe = pDevice->newComputePipelineState(mtl_fcn, &pError);
  if (!mtl_pipe)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  //
  // Data setup
  //
  MTL::Buffer* old_tensor_buffer = pDevice->newBuffer(old_tensor.size() * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer* tensor_permutation_buffer = pDevice->newBuffer(tensor_permutation.size() * sizeof(float), MTL::ResourceStorageModeShared);
  float* old_tensor_buffer_ptr = (float*)old_tensor_buffer->contents();

  for (int i = 0; i < old_tensor.size(); i++)
  {
    old_tensor_buffer_ptr[i] = static_cast<float>(old_tensor(i));
  }

  auto kernel_size = old_tensor.size();
  MTL::Size grid(kernel_size, 1, 1);
  MTL::Size threadgroup(std::min(128_z, kernel_size), 1, 1);

  //
  // Launch compute
  //
  MTL::CommandQueue* pQueue = pDevice->newCommandQueue();

  MTL::CommandBuffer* pCommandBuffer = pQueue->commandBuffer();
  if (!pCommandBuffer)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  MTL::ComputeCommandEncoder* pComputeEncoder = pCommandBuffer->computeCommandEncoder();
  if (!pComputeEncoder)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  //
  // Set buffer
  //
  pComputeEncoder->setComputePipelineState(mtl_pipe);
  pComputeEncoder->setBuffer(old_tensor_buffer, 0, 0);
  pComputeEncoder->setBuffer(tensor_permutation_buffer, 0, 1);
  pComputeEncoder->dispatchThreads(grid, threadgroup);
  pComputeEncoder->endEncoding();
  pCommandBuffer->commit();

  //
  // Device snync
  //
  pCommandBuffer->waitUntilCompleted();

  //
  // Free memory
  //
  pDevice->release();
  mtl_fcn->release();
  mtl_pipe->release();
  pComputeLibrary->release();

  //
  // Copy back to tensors
  //
  float* tensor_permutation_buffer_ptr = (float*)tensor_permutation_buffer->contents();

  for (int i = 0; i < old_tensor.size(); i++)
  {
    tensor_permutation(i) = static_cast<double>(tensor_permutation_buffer_ptr[i]);
  }
}

//
// Contract
//
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename accessor, typename const_accessor>
void metal_contract(
  const TensorView<const_accessor, dimension_A>& tensor_A,
  const TensorView<const_accessor, dimension_B>& tensor_B,
  TensorView<accessor, dimension_C>& tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  //
  // Create algorithm
  //
  std::string shader = R"(
    #include <metal_stdlib>

    using namespace metal;

    kernel void contract(
                      device const float * tensor_A,
                      device const float * tensor_B,
                      device atomic<float> * tensor_C,
                      uint compute_id [[thread_position_in_grid]]) {

        x_sizes_C_x
        x_strides_A_x
        x_strides_B_x
        x_strides_C_x
        x_contraction_dimensions_A_x
        x_contraction_dimensions_B_x
        x_contraction_lengths_x

        //
        // Get C_multiindex and contraction ids
        //
        uint C_multiindex[dimension_C];
        uint contraction_ids[contractions + 1];
        {
          uint index = compute_id;
          for (uint d = 0; d < dimension_C - 1; d++) {
            C_multiindex[d] = index % sizes_C[d];
            index /= sizes_C[d];
          }
          C_multiindex[dimension_C - 1] = index % sizes_C[dimension_C - 1];
          index /= sizes_C[dimension_C - 1];
          for (uint c = 0; c < contractions; c++) {
            contraction_ids[c] = index % contraction_lengths[c];
            index /= contraction_lengths[c];
          }
        }

        //
        // Get A_pre_multiindex and B_pre_multiindex
        //
        // add 1 to length of array to get around "zero-length arrays are not permitted in C++"
        uint A_pre_multiindex[dimension_A - contractions + 1];
        uint B_pre_multiindex[dimension_B - contractions + 1];
        for (uint d = 0; d < dimension_A - contractions; d++)
        {
          A_pre_multiindex[d] = C_multiindex[d];
        }
        for (uint d = 0; d < dimension_B - contractions; d++)
        {
          B_pre_multiindex[d] = C_multiindex[(dimension_A - contractions) + d];
        }

        //
        // Get A_multiindex
        //
        float value_A = 0.0;
        {
          uint A_multiindex[dimension_A];
          for (uint d = 0; d < dimension_A; d++)
          {
            bool is_contraction_dimension = false;
            uint dprime = d;
            for (uint c = 0; c < contractions; c++)
            {
              if(d == contraction_dimensions_A[c])
              {
                A_multiindex[d] = contraction_ids[c];
                is_contraction_dimension = true;
              }
              else if(d > contraction_dimensions_A[c])
              {
                dprime--;
              }
            }
            if(!is_contraction_dimension)
            {
              A_multiindex[d] = A_pre_multiindex[dprime];
            }
          }

          // Get A value
          uint id_A = 0;
          for(uint d = 0; d < dimension_A; ++d)
          {
            id_A += strides_A[d]*A_multiindex[d];
          }
          value_A = tensor_A[id_A];
        }

        //
        // Get B_multiindex
        //
        float value_B = 0.0;
        {
          uint B_multiindex[dimension_B];
          for (uint d = 0; d < dimension_B; d++)
          {
            bool is_contraction_dimension = false;
            uint dprime = d;
            for (uint c = 0; c < contractions; c++)
            {
              if(d == contraction_dimensions_B[c])
              {
                B_multiindex[d] = contraction_ids[c];
                is_contraction_dimension = true;
              }
              else if(d > contraction_dimensions_B[c])
              {
                dprime--;
              }
            }
            if(!is_contraction_dimension)
            {
              B_multiindex[d] = B_pre_multiindex[dprime];
            }
          }

          // Get B value
          uint id_B = 0;
          for(uint d = 0; d < dimension_B; ++d)
          {
            id_B += strides_B[d]*B_multiindex[d];
          }
          value_B = tensor_B[id_B];
        }

        float value_C = value_A * value_B;

        uint id_C = 0;
        for(uint d = 0; d < dimension_C; ++d)
        {
          id_C += strides_C[d]*C_multiindex[d];
        }
        atomic_fetch_add_explicit(tensor_C + id_C, value_C, memory_order_relaxed);
    })";

  //
  // Set contraction metadata
  //
  boba::Array<size_t, contractions> contraction_lengths;
  size_t contraction_size = 1;
  for (size_t c = 0; c < contractions; c++)
  {
    contraction_lengths[c] = tensor_A.sizes(contraction_dimensions_A[c]);
    contraction_size *= contraction_lengths[c];
  }

  replace_all_string(shader, "x_sizes_C_x", make_metal_string("sizes_C", tensor_C.sizes(), 8));
  replace_all_string(shader, "x_strides_A_x", make_metal_string("strides_A", tensor_A.strides(), 8));
  replace_all_string(shader, "x_strides_B_x", make_metal_string("strides_B", tensor_B.strides(), 8));
  replace_all_string(shader, "x_strides_C_x", make_metal_string("strides_C", tensor_C.strides(), 8));
  replace_all_string(shader, "x_contraction_dimensions_A_x", make_metal_string("contraction_dimensions_A", contraction_dimensions_A, 8));
  replace_all_string(shader, "x_contraction_dimensions_B_x", make_metal_string("contraction_dimensions_B", contraction_dimensions_B, 8));
  replace_all_string(shader, "x_contraction_lengths_x", make_metal_string("contraction_lengths", contraction_lengths, 8));

  replace_all_string(shader, "dimension_A", std::to_string(dimension_A));
  replace_all_string(shader, "dimension_B", std::to_string(dimension_B));
  replace_all_string(shader, "dimension_C", std::to_string(dimension_C));
  replace_all_string(shader, "contractions", std::to_string(contractions));

  //
  // Initialize pipeline
  //

  // TODO<optimization> Expensive to do!  Can we do this once per run?
  // Idea, make a std::map of  parameters -> mtl_fcn
  // We can check if this has been compiled before and call the pre-compiled version

  MTL::Device* pDevice = MTL::CreateSystemDefaultDevice();
  NS::Error* pError = nullptr;

  MTL::Library* pComputeLibrary = pDevice->newLibrary(NS::String::string(shader.c_str(), NS::UTF8StringEncoding), nullptr, &pError);
  if (!pComputeLibrary)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  MTL::Function* mtl_fcn = pComputeLibrary->newFunction(NS::String::string("contract", NS::UTF8StringEncoding));

  MTL::ComputePipelineState* mtl_pipe = pDevice->newComputePipelineState(mtl_fcn, &pError);
  if (!mtl_pipe)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  //
  // Data setup
  //
  MTL::Buffer* tensor_A_buffer = pDevice->newBuffer(tensor_A.size() * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer* tensor_B_buffer = pDevice->newBuffer(tensor_B.size() * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer* tensor_C_buffer = pDevice->newBuffer(tensor_C.size() * sizeof(float), MTL::ResourceStorageModeShared);

  float* tensor_A_buffer_ptr = (float*)tensor_A_buffer->contents();
  float* tensor_B_buffer_ptr = (float*)tensor_B_buffer->contents();
  float* tensor_C_buffer_ptr = (float*)tensor_C_buffer->contents();

  for (int i = 0; i < tensor_A.size(); i++)
  {
    tensor_A_buffer_ptr[i] = static_cast<float>(tensor_A(i));
  }
  for (int i = 0; i < tensor_B.size(); i++)
  {
    tensor_B_buffer_ptr[i] = static_cast<float>(tensor_B(i));
  }
  for (int i = 0; i < tensor_C.size(); i++)
  {
    tensor_C_buffer_ptr[i] = static_cast<float>(0.0);
  }

  auto kernel_size = tensor_C.size() * contraction_size;
  MTL::Size grid(kernel_size, 1, 1);
  MTL::Size threadgroup(std::min(128_z, kernel_size), 1, 1);

  //
  // Launch compute
  //
  MTL::CommandQueue* pQueue = pDevice->newCommandQueue();

  MTL::CommandBuffer* pCommandBuffer = pQueue->commandBuffer();
  if (!pCommandBuffer)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  MTL::ComputeCommandEncoder* pComputeEncoder = pCommandBuffer->computeCommandEncoder();
  if (!pComputeEncoder)
  {
    printf("%s", pError->localizedDescription()->utf8String());
    assert(false);
  }

  //
  // Set buffer
  //
  pComputeEncoder->setComputePipelineState(mtl_pipe);
  pComputeEncoder->setBuffer(tensor_A_buffer, 0, 0);
  pComputeEncoder->setBuffer(tensor_B_buffer, 0, 1);
  pComputeEncoder->setBuffer(tensor_C_buffer, 0, 2);
  pComputeEncoder->dispatchThreads(grid, threadgroup);
  pComputeEncoder->endEncoding();
  pCommandBuffer->commit();

  //
  // Device sync
  //
  pCommandBuffer->waitUntilCompleted();

  //
  // Free memory
  //
  pDevice->release();
  mtl_fcn->release();
  mtl_pipe->release();
  pComputeLibrary->release();

  //
  // Copy back to tensors
  //
  for (int i = 0; i < tensor_C.size(); i++)
  {
    tensor_C(i) = static_cast<double>(tensor_C_buffer_ptr[i]);
  }
}

template <size_t reductions, size_t dimension_A, size_t dimension_C, typename accessor, typename const_accessor>
void metal_reduce(
  const TensorView<const_accessor, dimension_A>& tensor_A,
  TensorView<accessor, dimension_C>& tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  if constexpr (reductions == 1)
  {
    metal_reduce(tensor_A, tensor_C, contraction_dimensions[0]);
  }
  else if constexpr (reductions == 2)
  {
    metal_reduce(tensor_A, tensor_C, contraction_dimensions[0], contraction_dimensions[1]);
  }
}

#else

/**
 * @brief Reports that Metal permutation support is unavailable.
 */
template <typename const_accessor, typename accessor, size_t dimension>
void metal_permute(
  TensorView<const_accessor, dimension>& old_tensor,
  TensorView<accessor, dimension>& tensor_permutation,
  Array<size_t, dimension>& permutations)
{
  ::boba::detail::ignore(old_tensor);
  ::boba::detail::ignore(tensor_permutation);
  ::boba::detail::ignore(permutations);
  boba_error("metal_permute requires a Metal build.");
}

/**
 * @brief Reports that Metal contraction support is unavailable.
 */
template <size_t contractions, size_t dimension_A, size_t dimension_B, size_t dimension_C, typename accessor, typename const_accessor>
void metal_contract(
  const TensorView<const_accessor, dimension_A>& tensor_A,
  const TensorView<const_accessor, dimension_B>& tensor_B,
  TensorView<accessor, dimension_C>& tensor_C,
  const boba::Array<size_t, contractions> contraction_dimensions_A,
  const boba::Array<size_t, contractions> contraction_dimensions_B)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_B);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions_A);
  ::boba::detail::ignore(contraction_dimensions_B);
  boba_error("metal_contract requires a Metal build.");
}

/**
 * @brief Reports that Metal reduction support is unavailable.
 */
template <size_t reductions, size_t dimension_A, size_t dimension_C, typename accessor, typename const_accessor>
void metal_reduce(
  const TensorView<const_accessor, dimension_A>& tensor_A,
  TensorView<accessor, dimension_C>& tensor_C,
  const boba::Array<size_t, reductions> contraction_dimensions)
{
  ::boba::detail::ignore(tensor_A);
  ::boba::detail::ignore(tensor_C);
  ::boba::detail::ignore(contraction_dimensions);
  boba_error("metal_reduce requires a Metal build.");
}

#endif

} // namespace detail
} // namespace boba
