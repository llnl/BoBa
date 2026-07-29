// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../include/BOBA/boba.hpp"

#ifdef BOBA_UMPIRE
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/strategy/QuickPool.hpp"
#endif

// This source file handles backend setup and initialization.

namespace boba
{

namespace detail
{

#ifdef BOBA_CUDA_LIBS
cublasHandle_t cublas_handle;
cusolverDnHandle_t cusolver_handle;
#ifdef BOBA_CUTENSOR
cutensorHandle_t cutensor_handle;
#endif
#endif

#ifdef BOBA_HIP_LIBS
hipblasHandle_t hipblas_handle;
hipsolverDnHandle_t hipsolver_handle;

rocblas_handle _rocblas_handle;
#endif
#ifdef BOBA_HIPTENSOR
hiptensorHandle_t hiptensor_handle;
#endif

#ifdef BOBA_UMPIRE
using pool_type = umpire::strategy::QuickPool;
umpire::Allocator device_allocator;
umpire::Allocator host_allocator;
#else
detail::DummyAllocatorStruct dummy_allocator;
#endif
} // namespace detail

// ---------------------------------
// Apple Metal
// ---------------------------------
#ifdef BOBA_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#ifdef BOBA_METAL
#include "metal_wrap/wrapper.hpp"
#endif

namespace metal
{

NS::Error* _error;
MTL::Device* _device;
MTL::CommandQueue* _queue;
MTL::CommandBuffer* _command_buffer;

void _initialize()
{
  _device = MTL::CreateSystemDefaultDevice();
  _queue = _device->newCommandQueue();
  _command_buffer = _queue->commandBuffer();
  if (!_command_buffer)
  {
    printf("%s", _error->localizedDescription()->utf8String());
    assert(false);
  }
}

MTL::Device* device()
{
  if (_device == nullptr)
  {
    _initialize();
  }

  return _device;
};

MTL::CommandQueue* queue()
{
  return _queue;
};

MTL::CommandBuffer* command_buffer()
{
  return _command_buffer;
};

void device_synchronize()
{
  _command_buffer->waitUntilCompleted();
};

} // namespace metal

namespace boba
{
namespace detail
{
MTL::Device* pDevice;
}
} // namespace boba
#endif

namespace detail
{

/**
 * \brief Initialize Umpire-backed allocators when Umpire support is enabled.
 */
#ifdef BOBA_UMPIRE
inline void init_umpire()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string host_allocator_base_name = "HOST";
  std::string host_pool_name = "boba::pool_host::" + host_allocator_base_name;
  host_allocator = rm.getAllocator(host_allocator_base_name);
  host_allocator = rm.makeAllocator<pool_type>(host_pool_name, host_allocator);

  if constexpr (boba_cuda_enabled() or boba_hip_enabled())
  {
    std::string device_base_allocator_name = "DEVICE";
    device_allocator = rm.getAllocator(device_base_allocator_name);
    // Customize base allocator into new allocator
    size_t starting_size = 512 * 1024 * 1024;                           // 512MiB initial allocation size from base
    size_t min_size_increase = 1 * 1024 * 1024;                         // 1MiB at least for each subsequent allocation from base
    size_t alignment = 128;                                             // min alignment of each allocation
    size_t num_blocks = 2;                                              // heuristic to determine when to coalesce allocations from base
    auto coalesce_heuristic = pool_type::blocks_releasable(num_blocks); // this helps reduce fragmentation
    std::string device_pool_name = "boba::pool_device::" + device_base_allocator_name;
    device_allocator = rm.makeAllocator<pool_type>(device_pool_name, device_allocator, starting_size, min_size_increase, alignment, coalesce_heuristic);
  }
}
#else
inline void init_umpire()
{
}
#endif

/**
 * \brief Initialize CUDA library handles.
 */
#ifdef BOBA_CUDA_LIBS
inline void init_cuda_libs()
{
  boba::detail::cublas_assert(cublasCreate(&cublas_handle));
  boba::detail::cusolver_assert(cusolverDnCreate(&cusolver_handle));
}

/**
 * \brief Finalize CUDA library handles.
 */
inline void finalize_cuda_libs()
{
  boba::detail::cublas_assert(cublasDestroy(cublas_handle));
  boba::detail::cusolver_assert(cusolverDnDestroy(cusolver_handle));
}
#else
inline void init_cuda_libs()
{
}

inline void finalize_cuda_libs()
{
}
#endif

/**
 * \brief Initialize cuTENSOR.
 */
#ifdef BOBA_CUTENSOR
inline void init_cutensor()
{
  boba::detail::cutensor_assert(cutensorCreate(&cutensor_handle));
}

/**
 * \brief Finalize cuTENSOR.
 */
inline void finalize_cutensor()
{
  boba::detail::cutensor_assert(cutensorDestroy(cutensor_handle));
}
#else
inline void init_cutensor()
{
}

inline void finalize_cutensor()
{
}
#endif

/**
 * \brief Initialize HIP library handles.
 */
#ifdef BOBA_HIP_LIBS
inline void init_hip_libs()
{
  boba::detail::hipblas_assert(hipblasCreate(&hipblas_handle));
  boba::detail::hipsolver_assert(hipsolverDnCreate(&hipsolver_handle));
  boba::detail::rocblas_assert(rocblas_create_handle(&_rocblas_handle));
  rocblas_initialize();
}

/**
 * \brief Finalize HIP library handles.
 */
inline void finalize_hip_libs()
{
  boba::detail::hipblas_assert(hipblasDestroy(hipblas_handle));
  boba::detail::hipsolver_assert(hipsolverDnDestroy(hipsolver_handle));
  boba::detail::rocblas_assert(rocblas_destroy_handle(_rocblas_handle));
}
#else
inline void init_hip_libs()
{
}

inline void finalize_hip_libs()
{
}
#endif

/**
 * \brief Initialize hipTensor.
 */
#ifdef BOBA_HIPTENSOR
inline void init_hiptensor()
{
  boba::detail::hiptensor_assert(hiptensorCreate(&hiptensor_handle));
}

/**
 * \brief Finalize hipTensor.
 */
inline void finalize_hiptensor()
{
  boba::detail::hiptensor_assert(hiptensorDestroy(hiptensor_handle));
}
#else
inline void init_hiptensor()
{
}

inline void finalize_hiptensor()
{
}
#endif

} // namespace detail

// ---------------------------------
// Init/finalize calls
// ---------------------------------
/**
 * \brief Initialize BOBA runtime state and backend handles.
 *
 * Sets up any available backend library handles and configures allocators.
 */
void init()
{
  // Initializes objects and handles needed by TPLS

  // ---------------------------------
  // Cusolver handles
  // ---------------------------------
  checkpoint();
  detail::init_cuda_libs();
  detail::init_cutensor();
  detail::init_hip_libs();
  detail::init_hiptensor();
  checkpoint();

  // ---------------------------------
  // Umpire parameters and objects
  // ---------------------------------
  detail::init_umpire();
  checkpoint();

  // ---------------------------------
  // end
  // ---------------------------------
}

/**
 * \brief Finalize BOBA runtime state and backend handles.
 *
 * Destroys backend library handles.
 */
void finalize()
{
  // ---------------------------------
  // Cusolver handles
  // ---------------------------------
  checkpoint();
  detail::finalize_cuda_libs();
  detail::finalize_cutensor();
  detail::finalize_hip_libs();
  detail::finalize_hiptensor();
  checkpoint();

  // ---------------------------------
  // end
  // ---------------------------------
}

namespace detail
{

#ifdef BOBA_UMPIRE

/**
 * \brief Print leaked Umpire allocations to a stream.
 *
 * \param[in] f output stream
 * \param[in] print_backtrace whether to print allocation backtraces when available
 */
void print_umpire_records(FILE* f, bool print_backtrace)
{
  try
  {
    auto& rm = umpire::ResourceManager::getInstance();

    auto umpire_allocator_ids = rm.getAllocatorIds();

    std::sort(umpire_allocator_ids.begin(), umpire_allocator_ids.end());

    for (auto umpire_allocator_id : umpire_allocator_ids)
    {
      auto umpire_allocator = rm.getAllocator(umpire_allocator_id);

      for (auto const& record : umpire::get_leaked_allocations(umpire_allocator))
      {

        fprintf(f, "%s 0x%012llx: %16zu %s\n", "umpire_record", (unsigned long long)record.ptr, (size_t)record.size, record.strategy->getName().c_str());
#ifdef UMPIRE_ENABLE_BACKTRACE
        if (print_backtrace)
        {
          // print the backtrace associated with the pointer
          auto bt_str = umpire::util::backtracer<>::print(record.allocation_backtrace);
          fprintf(f, "%s\n", bt_str.c_str());
        }
#else
        (void)print_backtrace;
#endif
      }
    }
  }
  catch (const std::exception& e)
  {
    fprintf(f, "Umpire threw an exception %s\n", e.what());
  }
}

/**
 * \brief Print Umpire allocator statistics to a stream.
 *
 * \param[in] f output stream
 */
void print_umpire_stats(FILE* f)
{
  try
  {
    auto& rm = umpire::ResourceManager::getInstance();

    auto umpire_allocator_ids = rm.getAllocatorIds();

    std::sort(umpire_allocator_ids.begin(), umpire_allocator_ids.end());

    for (auto umpire_allocator_id : umpire_allocator_ids)
    {
      auto umpire_allocator = rm.getAllocator(umpire_allocator_id);

      auto const& name = umpire_allocator.getName();
      const int id = umpire_allocator.getId();
      const int parentId = umpire_allocator.getParent() ? umpire_allocator.getParent()->getId() : -1;

      const size_t highWatermark = umpire_allocator.getHighWatermark();
      const size_t currentSize = umpire_allocator.getCurrentSize();
      const size_t actualSize = umpire_allocator.getActualSize();
      const size_t allocationCount = umpire_allocator.getAllocationCount();

      fprintf(f, "%-32s(%2i)", name.c_str(), id);
      fprintf(f, " parent(%2i)", parentId);
      fprintf(f, " %-16s %12zu", "currentSize", currentSize);
      fprintf(f, " %-16s %12zu", "highWatermark", highWatermark);
      fprintf(f, " %-16s %12zu", "actualSize", actualSize);
      fprintf(f, " %-16s %12zu", "allocationCount", allocationCount);
      fprintf(f, "\n");
    }
  }
  catch (const std::exception& e)
  {
    fprintf(f, "Umpire threw an exception %s\n", e.what());
  }
}

#endif

} // namespace detail

} // namespace boba
// ---------------------------------
// end
// ---------------------------------
