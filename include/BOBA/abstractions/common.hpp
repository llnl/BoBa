// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <iostream>
#include <string>

/**
 * \brief
 * Top level namespace
 */

namespace boba
{

/**
 * @brief Initializes BoBa runtime state.
 */
extern void init();

/**
 * @brief Finalizes BoBa runtime state.
 */
extern void finalize();

/**
 * \brief
 * Namespace for functions and classes most users can ignore.
 */

namespace detail
{

// -------------------------------------------------------------------------------------
// Printing functions
// -------------------------------------------------------------------------------------

const std::string ascii =
  R"(
            ___             ___           
    o O O  | _ )    ___    | _ )   __ _   
   o       | _ \   / _ \   | _ \  / _` |  
  TS__[O]  |___/   \___/   |___/  \__,_|  
 {======|_|"""""|_|"""""|_|"""""|_|"""""| 
./o--000'"`-0-0-'"`-0-0-'"`-0-0-'"`-0-0-' 
)";

const std::string license =
  "BoBa is distributed under the Apache-2.0 with LLVM exception License.\n";

} // namespace detail

/**
 * @brief Returns true when the current build is a CI build.
 */
inline constexpr bool is_ci_mode()
{
#if defined(BOBA_CI)
  return true;
#else
  return false;
#endif
}

/**
 * \brief
 * The most important function in the whole code. If you don't have splash
 * art are you even a real library?
 */

inline void splash()
{
  if constexpr (not is_ci_mode())
  {
    std::cout << detail::ascii << std::endl
              << detail::license << std::endl;
  }
}

/**
 * @brief Returns an indentation string.
 * @param i Number of indentation levels.
 * @return A string containing four spaces per indentation level.
 */
inline std::string write_indent(size_t i)
{
  return std::string(static_cast<unsigned long>(4 * i), ' ');
}

// -------------------------------------------------------------------------------------
// Execution spaces
// -------------------------------------------------------------------------------------

enum struct execution_space : size_t
{
  CPU,
  CUDA,
  HIP
};

constexpr execution_space host_space = execution_space::CPU;

/**
 * @brief Returns true when an execution space is CUDA.
 * @param space Execution space to inspect.
 * @return `true` for CUDA, `false` otherwise.
 */
consteval bool is_cuda(execution_space space)
{
  return space == execution_space::CUDA;
}

/**
 * @brief Returns true when an execution space is the host backend.
 * @param space Execution space to inspect.
 * @return `true` for CPU, `false` otherwise.
 */
consteval bool is_host(execution_space space)
{
  return space == execution_space::CPU;
}

/**
 * @brief Returns true when an execution space is HIP.
 * @param space Execution space to inspect.
 * @return `true` for HIP, `false` otherwise.
 */
consteval bool is_hip(execution_space space)
{
  return space == execution_space::HIP;
}

/**
 * @brief Returns true when an execution space is a GPU backend.
 * @param space Execution space to inspect.
 * @return `true` for CUDA and HIP, `false` otherwise.
 */
consteval bool is_gpu(execution_space space)
{
  return is_cuda(space) or is_hip(space);
}

/**
 * @brief Returns true when CUDA support is enabled.
 */
inline constexpr bool boba_cuda_enabled()
{
#if defined(BOBA_CUDA)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when HIP support is enabled.
 */
inline constexpr bool boba_hip_enabled()
{
#if defined(BOBA_HIP)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when CPU support is enabled.
 */
inline constexpr bool is_cpu_enabled()
{
#if defined(BOBA_CPU)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when MPI support is enabled.
 */
inline constexpr bool is_boba_mpi_enabled()
{
#if defined(BOBA_ENABLE_MPI)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when MATLAB support is enabled.
 */
inline constexpr bool is_boba_MATLAB_enabled()
{
#if defined(BOBA_MATLAB)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when HDF5 support is enabled.
 */
inline constexpr bool is_boba_hdf5_enabled()
{
#if defined(BOBA_HDF5)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when the current build has debug assertions enabled.
 */
inline constexpr bool is_boba_debug_mode()
{
#if defined(BOBA_DEBUG)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when Eigen support is enabled.
 */
inline constexpr bool boba_eigen_enabled()
{
#if defined(BOBA_ENABLE_EIGEN)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when RAJA support is enabled.
 */
inline constexpr bool boba_raja_enabled()
{
#if defined(BOBA_RAJA)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when Umpire support is enabled.
 */
inline constexpr bool boba_umpire_enabled()
{
#if defined(BOBA_UMPIRE)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when CUDA library support is enabled.
 */
inline constexpr bool boba_cuda_libs_enabled()
{
#if defined(BOBA_CUDA_LIBS)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when HIP library support is enabled.
 */
inline constexpr bool boba_hip_libs_enabled()
{
#if defined(BOBA_HIP_LIBS)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when cuTensor support is enabled.
 */
inline constexpr bool boba_cutensor_enabled()
{
#if defined(BOBA_CUTENSOR)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when hipTensor support is enabled.
 */
inline constexpr bool boba_hiptensor_enabled()
{
#if defined(BOBA_HIPTENSOR)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when Eigen tensor support is enabled.
 */
inline constexpr bool boba_eigen_tensor_enabled()
{
#if defined(BOBA_EIGEN_TENSOR)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns true when Metal support is enabled.
 */
inline constexpr bool boba_metal_enabled()
{
#if defined(BOBA_METAL)
  return true;
#else
  return false;
#endif
}

/**
 * @brief Returns the configured default execution space.
 */
inline constexpr execution_space default_execution_space = []() consteval
{
  if constexpr (boba_cuda_enabled())
  {
    return execution_space::CUDA;
  }
  else if constexpr (boba_hip_enabled())
  {
    return execution_space::HIP;
  }
  else
  {
    return execution_space::CPU;
  }
}();

namespace detail
{

/**
 * \brief
 * Returns a string version of the execution space name
 * @param space Execution space to name.
 * @return The execution space name.
 */

inline std::string execution_space_name(::boba::execution_space space = default_execution_space)
{
  switch (space)
  {
  case execution_space::CPU:
    return "CPU";
  case execution_space::CUDA:
    return "CUDA";
  case execution_space::HIP:
    return "HIP";
  default:
    return "unknown space";
  }
}

} // namespace detail

/**
 * @brief Writes an execution-space name to a stream.
 * @param stream Output stream.
 * @param space Execution space to print.
 * @return `stream`.
 */
inline std::ostream& operator<<(std::ostream& stream, ::boba::execution_space space)
{
  stream << detail::execution_space_name(space);
  return stream;
}

/**
 * @brief Returns the configured default execution-space name.
 * @return The name of `default_execution_space`.
 */
inline std::string default_execution_space_name()
{
  return ::boba::detail::execution_space_name(default_execution_space);
}

/**
 * @brief Appends compile-time build flags to a name suffix.
 * @param name_flag Initial name suffix.
 * @return `name_flag` with enabled build flags appended.
 */
inline std::string name_flag(std::string name_flag = "")
{
  if constexpr (is_boba_mpi_enabled())
  {
    name_flag += "_mpi";
  }
#ifdef BOBA_ASAN
  name_flag += "_asan";
#endif
#ifdef BOBA_UBSAN
  name_flag += "_ubsan";
#endif
  if constexpr (is_cpu_enabled())
  {
    name_flag += "_cpu";
  }
  if constexpr (boba_cuda_enabled())
  {
    name_flag += "_cuda";
  }
  if constexpr (boba_hip_enabled())
  {
    name_flag += "_hip";
  }
  if constexpr (boba_umpire_enabled())
  {
    name_flag += "_ump";
  }
#ifdef BOBA_PROFILING
  name_flag += "_prof";
#endif
#ifdef BOBA_ENABLE_CALIPER
  name_flag += "_cali";
#endif
  if constexpr (is_boba_debug_mode())
  {
    name_flag += "_debug";
  }
  return name_flag;
}

namespace detail
{

// -----------------------------------------------------
// Host-device
// -----------------------------------------------------
#ifdef BOBA_CUDA
#define __boba_host_device__ __host__ __device__
#define __boba_device__ __device__
#define __boba_host__ __host__
#elif defined(BOBA_HIP)
#define __boba_host_device__ __host__ __device__
#define __boba_device__ __device__
#define __boba_host__ __host__
#else
#define __boba_host_device__ /* nothing */
#define __boba_device__      /* nothing */
#define __boba_host__        /* nothing */
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define BOBA_DEVICE_CODE
#endif

/**
 * @brief Suppresses unused-argument warnings.
 * @tparam Ts Ignored argument types.
 */
template <typename... Ts>
__boba_host_device__ constexpr void ignore(Ts&&...) noexcept
{
}

/**
 * \brief
 * Synchronize the host and the device
 * @tparam space Execution space to synchronize.
 */

template <boba::execution_space space>
inline void synchronize();

/**
 * \brief
 * Device agnostic synchronizer
 * Performs a host/device synchronization for the active accelerator backend.
 */

inline void device_sync() noexcept
{
#if defined(BOBA_CUDA) && !defined(BOBA_DEVICE_CODE)
  synchronize<boba::execution_space::CUDA>();
#elif defined(BOBA_HIP) && !defined(BOBA_DEVICE_CODE)
  synchronize<boba::execution_space::HIP>();
#endif
}

/**
 * @brief Synchronizes CUDA threads within the current block.
 */
__boba_host_device__ inline void cuda_thread_synchronize();

/**
 * @brief Synchronizes HIP threads within the current block.
 */
__boba_host_device__ inline void hip_thread_synchronize();

/**
 * \brief
 * Synchronize the threads in a GPU block. Does nothing on CPUs
 * Synchronizes all threads within a block
 */

__boba_host_device__ inline void thread_synchronize()
{
#ifdef BOBA_DEVICE_CODE
#if defined(BOBA_CUDA)
  cuda_thread_synchronize();
#endif
#if defined(BOBA_HIP)
  hip_thread_synchronize();
#endif
#endif
}

#ifdef BOBA_DEVICE_CODE
#define __boba_shared__ __shared__
#else
#define __boba_shared__ /* nothing */
#endif

} // namespace detail

/**
 * @brief Reports whether the current compilation context is device code.
 * @return `true` when compiling device code, `false` when compiling host code.
 */
constexpr bool is_device_context() noexcept
{
#ifdef BOBA_DEVICE_CODE
  return true;
#else
  return false;
#endif
}

} // namespace boba
