# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
cmake_minimum_required(VERSION 3.20)
project(example LANGUAGES CXX )
set(USING_BOBA_DEFAULT_CUDA_TOOLCHAIN OFF)
set(CMAKE_CXX_STANDARD 20)

set(SOURCEFILE ../tests/test_boba_tensor_train.cpp)

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_CXX_STANDARD 20)
    message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
    if(NOT CMAKE_CUDA_FLAGS) # assume nvcc as default
        set(USING_BOBA_DEFAULT_CUDA_TOOLCHAIN ON)
        set(HOST_OPT_FLAGS "--expt-relaxed-constexpr --extended-lambda -x cu -Xcompiler -O3 -Xcompiler -finline-functions ")
        set(CMAKE_CUDA_FLAGS "-O3 ${HOST_OPT_FLAGS}")
    endif()
elseif(ENABLE_HIP)
    if(NOT ROCM_PATH)
    # First try finding paths given by the user
        find_path(ROCM_PATH
            hip
            PATHS
            $ENV{ROCM_DIR}
            $ENV{ROCM_PATH}
            $ENV{HIP_PATH}
            ${HIP_PATH}/..
            ${HIP_ROOT_DIR}/../
            ${ROCM_ROOT_DIR}
            /opt/rocm
            NO_DEFAULT_PATH
            NO_CMAKE_ENVIRONMENT_PATH
            NO_CMAKE_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH)

    # If that fails, use CMake default paths
        if(NOT ROCM_PATH)
            find_path(ROCM_PATH hip)
        endif()
    endif()

    # Update CMAKE_PREFIX_PATH to make sure all the configs that hip depends on are
    # found.
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${ROCM_PATH};${ROCM_ROOT_DIR}/lib/cmake")

    find_package(hip REQUIRED CONFIG PATHS  ${HIP_PATH} ${ROCM_PATH} ${ROCM_ROOT_DIR}/lib/cmake/hip)
    message(STATUS "ROCM path:        ${ROCM_PATH}")
    message(STATUS "HIP version:      ${hip_VERSION}")
endif()

find_package(BOBA CONFIG REQUIRED)
add_executable(example ${SOURCEFILE})
target_link_libraries(example PRIVATE BOBA::BOBA)

if(ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries( example PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

if (ENABLE_CUDA)
    message(STATUS "USING_BOBA_DEFAULT_CUDA_TOOLCHAIN: ${USING_BOBA_DEFAULT_CUDA_TOOLCHAIN}")
    set_property(TARGET example PROPERTY CUDA_STANDARD "20")
    set_source_files_properties (${SOURCEFILE} PROPERTIES LANGUAGE CUDA)
    if(ENABLE_OPENMP AND USING_BOBA_DEFAULT_CUDA_TOOLCHAIN)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    endif()
endif()

if (ENABLE_HIP)
    message(STATUS "Configuring HIP build")
    set_property(TARGET example PROPERTY CXX_STANDARD 20) # HIP uses CXX_STANDARD
    set_source_files_properties(${SOURCEFILE} PROPERTIES LANGUAGE HIP)
endif()



