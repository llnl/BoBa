# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#------------------------------------------------------------------------------
# 3rd Party Dependencies
#------------------------------------------------------------------------------

# Policy to use <PackageName>_ROOT variable in find_<Package> commands
# Policy added in 3.12+
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(TPL_DEPS)

#------------------------------------------------------------------------------
# Create global variable to toggle between GPU targets
#------------------------------------------------------------------------------
if(BOBA_ENABLE_CUDA)
    set(boba_device_depends cuda CACHE STRING "" FORCE)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_NAMEFLAG "_cuda")
endif()
if(BOBA_ENABLE_HIP)
    set(boba_device_depends blt::hip CACHE STRING "" FORCE)
    set(HIP_NAMEFLAG "_hip")
endif()

#------------------------------------------------------------------------------
# fmt
#------------------------------------------------------------------------------
if ((UMPIRE_DIR) AND NOT FMT_DIR)
    message(FATAL_ERROR "FMT_DIR is required if RAJA_DIR or UMPIRE_DIR is provided.")
endif()

if (FMT_DIR AND BOBA_ENABLE_FMT)
    if (NOT EXISTS "${FMT_DIR}")
        message(FATAL_ERROR "Given FMT_DIR does not exist: ${FMT_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${FMT_DIR}")
        message(FATAL_ERROR "Given FMT_DIR is not a directory: ${FMT_DIR}")
    endif()

    find_package(fmt REQUIRED PATHS ${FMT_DIR} NO_DEFAULT_PATH)

    message(STATUS "Checking for expected fmt target 'fmt'")
    if (NOT TARGET fmt::fmt)
        message(FATAL_ERROR "fmt failed to load: ${FMT_DIR}")
    else()
        message(STATUS "fmt loaded: ${FMT_DIR}")
        set(FMT_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "fmt support is OFF")
    set(FMT_FOUND FALSE CACHE BOOL "" FORCE)
endif()

if ((BOBA_ENABLE_UMPIRE) AND NOT FMT_FOUND)
    message(FATAL_ERROR "FMT_DIR is required if Umpire is enabled.")
endif()
#------------------------------------------------------------------------------
# Camp (needed by RAJA and Umpire)
#------------------------------------------------------------------------------
if ((RAJA_DIR OR UMPIRE_DIR) AND NOT CAMP_DIR)
    message(FATAL_ERROR "CAMP_DIR is required if RAJA_DIR or UMPIRE_DIR is provided.")
endif()

if (CAMP_DIR AND BOBA_ENABLE_CAMP)
    if (NOT EXISTS "${CAMP_DIR}")
        message(FATAL_ERROR "Given CAMP_DIR does not exist: ${CAMP_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${CAMP_DIR}")
        message(FATAL_ERROR "Given CAMP_DIR is not a directory: ${CAMP_DIR}")
    endif()

    find_package(camp REQUIRED PATHS ${CAMP_DIR})

    message(STATUS "Checking for expected Camp target 'camp'")
    if (NOT TARGET camp)
        message(FATAL_ERROR "Camp failed to load: ${CAMP_DIR}")
    else()
        message(STATUS "Camp loaded: ${CAMP_DIR}")
        set(CAMP_FOUND TRUE CACHE BOOL "" FORCE)
    endif()

    # Note: camp sets a compile feature that is not available on XL
    set_target_properties(camp PROPERTIES INTERFACE_COMPILE_FEATURES "")
else()
    message(STATUS "Camp support is OFF")
    set(CAMP_FOUND FALSE CACHE BOOL "" FORCE)
endif()

#------------------------------------------------------------------------------
# UMPIRE
#------------------------------------------------------------------------------
if (UMPIRE_DIR AND BOBA_ENABLE_UMPIRE)
    if (NOT EXISTS "${UMPIRE_DIR}")
        message(FATAL_ERROR "Given UMPIRE_DIR does not exist: ${UMPIRE_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${UMPIRE_DIR}")
        message(FATAL_ERROR "Given UMPIRE_DIR is not a directory: ${UMPIRE_DIR}")
    endif()

    find_package(umpire REQUIRED PATHS ${UMPIRE_DIR} )

    message(STATUS "Checking for expected Umpire target 'umpire'")
    if (NOT TARGET umpire)
        message(FATAL_ERROR "Umpire failed to load: ${UMPIRE_DIR}")
    else()
        message(STATUS "Umpire loaded: ${UMPIRE_DIR}")
        set(UMPIRE_FOUND TRUE CACHE BOOL "" FORCE)
        set(UMP_NAMEFLAG "_ump")
    endif()
else()
    message(STATUS "Umpire support is OFF")
    set(UMPIRE_FOUND FALSE CACHE BOOL "" FORCE)
endif()


#------------------------------------------------------------------------------
# RAJA
#------------------------------------------------------------------------------
if (RAJA_DIR AND BOBA_ENABLE_RAJA)
    if (NOT EXISTS "${RAJA_DIR}")
        message(FATAL_ERROR "Given RAJA_DIR does not exist: ${RAJA_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${RAJA_DIR}")
        message(FATAL_ERROR "Given RAJA_DIR is not a directory: ${RAJA_DIR}")
    endif()

    find_package(RAJA REQUIRED PATHS ${RAJA_DIR} )

    message(STATUS "Checking for expected RAJA target 'RAJA'")
    if (NOT TARGET RAJA)
        message(FATAL_ERROR "RAJA failed to load: ${RAJA_DIR}")
    else()
        message(STATUS "RAJA loaded: ${RAJA_DIR}")
        set(RAJA_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "RAJA support is OFF" )
    set(RAJA_FOUND FALSE CACHE BOOL "" FORCE)
endif()

#------------------------------------------------------------------------------
# EIGEN
#------------------------------------------------------------------------------
if (EIGEN_DIR AND BOBA_ENABLE_EIGEN)
    if (NOT EXISTS "${EIGEN_DIR}")
        message(FATAL_ERROR "Given EIGEN_DIR does not exist: ${EIGEN_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${EIGEN_DIR}")
        message(FATAL_ERROR "Given EIGEN_DIR is not a directory: ${EIGEN_DIR}")
    endif()

    find_package(Eigen3 REQUIRED PATHS ${EIGEN_DIR} )

    message(STATUS "Checking for expected Eigen target 'Eigen3::Eigen'")
    if (NOT TARGET Eigen3::Eigen)
        message(FATAL_ERROR "Eigen failed to load: ${EIGEN_DIR}")
    else()
        message(STATUS "Eigen loaded: ${EIGEN_DIR}")
        set(EIGEN_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "Eigen support is OFF")
    set(EIGEN_FOUND FALSE CACHE BOOL "" FORCE)
endif()

#------------------------------------------------------------------------------
# Caliper
#------------------------------------------------------------------------------
if (CALIPER_DIR AND BOBA_ENABLE_CALIPER)
    if (NOT EXISTS "${CALIPER_DIR}")
        message(FATAL_ERROR "Given CALIPER_DIR does not exist: ${CALIPER_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${CALIPER_DIR}")
        message(FATAL_ERROR "Given CALIPER_DIR is not a directory: ${CALIPER_DIR}")
    endif()

    find_package(caliper REQUIRED PATHS ${CALIPER_DIR} )

    message(STATUS "Checking for expected Caliper target 'caliper'")
    if (NOT TARGET caliper)
        message(FATAL_ERROR "Caliper failed to load: ${CALIPER_DIR}")
    else()
        message(STATUS "Caliper loaded: ${CALIPER_DIR}")
        get_target_property(CALIPER_INCLUDE_DIRS_BOBA caliper INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "Caliper Includes: ${CALIPER_INCLUDE_DIRS_BOBA}")
        set_property(TARGET caliper
                     APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                    ${CALIPER_INCLUDE_DIRS_BOBA})
        set(CALIPER_FOUND TRUE CACHE BOOL "" FORCE)
        set(CALI_NAMEFLAG "_cali" )
    endif()
else()
    message(STATUS "Caliper support is OFF")
    set(CALIPER_FOUND FALSE CACHE BOOL "" FORCE)
endif()

#------------------------------------------------------------------------------
# HDF5
#------------------------------------------------------------------------------
if (HDF5_DIR AND BOBA_ENABLE_HDF5)

    if (NOT EXISTS "${HDF5_DIR}")
        message(FATAL_ERROR "Given HDF5_DIR does not exist: ${HDF5_DIR}")
    endif()

    if (NOT IS_DIRECTORY "${HDF5_DIR}")
        message(FATAL_ERROR "Given HDF5_DIR is not a directory: ${HDF5_DIR}")
    endif()

    find_package(hdf5 REQUIRED PATHS ${HDF5_DIR} )
    message(STATUS "Checking for expected HDF5 target 'hdf5-static'")
    if (NOT TARGET hdf5-static)
        message(FATAL_ERROR "HDF5 failed to load: ${HDF5_DIR}")
    else()
        message(STATUS "HDF5 loaded: ${HDF5_DIR}")
        set(HDF5_FOUND TRUE CACHE BOOL "" FORCE)
    endif()
else()
    message(STATUS "HDF5 support is OFF" )
    set(HDF5_FOUND FALSE CACHE BOOL "" FORCE)
endif()

#------------------------------------------------------------------------------
# HIP
#------------------------------------------------------------------------------
if(BOBA_ENABLE_HIP AND BOBA_HIP_LIBS)
    if (HIPSOLVER_DIR)
        if (NOT EXISTS "${HIPSOLVER_DIR}")
            message(FATAL_ERROR "Given HIPSOLVER_DIR does not exist: ${HIPSOLVER_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${HIPSOLVER_DIR}")
            message(FATAL_ERROR "Given HIPSOLVER_DIR is not a directory: ${HIPSOLVER_DIR}")
        endif()

        find_package(hipsolver REQUIRED PATHS ${HIPSOLVER_DIR} )

        message(STATUS "Checking for expected HIPSOLVER target 'roc::hipsolver'")
        if (NOT TARGET roc::hipsolver)
            message(FATAL_ERROR "HIPSOLVER failed to load: ${HIPSOLVER_DIR}")
        else()
            message(STATUS "hipsolver loaded: ${HIPSOLVER_DIR}")
            get_target_property(HIPSOLVER_INCLUDE_DIRS_BOBA roc::hipsolver INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "HIPSOLVER Includes: ${HIPSOLVER_INCLUDE_DIRS_BOBA}")
            set_property(TARGET roc::hipsolver
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${HIPSOLVER_INCLUDE_DIRS_BOBA})
            set(HIPSOLVER_FOUND TRUE CACHE BOOL "" FORCE)
        endif()
    else()
        message(STATUS "hipsolver support is OFF")
        set(HIPSOLVER_FOUND FALSE CACHE BOOL "" FORCE)
    endif()

    if (HIPBLAS_DIR)
        if (NOT EXISTS "${HIPBLAS_DIR}")
            message(FATAL_ERROR "Given HIPBLAS_DIR does not exist: ${HIPBLAS_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${HIPBLAS_DIR}")
            message(FATAL_ERROR "Given HIPBLAS_DIR is not a directory: ${HIPBLAS_DIR}")
        endif()

        find_package(hipblas REQUIRED PATHS ${HIPBLAS_DIR} )

        message(STATUS "Checking for expected hipblas target 'roc::hipblas'")
        if (NOT TARGET roc::hipblas)
            message(FATAL_ERROR "hipblas failed to load: ${HIPBLAS_DIR}")
        else()
            message(STATUS "hipblas loaded: ${HIPBLAS_DIR}")
            get_target_property(HIPBLAS_INCLUDE_DIRS_BOBA roc::hipblas INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "hipblas Includes: ${HIPBLAS_INCLUDE_DIRS_BOBA}")
            set_property(TARGET roc::hipblas
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${HIPBLAS_INCLUDE_DIRS_BOBA})
            set(HIPBLAS_FOUND TRUE CACHE BOOL "" FORCE)
        endif()
    else()
        message(STATUS "hipblas support is OFF")
        set(HIPBLAS_FOUND FALSE CACHE BOOL "" FORCE)
    endif()

    if (ROCSOLVER_DIR)
        if (NOT EXISTS "${ROCSOLVER_DIR}")
            message(FATAL_ERROR "Given ROCSOLVER_DIR does not exist: ${ROCSOLVER_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${ROCSOLVER_DIR}")
            message(FATAL_ERROR "Given ROCSOLVER_DIR is not a directory: ${ROCSOLVER_DIR}")
        endif()

        find_package(ROCSOLVER REQUIRED PATHS ${ROCSOLVER_DIR} )

        message(STATUS "Checking for expected ROCSOLVER target 'roc::rocsolver'")
        if (NOT TARGET roc::rocsolver)
            message(FATAL_ERROR "ROCSOLVER failed to load: ${ROCSOLVER_DIR}")
        else()
            message(STATUS "rocsolver loaded: ${ROCSOLVER_DIR}")
            get_target_property(ROCSOLVER_INCLUDE_DIRS_BOBA roc::rocsolver INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "ROCSOLVER Includes: ${ROCSOLVER_INCLUDE_DIRS_BOBA}")
            set_property(TARGET roc::rocsolver
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${ROCSOLVER_INCLUDE_DIRS_BOBA})
            set(ROCSOLVER_FOUND TRUE CACHE BOOL "" FORCE)
        endif()
    else()
        message(STATUS "rocsolver support is OFF")
        set(ROCSOLVER_FOUND FALSE CACHE BOOL "" FORCE)
    endif()

    if (ROCBLAS_DIR)
        if (NOT EXISTS "${ROCBLAS_DIR}")
            message(FATAL_ERROR "Given ROCBLAS_DIR does not exist: ${ROCBLAS_DIR}")
        endif()

        if (NOT IS_DIRECTORY "${ROCBLAS_DIR}")
            message(FATAL_ERROR "Given ROCBLAS_DIR is not a directory: ${ROCBLAS_DIR}")
        endif()

        find_package(rocblas REQUIRED PATHS ${ROCBLAS_DIR} )

        message(STATUS "Checking for expected rocblas target 'roc::rocblas'")
        if (NOT TARGET roc::rocblas)
            message(FATAL_ERROR "rocblas failed to load: ${ROCBLAS_DIR}")
        else()
            message(STATUS "rocblas loaded: ${ROCBLAS_DIR}")
            get_target_property(ROCBLAS_INCLUDE_DIRS_BOBA roc::rocblas INTERFACE_INCLUDE_DIRECTORIES)
            message(STATUS "rocblas Includes: ${ROCBLAS_INCLUDE_DIRS_BOBA}")
            set_property(TARGET roc::rocblas
                 APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                ${ROCBLAS_INCLUDE_DIRS_BOBA})
            set(ROCBLAS_FOUND TRUE CACHE BOOL "" FORCE)
        endif()
    else()
        message(STATUS "rocblas support is OFF")
        set(ROCBLAS_FOUND FALSE CACHE BOOL "" FORCE)
    endif()
endif()

#------------------------------------------------------------------------------
# Remove exported OpenMP flags because they are not language agnostic
#------------------------------------------------------------------------------
set(_props)
if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13.0" )
    list(APPEND _props INTERFACE_LINK_OPTIONS)
endif()
list(APPEND _props INTERFACE_COMPILE_OPTIONS)

foreach(_target RAJA camp umpire umpire_alloc Eigen3::Eigen caliper)
    if(TARGET ${_target})
#        message(STATUS "Removing OpenMP Flags from target[${_target}]")

        foreach(_prop ${_props})
            get_target_property(_flags ${_target} ${_prop})
            if ( _flags )
                string( REPLACE "${OpenMP_CXX_FLAGS}" ""
                        correct_flags "${_flags}" )
                string( REPLACE "${OpenMP_Fortran_FLAGS}" ""
                        correct_flags "${correct_flags}" )

                set_target_properties( ${_target} PROPERTIES ${_prop} "${correct_flags}" )
            endif()
        endforeach()
    endif()
endforeach()

# Newer versions of RAJA keeps its flags in a specific target
if(TARGET RAJA)
    get_target_property(_flags RAJA INTERFACE_LINK_LIBRARIES)
    if ( _flags )
#        message(STATUS "RAJA Flags: ${_flags}")
        list(REMOVE_ITEM _flags "RAJA::openmp")
        set_target_properties( RAJA PROPERTIES INTERFACE_LINK_LIBRARIES "${_flags}" )
    endif()
endif()


#------------------------------------------------------------------------------
# Targets that need to be exported but don't have a CMake config file
#------------------------------------------------------------------------------
blt_list_append(TO TPL_DEPS ELEMENTS cuda cuda_runtime IF BOBA_ENABLE_CUDA)
blt_list_append(TO TPL_DEPS ELEMENTS CUDA::cublas CUDA::cublasLt CUDA::cusolver CUDA::cusparse IF BOBA_ENABLE_CUDA)
blt_list_append(TO TPL_DEPS ELEMENTS blt_hip blt_hip_runtime IF BOBA_ENABLE_HIP)
blt_list_append(TO TPL_DEPS ELEMENTS openmp IF BOBA_ENABLE_OPENMP)

foreach(dep ${TPL_DEPS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        message(STATUS "Putting ${dep} in export set")
        install(TARGETS  ${dep}
                EXPORT  BOBATargets
                PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ GROUP_WRITE
                DESTINATION  lib
                )
    endif()
endforeach()
