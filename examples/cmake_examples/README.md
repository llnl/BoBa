# Example of Using BoBa in your CMake-enabled Project

The example in this folder demonstrates how a client can set up CMake to integrate BoBa into their code. The test in this folder is a stripped-down variant of one of BoBa's tests, [test_boba_tensor_train.cpp](../tests/test_boba_tensor_train.cpp).

### Step 1: Set BOBA_DIR

We need to set this environment variable to the location of the cloned BoBa repo. It is used by the example installation scripts.

> BOBA_DIR=/path/to/boba/repo

### Step 2: Install BoBa TPLs and  BoBa Export

These examples assume you have already built the needed TPL dependencies. The following scripts show you how to create the BoBa export, which can be used in your code. They can be modified for your own code:

- [install_example_cpu.bash](install_example_cpu.bash)
- [install_example_h100.bash](install_example_h100.bash)
- [install_example_hip.bash](install_example_hip.bash)
- and more...

The [install_example_cpu.bash](install_example_cpu.bash) script demonstrates a CPU-based build, which loads compilers and other required modules. It will set up a build directory structure for CMake, delineated by variant; **example_cpu** in this case (see below). The output executable is called **"example"** and is found in that directory after `make -j 10` is called from the bash script.

- build
    - example_cpu
    - example_cuda
    - example_hip

Note that the above scripts define a path for CMake ``-DBOBA_DIR=${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT}``, which is the location of the BoBa shared export, something like this:

> -DBOBA_DIR=${BOBA_DIR}/install/BOBA_cpu/share/boba/cmake

### Step 3:  Run the example

Enter the created directory `example_xxx`:

> cd build/example_cpu

Run the produced example:

> ./example

## Additional Discussion

In the CMakeLists.txt there are only a few lines needed to bring in the shared export:

```
    find_package(BOBA REQUIRED)
    add_executable(example test_boba_tensor_train.cpp)
    target_link_libraries(example PRIVATE BOBA::BOBA)
```

The CUDA or HIP variants are a little more complicated in that we also add prior `find_package(CUDAToolkit REQUIRED)` or `find_package(hip REQUIRED ...)` since we need to link against GPU based runtime, BLAS, and Solver libraries, and those packages are usually "System" installed. Additionally for the CUDA variant we need to indicate which files should be compiled by nvcc:

> set_source_files_properties (test_boba_tensor_train.cpp PROPERTIES LANGUAGE CUDA)

Additionally, we set a few compiler flags needed to build the test, in this case a minimal set for a release build.

# BoBa CMake Export

In the BoBa installation tree pointed to by `-DBOBA_DIR`, there are several files used to set up everything so that `find_package(BOBA ..)` will detect and bring in information on the third-party libraries (TPLs) used in the BoBa library. The TPLs involved in the export are listed in order of occurrence:
1. camp [https://github.com/LLNL/camp]
2. RAJA [https://raja.readthedocs.io/en/develop/]
3. umpire [https://umpire.readthedocs.io/en/develop/]
4. eigen [https://gitlab.com/libeigen/eigen]
5. fmt [https://fmt.dev/latest/index.html]
6. Caliper [https://software.llnl.gov/Caliper/]
7. HDF5 [https://www.hdfgroup.org/solutions/hdf5/]
8. hipblas [https://hipblas.readthedocs.io/en/latest/] (HIP only)
9. hipsolver [https://hipsolver.readthedocs.io/en/latest/] (HIP only)
10. rocblas [https://rocm.docs.amd.com/projects/rocBLAS/en/latest/] (HIP only)
11. rocsolver [https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/] (HIP only)

Each of these TPLs is detected by the same CMake pattern (using RAJA for example):

```
    if (NOT TARGET RAJA)
      set(BOBA_RAJA_DIR "@RAJA_DIR@")
      if(NOT raja_DIR)
        set(raja_DIR ${BOBA_RAJA_DIR})
      endif()

      find_dependency(RAJA CONFIG NO_DEFAULT_PATH PATHS
        ${raja_DIR}
        ${raja_DIR}/lib/cmake/raja
        ${BOBA_PREFIX_PATH})
    endif ()
```

Where **"@RAJA_DIR@"** is populated by CMake at build time coming from the `install_example_xxx.bash` script entry (cpu example):
> -DRAJA_DIR=${BOBA_DIR}/install/raja_cpu/lib/cmake/raja/ \

> The client can override this location by specifying their own `raja_DIR` in their script. Then CMake calls the `find_dependency()` macro, which is a wrapper around `find_package` but also includes the `QUIET` and `REQUIRED` settings used in the original `find_package` call. In RAJA's case it is a required package. See [SetupBoBaThirdParty.cmake](../../cmake/SetupBoBaThirdParty.cmake).

You can refer to [BOBA-config.cmake.in](../../share/boba/cmake/BOBA-config.cmake.in) which is used to build/populate the TPL export, along with a few other settings.
