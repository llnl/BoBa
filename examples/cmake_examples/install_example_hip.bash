#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ml gcc/13.3.1
ml cmake/3.29.2
ml rocm/7.2.1
ml rocmcc/7.2.1
ml python/3.9.12

rm -rf example_hip
mkdir -p example_hip
cd example_hip

: "${BOBA_TPL_FARM:?Supply an ENV variable BOBA_TPL_FARM}"

if [ ! -d "$BOBA_TPL_FARM" ]; then
  echo "BOBA_TPL_FARM is set but not a directory: $BOBA_TPL_FARM"
  exit 1
fi

echo "BOBA_TPL_FARM found: $BOBA_TPL_FARM"

python3  ../../../boba_builder.py \
  --CALIPER_DIR ${BOBA_TPL_FARM}/install/Caliper_hip \
  --FMT_DIR ${BOBA_TPL_FARM}/install/fmt_hip \
  --UMPIRE_DIR ${BOBA_TPL_FARM}/install/umpire_hip \
  --RAJA_DIR ${BOBA_TPL_FARM}/install/RAJA_hip \
  --BLT_DIR ${BOBA_TPL_FARM}/tpl/blt \
  --CAMP_DIR ${BOBA_TPL_FARM}/install/camp_hip \
  --EIGEN_DIR ${BOBA_TPL_FARM}/install/eigen_hip \
  --HDF5_DIR ${BOBA_TPL_FARM}/install/hdf5_hip \
  --boba \
  --clone Off \
  --openmp Off \
  --examples Off \
  --tests Off

RELATIVE_PATH_TO_EXPORT=install/BOBA_hip/share/boba/cmake
if [[ -z "${BOBA_DIR}" ]]; then
BOBA_DIR=$(pwd)
fi

if [ -f $BOBA_DIR/$RELATIVE_PATH_TO_EXPORT/boba-config.cmake ]; then
  echo "BoBa export found"
else
  echo "Supply an ENV variable BOBA_DIR such that BOBA_DIR/$RELATIVE_PATH_TO_EXPORT/boba-config.cmake exists"
  echo "or that ${SOURCE_DIR}/${RELATIVE_PATH_TO_EXPORT}/boba-config.cmake exists"
  exit 1
fi

#In 7.2.x:
#- ROCm ships a more complete OpenMP offload runtime:
#- libompdevice.a
#- libomptarget.so
#- libomptarget-amdgpu.bc

# if you want to add OpenMP
# we modify cmake defines for CPU only OpenMP otherwise cmake will pull in offload libs and cause linker errors
#-DOpenMP_CXX_FLAGS="-fopenmp" \
#-DOpenMP_CXX_LIB_NAMES="omp" \
#-DOpenMP_omp_LIBRARY=/opt/rocm-7.2.1/lib/llvm/lib/libomp.so \
#-DCMAKE_CXX_FLAGS="-fopenmp --offload-arch=gfx942" \
#-DCMAKE_EXE_LINKER_FLAGS="-fopenmp--offload-arch=gfx942" \

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.1/llvm/bin/amdclang++ \
  -DENABLE_OPENMP=Off \
  -DENABLE_HIP=On \
  -DCMAKE_HIP_ARCHITECTURES="gfx942" \
  -DCMAKE_CXX_FLAGS="--offload-arch=gfx942" \
  -DCMAKE_EXE_LINKER_FLAGS="--offload-arch=gfx942" \
  -DROCM_ROOT_DIR="/opt/rocm-7.2.1" \
  -DBOBA_DIR=${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT} \
  -DCMAKE_VERBOSE_MAKEFILE=On \
  -DHIPBLAS_DIR="/opt/rocm-7.2.1/lib/cmake/hipblas" \
  -DHIPSOLVER_DIR="/opt/rocm-7.2.1/lib/cmake/hipsolver" \
  -DROCBLAS_DIR="/opt/rocm-7.2.1/lib/cmake/rocblas" \
  -DROCSOLVER_DIR="/opt/rocm-7.2.1/lib/cmake/rocsolver" \
  ../

make VERBOSE=1 -j 10
