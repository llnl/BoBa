#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

ml gcc/13.3.1
ml cuda/12.6.0
ml cmake/3.30.5
ml python/3.9.12

COMP_GCC_VER=13.3.1
CMAKE_CXX=/usr/tce/packages/gcc/gcc-${COMP_GCC_VER}/bin/g++
COMP_ARCH=sm_90

rm -rf example_cuda_h100
mkdir -p example_cuda_h100
cd example_cuda_h100

: "${BOBA_TPL_FARM:?Supply an ENV variable BOBA_TPL_FARM}"

if [ ! -d "$BOBA_TPL_FARM" ]; then
  echo "BOBA_TPL_FARM is set but not a directory: $BOBA_TPL_FARM"
  exit 1
fi

echo "BOBA_TPL_FARM found: $BOBA_TPL_FARM"

python3  ../../../boba_builder.py \
  --CALIPER_DIR ${BOBA_TPL_FARM}/install/Caliper_cuda_h100 \
  --FMT_DIR ${BOBA_TPL_FARM}/install/fmt_cuda_h100 \
  --UMPIRE_DIR ${BOBA_TPL_FARM}/install/umpire_cuda_h100 \
  --RAJA_DIR ${BOBA_TPL_FARM}/install/RAJA_cuda_h100 \
  --BLT_DIR ${BOBA_TPL_FARM}/tpl/blt \
  --CAMP_DIR ${BOBA_TPL_FARM}/install/camp_cuda_h100 \
  --EIGEN_DIR ${BOBA_TPL_FARM}/install/eigen_cuda_h100 \
  --HDF5_DIR ${BOBA_TPL_FARM}/install/hdf5_cuda_h100 \
  --boba \
  --clone Off \
  --openmp On \
  --examples Off \
  --tests Off

RELATIVE_PATH_TO_EXPORT=install/BOBA_cuda_h100/share/boba/cmake
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

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX} \
  -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CXX} \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=On \
  -DBOBA_DIR=${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT} \
  -DCMAKE_VERBOSE_MAKEFILE=On \
  ../

make VERBOSE=1 -j 10
