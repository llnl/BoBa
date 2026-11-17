#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ml gcc/13.3.1
ml cuda/12.6.0
ml cmake/3.30.5
ml python/3.9.12

COMP_GCC_VER=13.3.1
CMAKE_CXX=/usr/tce/packages/gcc/gcc-${COMP_GCC_VER}/bin/g++

rm -rf example_cuda_kokkos
mkdir -p example_cuda_kokkos
cd example_cuda_kokkos

: "${BOBA_TPL_FARM:?Supply an ENV variable BOBA_TPL_FARM}"
: "${KOKKOS_DIR:?Supply KOKKOS_DIR pointing to the installed Kokkos package}"

if [ ! -d "${BOBA_TPL_FARM}" ]; then
  echo "BOBA_TPL_FARM is set but not a directory: ${BOBA_TPL_FARM}"
  exit 1
fi

echo "BoBa builder phase"
python3 ../../../boba_builder.py \
  --BLT_DIR "${BOBA_TPL_FARM}/tpl/blt" \
  --KOKKOS_DIR "${KOKKOS_DIR}" \
  --boba \
  --clone Off \
  --RAJA Off \
  --kokkos On \
  --camp Off \
  --umpire Off \
  --eigen Off \
  --fmt Off \
  --Caliper Off \
  --hdf5 Off \
  --openmp Off \
  --examples Off \
  --tests Off

RELATIVE_PATH_TO_EXPORT=install/BOBA_cuda_h100/share/boba/cmake
if [[ -z "${BOBA_DIR:-}" ]]; then
  BOBA_DIR=$(pwd)
fi

if [ ! -f "${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT}/boba-config.cmake" ]; then
  echo "BoBa export not found: ${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT}/boba-config.cmake"
  exit 1
fi

echo "Miniapp installation phase"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CMAKE_CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CMAKE_CXX}" \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DENABLE_CUDA=On \
  -DENABLE_OPENMP=Off \
  -DSOURCEFILE=../tests/test_abstractions.cpp \
  -DBOBA_DIR="${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT}" \
  -DCMAKE_VERBOSE_MAKEFILE=On \
  ../

make VERBOSE=1 -j 10

echo "Run phase"
./example
