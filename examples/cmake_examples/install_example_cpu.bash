#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

CMAKE_CXX=c++

if [[ "$HOSTNAME" =~ .*"dane".* ]]; then
ml gcc/13.3.1
ml cmake/3.23.1
COMP_GCC_VER=13.3.1
CMAKE_CXX=/usr/tce/packages/gcc/gcc-${COMP_GCC_VER}/bin/g++
fi
if [[ "$HOSTNAME" =~ .*"rzhound".* ]]; then
ml gcc/13.3.1
ml cmake/3.23.1
COMP_GCC_VER=13.3.1
CMAKE_CXX=/usr/tce/packages/gcc/gcc-${COMP_GCC_VER}/bin/g++
fi

rm -rf example_cpu
mkdir -p example_cpu
cd example_cpu

: "${BOBA_TPL_FARM:?Supply an ENV variable BOBA_TPL_FARM}"

if [ ! -d "$BOBA_TPL_FARM" ]; then
  echo "BOBA_TPL_FARM is set but not a directory: $BOBA_TPL_FARM"
  exit 1
fi

echo "BOBA_TPL_FARM found: $BOBA_TPL_FARM"

python3  ../../../boba_builder.py \
  --BLT_DIR ${BOBA_TPL_FARM}/tpl/blt \
  --FMT_DIR ${BOBA_TPL_FARM}/install/fmt_cpu \
  --EIGEN_DIR ${BOBA_TPL_FARM}/install/eigen_cpu \
  --CALIPER_DIR ${BOBA_TPL_FARM}/install/Caliper_cpu \
  --HDF5_DIR ${BOBA_TPL_FARM}/install/hdf5_cpu \
  --boba \
  --RAJA Off \
  --camp Off \
  --openmp Off \
  --umpire Off \
  --Caliper Off \
  --examples Off \
  --tests Off

RELATIVE_PATH_TO_EXPORT=install/BOBA_cpu/share/boba/cmake
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
  -DBOBA_ENABLE_FMT=On \
  -DENABLE_OPENMP=Off \
  -DBOBA_ENABLE_RAJA=Off \
  -DBOBA_ENABLE_CAMP=Off \
  -DBOBA_ENABLE_CALIPER=Off \
  -DBOBA_ENABLE_UMPIRE=Off \
  -DBOBA_DIR=${BOBA_DIR}/${RELATIVE_PATH_TO_EXPORT} \
  -DCMAKE_VERBOSE_MAKEFILE=On \
  ../

make VERBOSE=1 -j 10
