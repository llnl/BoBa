#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
import os
import sys

#
# This script clones the needed BoBa tpls according to the files defined in recipes/
#

boba_dir = Path(os.path.dirname(__file__)).absolute()
sys.path.append(f"{boba_dir}/scripts/util")

import socket
import subprocess
from os.path import join, exists
from shutil import rmtree, copytree, copyfile
import re
import argparse
import json

from machines import \
    get_machine

from build_util import \
    safe_mkdir, \
    safe_rmdir, \
    safe_copy_dir, \
    command_runner, \
    tpl_splash, \
    get_tpl_dir_string, \
    check_external_tpl_cmake_dir

parser = argparse.ArgumentParser(description='BoBa TPL builder')

parser.add_argument(
    '--no_build',
    action='store_true',
    help='Clones submodule repos that will be needed, but does not build them, useful for CI')

parser.add_argument(
    '--clone',
    choices= ['Force', 'Off', 'Default'],
    default= 'Default',
    help='Prevents cloning of submodules, useful for CI')

parser.add_argument(
    '--force',
    action='store_true',
    help='Forces a TPL clone and build')

parser.add_argument(
    '--boba',
    action='store_true',
    help='Install BoBa')

parser.add_argument(
    '--ci',
    action='store_true',
    help='Use optimzations intended for gitlab CI.')

parser.add_argument(
    '--recipe',
    help='Use a specific recipe.')

# Options and their default value for cmake
options = {}
options['cpu'] = 'On'
options['checkpoints'] = 'Off'
options['debug'] = 'Off'
options['openmp'] = 'Off'
options['examples'] = 'Off'
options['tests'] = 'Off'
options['tutorials'] = 'Off'
options['mpi'] = 'Off'

for (key, value) in options.items():
    parser.add_argument(
        f'--{key}',
        choices= ['On', 'Off'],
        default= value,
        help=f'Compile with flag: {key}')

tpls = {}
tpls['hdf5'] = {}
tpls['blt'] = {}
tpls['fmt'] = {}
tpls['camp'] = {}
tpls['RAJA'] = {}
tpls['umpire'] = {}
tpls['eigen'] = {}
tpls['Caliper'] = {}
tpls['metal_cpp_compute'] = {}

for tpl in tpls.keys():
    tpls[tpl]['cmake'] = {}
    default_choice = 'On'
    if tpl == 'metal_cpp_compute':
        default_choice = 'Off'
    parser.add_argument(
        f'--{tpl}',
        choices=['On', 'Off'],
        default=default_choice,
        help=f'Build {tpl}')
    tpl_dir_string = get_tpl_dir_string(tpl)
    parser.add_argument(
        f'--{tpl_dir_string}',
        help=f'Provide path to pre-installed {tpl}')

args = parser.parse_args()

if not args.ci and os.environ.get("BOBA_CI"):
    args.ci = True

no_build = args.no_build
install_boba = args.boba
enable_openmp = args.openmp
print(f"enable_openmp = {enable_openmp}")

tpl_splash(f"Machine information")

machine, system, arch = get_machine()

# Assume this script is called from current_directory
# We can deduce that the BoBa repo has the same path as this script file
# boba/ boba_builder.py
#       ...
#       tpl/
#           RAJA
#           Umpire
#           ...
# By default, we clone the tpls and build from that location
# CI will do something different, see below
current_directory = Path(os.getcwd()).resolve()
boba_dir = Path(os.path.dirname(__file__)).resolve()
tpl_clone_base = os.path.join(boba_dir, "tpl")
tpl_source_base = os.path.join(current_directory, "tpl_source")
tpl_build_base = os.path.join(current_directory, "build")
tpl_install_base = os.path.join(current_directory, "install")

safe_mkdir(tpl_build_base)
safe_mkdir(tpl_install_base)

tpls['blt']['enable'] = args.blt
tpls['fmt']['enable'] = args.fmt
tpls['camp']['enable'] = args.camp
tpls['RAJA']['enable'] = args.RAJA
tpls['umpire']['enable'] = args.umpire
tpls['eigen']['enable'] = args.eigen
tpls['Caliper']['enable'] = args.Caliper
tpls['hdf5']['enable'] = args.hdf5
tpls['metal_cpp_compute']['enable'] = args.metal_cpp_compute
tpls['BOBA'] = {}
tpls['BOBA']['enable'] = 'On'

boba_cmake_options = {}
boba_cmake_options['BOBA_ENABLE_EIGEN'] = tpls['eigen']['enable']
boba_cmake_options['BOBA_ENABLE_FMT'] = tpls['fmt']['enable']
boba_cmake_options['BOBA_ENABLE_CALIPER'] = tpls['Caliper']['enable']
boba_cmake_options['BOBA_ENABLE_HDF5'] = tpls['hdf5']['enable']
boba_cmake_options['BOBA_ENABLE_RAJA'] = tpls['RAJA']['enable']
boba_cmake_options['BOBA_ENABLE_UMPIRE'] = tpls['umpire']['enable']
boba_cmake_options['BOBA_ENABLE_METAL'] = tpls['metal_cpp_compute']['enable']
boba_cmake_options['BOBA_ENABLE_CAMP'] = tpls['camp']['enable']

boba_cmake_options['BOBA_ENABLE_MPI'] = args.mpi
boba_cmake_options['BOBA_CHECKPOINTS'] = args.checkpoints
boba_cmake_options['BOBA_DEBUG'] = args.debug
boba_cmake_options['BOBA_ENABLE_EXAMPLES'] = args.examples
boba_cmake_options['BOBA_ENABLE_TESTS'] = args.tests
boba_cmake_options['BOBA_ENABLE_TUTORIALS'] = args.tutorials
boba_cmake_options['ENABLE_OPENMP'] = enable_openmp

#
# CTS 2
#
if machine in ['dane', 'rzwhippet', 'rzhound']:
    recipe = f'{boba_dir}/recipes/boba_build_dane.json'
if system in ['TOSS4']:
    recipe = f'{boba_dir}/recipes/boba_build_dane.json'

#
# CTS 2, GPU
#
if machine in ['matrix']:
    recipe = f'{boba_dir}/recipes/boba_build_matrix.json'

use_alternate_command_runner = False # TODO - this alt runner was needed for a LANL machine we no longer support

#
# ATS 4
#
if machine in ['tuolumne', 'rzadams']:
    recipe = f'{boba_dir}/recipes/boba_build_tuolumne.json'
if system in ['TOSS4_CRAY']:
    recipe = f'{boba_dir}/recipes/boba_build_tuolumne.json'

#
# Mac / fallback Linux
#
if machine in ['macbook', 'linux']:
    recipe = f'{boba_dir}/recipes/boba_build_macbook.json'
    # Not working yet, see https://lc.llnl.gov/gitlab/boba/boba/-/issues/280
    # recipe = f'{boba_dir}/recipes/boba_build_macbook_metal.json'

#
# MSU ICER H200
#
if machine in ['msu_h200']:
    recipe = f'{boba_dir}/recipes/boba_build_icer_hoppers.json'

if args.recipe:
    filename = args.recipe
else:
    filename = recipe

dirpath = Path(filename).resolve()
if dirpath.exists():
    # Open the JSON file
    with open(dirpath, 'r') as file:
        # Load JSON data into a Python dictionary
        tpls_arch = json.load(file)

with open(dirpath, 'r') as file:
    cmake_data = json.load(file)

if 'cmake_path' in cmake_data.keys():
    cmake_path = cmake_data['cmake_path']
else:
    print("Error: Recipe requires cmake_path... (see other recipes for examples or set this value to 'cmake')")
    quit()

cmake_data['libraries']['BOBA']['cmake'].update(boba_cmake_options)

export_suffix = {}
export_suffix["blt"] = ""
export_suffix["Caliper"] = "share/cmake/caliper"
export_suffix["hdf5"] = "cmake"
export_suffix["fmt"] = "lib/cmake/fmt"
export_suffix["camp"] = "lib/cmake/camp"
export_suffix["eigen"] = "share/eigen3/cmake"
export_suffix["RAJA"] = "lib/cmake/raja"
export_suffix["umpire"] = "lib/cmake/umpire"
export_suffix["metal_cpp_compute"] = "lib/cmake/metal_cpp_compute"

tpl_cmake_dirs = {}

check_external_tpl_cmake_dir('blt', args.BLT_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('fmt', args.FMT_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('camp', args.CAMP_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('RAJA', args.RAJA_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('metal_cpp_compute', args.METAL_CPP_COMPUTE_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('umpire', args.UMPIRE_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('eigen', args.EIGEN_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('Caliper', args.CALIPER_DIR, tpl_cmake_dirs)
check_external_tpl_cmake_dir('hdf5', args.HDF5_DIR, tpl_cmake_dirs)

for tpl in tpl_cmake_dirs.keys():
    cmake_data['libraries']['BOBA']['cmake'][f'BOBA_EXTERNAL_{tpl.upper()}'] = 'On'

if 'blt' in tpl_cmake_dirs:
    cmake_data['libraries']['blt']['cmake']['BLT_SOURCE_DIR'] = tpl_cmake_dirs['blt']['tpl_export_string']
if 'fmt' in tpl_cmake_dirs:
    cmake_data['libraries']['fmt']['cmake']['fmt_DIR'] = tpl_cmake_dirs['fmt']['tpl_export_string']
if 'camp' in tpl_cmake_dirs:
    cmake_data['libraries']['camp']['cmake']['camp_DIR'] = tpl_cmake_dirs['camp']['tpl_export_string']
if 'eigen' in tpl_cmake_dirs:
    cmake_data['libraries']['eigen']['cmake']['Eigen3_ROOT'] = tpl_cmake_dirs['eigen']['tpl_export_string']

def build_tpl(cmake_dict, tpl, architecture):

    is_boba = (tpl == 'BOBA')
    force = args.force or is_boba

    enable = False
    reason = 'not relevant for this arch'
    if (tpls[tpl]['enable'] != 'On'):
        enable = False
        reason = 'not enabled'

    if 'cmake' in cmake_dict[tpl].keys():
        if len(cmake_dict[tpl]['cmake']) > 0:
            enable = True
    if enable and (len(cmake_dict[tpl]['cmake']) > 0):
        enable = True
    if not is_boba:
        if tpl in tpl_cmake_dirs.keys():
            enable = False
            tpl_dir_string = tpl_cmake_dirs[tpl]['tpl_dir_string']
            tpl_export_string = tpl_cmake_dirs[tpl]['tpl_export_string']
            reason = f'provided externally as {tpl_dir_string}={tpl_export_string}'
            cmake_data['libraries']['BOBA']['cmake'][tpl_dir_string] = tpl_export_string
            if tpl == 'blt':
                print(f"tpl_dir_string = {tpl_dir_string}")
                cmake_data['libraries']['blt']['cmake']['BLT_SOURCE_DIR'] = tpl_export_string
            if tpl == 'eigen':
                print(f"tpl_dir_string = {tpl_dir_string}")
                cmake_data['libraries']['eigen']['cmake']['Eigen3_ROOT'] = tpl_export_string

    # Enable check
    if not enable:
        print(f"{tpl} will not be installed: {reason}")
        return

    # Define paths
    tpl_build_dir = os.path.join(tpl_build_base, f"{tpl}_{architecture}")

    if is_boba:
        tpl_source_dir = boba_dir
        tpl_install_dir = os.path.join(tpl_install_base, f"{tpl}_{architecture}")
    else:
        submodule_info = ""
        tpl_clone_dir = os.path.join(tpl_clone_base, tpl)
        tpl_source_dir = tpl_clone_dir # os.path.join(tpl_source_base, f"{tpl}_{architecture}", submodule_info)
        tpl_install_dir = os.path.join(tpl_install_base, f"{tpl}_{architecture}", submodule_info)

    if tpl == 'Caliper':
        cali_lib = os.path.join(tpl_install_dir, "lib")
        cal_lib_file = os.path.join(current_directory, f"caliper_lib_path_info_{architecture}")
        with open(cal_lib_file, "w") as file:
            file.write("#!/bin/bash \n")
            file.write("# This file needs to be sourced before Caliper can be used \n")
            append_ld_library_path = "export LD_LIBRARY_PATH={LD_LIBRARY_PATH}"
            file.write(f"{append_ld_library_path}:{cali_lib}\n") # note we manually set CMAKE_INSTALL_LIBDIR
            if machine in ['macbook']:
                append_dyld_library_path = f"export DYLD_LIBRARY_PATH=:$DYLD_LIBRARY_PATH"
                file.write(f"{append_dyld_library_path}:{cali_lib}\n") # note we manually set CMAKE_INSTALL_LIBDIR

    if tpl == 'hdf5':
        lib = os.path.join(tpl_install_dir, "lib")
        cal_lib_file = os.path.join(current_directory, f"hdf5_lib_path_info_{architecture}")
        with open(cal_lib_file, "w") as file:
            file.write("#!/bin/bash \n")
            file.write("# This file needs to be sourced before hdf5 can be used in a miniapp \n")
            append_ld_library_path = "export LD_LIBRARY_PATH={LD_LIBRARY_PATH}"
            file.write(f"{append_ld_library_path}:{lib}\n") # note we manually set CMAKE_INSTALL_LIBDIR

    #
    # Get export
    #
    if is_boba:
        for other_tpls in tpl_cmake_dirs.keys():
            tpl_dir_string = tpl_cmake_dirs[other_tpls]['tpl_dir_string']
            tpl_export_string = tpl_cmake_dirs[other_tpls]['tpl_export_string']
            cmake_dict['BOBA']['cmake'][tpl_dir_string] = tpl_export_string
    else:
        tpl_export = export_suffix[tpl]
        tpl_export_string = os.path.join(tpl_install_dir, tpl_export)
        tpl_cmake_dirs[tpl] = {}
        tpl_cmake_dirs[tpl]['tpl_dir_string'] = get_tpl_dir_string(tpl)
        tpl_cmake_dirs[tpl]['tpl_export_string'] = tpl_export_string

    # Cache/fetch BLT_SOURCE_DIR
    if tpl == 'blt':
        cmake_data['libraries']['blt']['cmake']['BLT_SOURCE_DIR'] = tpl_source_dir
    else:
        if 'BLT_SOURCE_DIR' in cmake_dict[tpl]['cmake'].keys():
            cmake_dict[tpl]['cmake']['BLT_SOURCE_DIR'] = cmake_data['libraries']['blt']['cmake']['BLT_SOURCE_DIR']

    # Cache/fetch fmt_DIR
    if tpl == 'fmt':
        cmake_data['libraries']['fmt']['cmake']['fmt_DIR'] = tpl_install_dir
    else:
        if 'fmt_DIR' in cmake_dict[tpl]['cmake'].keys():
            cmake_dict[tpl]['cmake']['fmt_DIR'] = cmake_data['libraries']['fmt']['cmake']['fmt_DIR']

    # Cache/fetch camp_DIR
    if tpl == 'camp':
        cmake_data['libraries']['camp']['cmake']['camp_DIR'] = tpl_install_dir
    else:
        if 'camp_DIR' in cmake_dict[tpl]['cmake'].keys():
            cmake_dict[tpl]['cmake']['camp_DIR'] = cmake_data['libraries']['camp']['cmake']['camp_DIR']

    # Cache/fetch Eigen3_ROOT
    if tpl == 'eigen':
        cmake_data['libraries']['eigen']['cmake']['Eigen3_ROOT'] = tpl_install_dir
    else:
        if 'Eigen3_ROOT' in cmake_dict[tpl]['cmake'].keys():
            cmake_dict[tpl]['cmake']['Eigen3_ROOT'] = cmake_data['libraries']['eigen']['cmake']['Eigen3_ROOT']

    # Check if this particular tpl version is already installed
    needs_install = True
    print(f"needs_install = {needs_install}")

    if (not needs_install) and not(args.clone == 'Force'):
        print(f"Nothing to do")
        return

    # Check if this particular tpl version has copied the source from the clone dir
    needs_source = True
    print(f"needs_source = {needs_source}")

    # Clone tpl
    phase = 'clone'
    if is_boba:
        # already cloned
        print("Skipping clone")
    elif ((needs_source and (args.clone != 'Off')) or (args.clone == 'Force')):
        os.chdir(boba_dir)
        command_runner(tpl, phase, ['git', 'submodule', 'update', '--init', '--depth=1', tpl_clone_dir], use_alternate_command_runner)
    else:
        print("Skipping clone")

    # Copy to source dir
    if is_boba:
        # nothing to do
        print("Skipping copy")
    elif needs_source and (tpl_source_dir != tpl_clone_dir):
        print(f"Copying cloned repo to source dir.")
        safe_rmdir(tpl_source_dir)
        safe_copy_dir(tpl_clone_dir, tpl_source_dir)
    else:
        print("Skipping source copy")

    if(no_build):
        print("Ending due to no_build")
        return
    if(tpl == 'blt'):
        print("Ending, blt is source-only")
        return

    # Machine-specific cmake commands
    os.chdir(boba_dir)
    cmake_commands = {}

    for (key, value) in cmake_dict[tpl]['cmake'].items():
        cmake_commands[key] = value

    # If nothing to do for this tpl+arch, return
    if not cmake_commands.keys():
        return

    cmake_commands["CMAKE_INSTALL_PREFIX"] = f'{tpl_install_dir}'
    cmake_commands["CMAKE_INSTALL_LIBDIR"] = f'{tpl_install_dir}/lib' # this controls whether export suffix has ../lib or ../lib64

    # Convert to cmake string
    cmake_command = []
    for (key, value) in cmake_commands.items():
        cmake_command.append(f"-D{key}={value}")

    phase = 'configure'
    if phase in tpls[tpl].keys():
        if architecture in tpls[tpl][phase].keys():
            cmake_command.append(f"-C {RAJA_HOSTCONFIG}")

    # Remove existing build
    safe_rmdir(tpl_build_dir)
    safe_rmdir(tpl_install_dir)

    # Build
    safe_mkdir(tpl_build_dir)
    cmake_command.append(f'-B{tpl_build_dir}')

    # Source
    cmake_command.append(f'-S{tpl_source_dir}')

    # Call cmake and make
    phase = 'cmake'
    os.chdir(current_directory)
    command_runner(tpl, phase, [cmake_path] + cmake_command, use_alternate_command_runner)
    os.chdir(tpl_build_dir)
    phase = 'make'

    build_parallelism = '32'
    command_runner(tpl, phase, ['make', '-j', build_parallelism], use_alternate_command_runner)
    os.chdir(tpl_build_dir)
    phase = 'install'
    command_runner(tpl, phase, ['make', 'install', '-j', build_parallelism], use_alternate_command_runner)

for tpl in cmake_data['libraries'].keys():
    if tpl == 'BOBA':
        continue
    if tpls[tpl]['enable'] == 'On':
        tpl_splash(tpl)
        build_tpl(cmake_data['libraries'], tpl, arch)

#
# Intellisense
# This file provides information to clang.  Here, we are providing
# clang with include directory locations.
if not(args.ci):
    os.chdir(boba_dir)
    with open(r".clangd", "w") as f:
        f.write('CompileFlags:\n')
        f.write('  Add:\n')
        f.write('  - "xc++"\n')
        f.write('  - "-std=c++20"\n')
        f.write(f'  - "-I{boba_dir}/include"\n')
        f.write(f'  - "-I{boba_dir}/boba.hpp"\n')
        for tpl in tpl_cmake_dirs.keys():
            tpl_install_dir = os.path.join(tpl_install_base, f"{tpl}_{arch}", "include")
            f.write(f'  - "-I{tpl_install_dir}"\n')

#
# Build BOBA
#
if install_boba:
    tpl_splash("BoBa")
    build_tpl(cmake_data['libraries'], 'BOBA', arch)
