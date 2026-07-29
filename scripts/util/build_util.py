# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import socket
import subprocess
import os
from os.path import join, exists
from shutil import rmtree, copytree, copyfile
from pathlib import Path
import sys
from subprocess import check_call
import platform

def safe_mkdir(path):
    os.umask(0o007)
    Path(path).mkdir(parents=True, mode=0o750, exist_ok=True)

def safe_rmdir(path):
    dirpath = Path(path).resolve()
    if dirpath.exists() and dirpath.is_dir():
        print(f"Removing {dirpath}")
        rmtree(dirpath)

def safe_copy_dir(src, dst):
    os.umask(0o007)
    print(f"Copying file {src} to {dst}")
    copytree(src, dst)

def command_runner(tpl, part, run_list, use_alternate_command_runner):

    run = " ".join(run_list)

    processing = f"Processing {tpl} {part}: {run}"
    print(processing)

    if use_alternate_command_runner:
        output = subprocess.run(
        run,
        shell=True,
        universal_newlines=False)#, #text=True,
        #capture_output=True)
    else:
        output = subprocess.run(
        run,
        shell=True,
        text=True,
        capture_output=True)

    if not (output.returncode == 0):
      # Test failed, print output
      print(run)
      print(output.stdout)
      print(output.stderr)
      print(f"{tpl} {part} failed")
      Exception(f"{tpl} {part} failed")
      sys.exit(1)
    else:
      print(f"{tpl} {part} success")

def tpl_splash(tpl):
    print("--------------------------")
    print(f" {tpl} ")
    print("--------------------------")

def get_tpl_dir_string(tpl):
    return tpl.upper() + "_DIR"

def check_external_tpl_cmake_dir(tpl, arg, tpl_cmake_dirs):
    if arg is None:
        return

    abs_path = os.path.abspath(arg)
    if not os.path.exists(abs_path):
        print(f"Given {get_tpl_dir_string(tpl)} does not exist: {abs_path}")
        sys.exit(1)

    if not os.path.isdir(abs_path):
        print(f"Given {get_tpl_dir_string(tpl)} is not a directory: {abs_path}")
        sys.exit(1)

    tpl_cmake_dirs[tpl] = {}
    tpl_cmake_dirs[tpl]['tpl_dir_string'] = get_tpl_dir_string(tpl)
    tpl_cmake_dirs[tpl]['tpl_export_string'] = abs_path
