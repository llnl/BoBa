# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from pathlib import Path
import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time

import yaml

#
# This script is useful for running all the tests defined by ci.yaml.
# It can be launched from BoBa itself or from a downstream project that uses
# BoBa's Makefile naming conventions.
#

boba_root = Path(__file__).resolve().parent
sys.path.append(f"{boba_root}/scripts/util")

from machines import get_machine

def prRed(skk): return ("\033[91m{}\033[00m" .format(skk))
def prGreen(skk): return ("\033[92m{}\033[00m" .format(skk))
def prYellow(skk): return ("\033[93m{}\033[00m" .format(skk))
def prCyan(skk): return ("\033[96m{}\033[00m" .format(skk))

def colored(sring, color):
  if 'red' in color:
    return prRed(sring)
  if 'green' in color:
    return prGreen(sring)
  if 'yellow' in color:
    return prYellow(sring)
  if 'cyan' in color:
    return prCyan(sring)
  return sring

parser = argparse.ArgumentParser()

parser.add_argument("--verbose", help="Increase output verbosity",
                    action="store_true")

parser.add_argument("--umpire", help="Test with Umpire on",
                    action="store_true")

parser.add_argument("--debug", help="Test debug mode",
                    action="store_true")

parser.add_argument("--asan", help="Test address sanitizer mode",
                    action="store_true")

parser.add_argument("--ubsan", help="Test undefined behavior sanitizer mode",
                    action="store_true")

parser.add_argument("--repo-root", help="Directory containing ci.yaml and test executables. Defaults to the current working directory.",
                    default=os.getcwd())

parser.add_argument("--keep-binaries", help="Do not remove tested binaries and objects.",
                    action="store_true")

parser.add_argument("--tests", help="Space- or comma-separated top-level tests to run from ci.yaml.",
                    default="")

args = parser.parse_args()

repo_root = Path(args.repo_root).resolve()
yaml_file = repo_root / "ci.yaml"
with open(yaml_file, encoding="utf-8") as stream:
  tests = yaml.safe_load(stream) or {}

selected_tests = {test for test in args.tests.replace(",", " ").split() if test}

_, _, space = get_machine()

def get_name_flags(space):
  if 'cpu' in space:
    return '_cpu'
  if 'cuda_h100' in space:
    return '_cuda_h100'
  if 'cuda' in space:
    return '_cuda'
  if 'hip' in space:
    return '_hip'
  print('unsupported space! ' + space, flush=True)
  sys.exit(1)

EXE_FLAGS = ""
NAME_FLAGS = ""
if args.asan:
  NAME_FLAGS += "_asan"
if args.ubsan:
  NAME_FLAGS += "_ubsan"
NAME_FLAGS += get_name_flags(space)
if args.umpire:
  NAME_FLAGS += "_ump"
if args.debug:
  NAME_FLAGS += "_debug"

def get_run_command(test):
  env_prefixes = []
  if args.asan:
    env_prefixes.append(("ASAN_OPTIONS", "print_stacktrace=1:print_suppressions=0"))
    env_prefixes.append(("LSAN_OPTIONS", f"suppressions={boba_root / 'util' / 'lsan.supp'}:print_stacktrace=1:report_error_type=1"))
  if args.ubsan:
    env_prefixes.append(("UBSAN_OPTIONS", f"suppressions={boba_root / 'util' / 'ubsan.supp'}:print_stacktrace=1:report_error_type=1"))
  argv = shlex.split(EXE_FLAGS)
  test_exe = repo_root / f"{test}{NAME_FLAGS}.out"
  argv.append(str(test_exe))
  return env_prefixes, argv, test_exe

def test_time_str(time):
  if time < 10.0:
    color_str = "green"
  elif time < 60.0:
    color_str = "yellow"
  else:
    color_str = "red"

  return colored(f"{time:.1f} sec", color_str)

g_asan_re = r"AddressSanitizer|LeakSanitizer"
g_ubsan_re = r"UndefinedBehaviorSanitizer"

def normalize_skip_tags(skip_value):
  if skip_value is None:
    return set()
  if isinstance(skip_value, list):
    return {str(tag).strip() for tag in skip_value if str(tag).strip()}
  return {tag for tag in shlex.split(str(skip_value)) if tag}

def should_skip(skip_value):
  skip_tags = normalize_skip_tags(skip_value)
  if not skip_tags:
    return False

  active_tags = {space}
  if "cuda" in space:
    active_tags.add("cuda")
  if "hip" in space:
    active_tags.add("hip")
  if args.asan:
    active_tags.add("asan")
  if args.ubsan:
    active_tags.add("ubsan")
  if args.umpire:
    active_tags.add("ump")
  if args.debug:
    active_tags.add("debug")

  return bool(skip_tags & active_tags)

def expand_env_value(value, env):
  previous = os.environ.copy()
  try:
    os.environ.clear()
    os.environ.update(env)
    return os.path.expandvars(value)
  finally:
    os.environ.clear()
    os.environ.update(previous)

def run_single_test(test_name, subtest_name, subtest, run_list):

  env_prefixes, argv = run_list
  argv = argv.copy()
  env = os.environ.copy()
  for key, value in env_prefixes:
    env[key] = value
  inputs = []
  for (key, value) in subtest.items():
    if 'environment' in key:
      for assignment in shlex.split(str(value)):
        key, sep, val = assignment.partition("=")
        if not sep:
          raise Exception(f'Invalid environment assignment "{assignment}" in {test_name}/{subtest_name}')
        env[key] = expand_env_value(val, env)
    elif 'inputs' in key:
      inputs = shlex.split(str(value))
    elif not ('skip' in key):
      raise Exception(f'Unrecognized test variable "{key}", set to: {value}')

  if args.asan:
    env["FAIL_NEVER"] = "1"
  if args.ubsan:
    env["FAIL_NEVER"] = "1"

  print(colored(f"  Running: {test_name}/{subtest_name}", 'cyan'), flush=True)

  argv.extend(inputs)
  if not Path(argv[0]).exists():
    raise FileNotFoundError(f"Missing executable {argv[0]}. Run `make {test_name}` or `make all` first.")

  run = " ".join(
      [f"{key}={value}" for (key, value) in env_prefixes] +
      [shlex.quote(arg) for arg in argv]
  )

  fout = tempfile.TemporaryFile()
  ferr = tempfile.TemporaryFile()

  test_process = subprocess.Popen(argv, cwd=repo_root, env=env, stdout=fout, stderr=ferr)
  return (test_process, run, fout, ferr, test_name, subtest_name)

def run_tests():

  tests_passing = True

  for (top_level_test_name, test) in tests.items():

    if selected_tests and top_level_test_name not in selected_tests:
      continue

    running_tests = []

    env_prefixes, _argv, test_exe = get_run_command(top_level_test_name)
    print(colored(f"Running: {test_exe.name}", 'cyan'), flush=True)

    for (subtest_name, subtest) in test.items():

      if "skip" in subtest and should_skip(subtest["skip"]):
        print(colored(f"  Skipping: {top_level_test_name}/{subtest_name}", 'cyan'), flush=True)
        continue

      run_list = (env_prefixes, _argv)
      try:
        (test_process, run, fout, ferr, executed_test_name, subtest_name) = run_single_test(top_level_test_name, subtest_name, subtest, run_list)
        running_tests.append((test_process, run, fout, ferr, executed_test_name, subtest_name))
      except Exception as exc:
        print(colored(f"\n  Test setup failed: {top_level_test_name}/{subtest_name}", 'red'), flush=True)
        print(str(exc), flush=True)
        tests_passing = False

    for test_process, run, fout, ferr, test_name, subtest_name in running_tests:

      start = time.time()
      returncode = test_process.wait()
      end = time.time()

      time_str = test_time_str(end - start)

      fout.seek(0)
      str_fout = fout.read().decode("utf-8")
      ferr.seek(0)
      str_ferr = ferr.read().decode("utf-8")
      fout.close()
      ferr.close()

      test_failure = not (returncode == 0)

      if args.asan:
        regex_hits = re.search(g_asan_re, str_ferr)
        test_failure = test_failure or (regex_hits is not None)
      if args.ubsan:
        regex_hits = re.search(g_ubsan_re, str_ferr)
        test_failure = test_failure or (regex_hits is not None)

      if test_failure:
        # Test failed, print output
        print(colored(f"\n  Test failed: {test_name}/{subtest_name}", 'red'), flush=True)
        print(f"\n\n run = \n{run}", flush=True)
        print(f"\n\n returncode = {returncode}", flush=True)
        print(f"\n\n fout = \n{str_fout}", flush=True)
        print(f"\n\n ferr = \n{str_ferr}", flush=True)
        tests_passing = False
      else:
        # Test passed, don't print output unless in verbose mode
        passed_str = colored(f"  Test passed: {test_name}/{subtest_name}", 'green')
        # Time the amount of time it took to wait for that test
        # note that this is a lower bound for the test runtime
        print(f"{passed_str}, process wait time : {time_str}", flush=True)
        if(args.verbose):
          if str_fout:
            print(f"\nstdout:\n{str_fout}", flush=True)
          if str_ferr:
            print(f"\nstderr:\n{str_ferr}", flush=True)

    #
    # rm test once complete
    #
    if not args.keep_binaries:
      if test_exe.exists():
        print(colored(f"  Removing {test_exe}", 'cyan'), flush=True)
        test_exe.unlink()

      object_path = repo_root / f"{top_level_test_name}{NAME_FLAGS}.o"
      if object_path.exists():
        print(colored(f"  Removing {object_path}", 'cyan'), flush=True)
        object_path.unlink()

  return tests_passing

tests_passing = run_tests()

if not tests_passing:
  sys.exit(1)
