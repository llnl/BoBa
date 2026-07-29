# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import socket
import subprocess
import os
from os.path import join, exists
from pathlib import Path
from machines import get_machine

def get_cwd():
    return os.getcwd()

####################################################
#  Script writer
####################################################
def write_sbatch_options(file, options):
	for (key, value) in options.items():
		if value is not None:
			file.write(f"#SBATCH --{key}={value}\n")
		else:
			file.write(f"#SBATCH --{key}\n")

def write_sbatch(options):
	sfilename = "run.sbatch"
	with open("./" + options['dirname'] + "/" + sfilename, "w") as f:
		f.write("#!/bin/bash --login \n")
		options['script']["job-name"] = options['dirname']
		write_sbatch_options(f, options['script'])
		for item in options['actions']:
			f.write(item + "\n")
	call_script = "cd ./" + options['dirname'] + " && sbatch run.sbatch"
	return call_script

def write_script(options):
    
	(machine, sys, ignore) = get_machine()
    
	if 'TOSS4_CRAY' in sys:
		return write_sbatch(options)
	if 'TOSS4' in sys:
		return write_sbatch(options)

	print('write_script - unsupported machine! ' + machine)
	quit()

def write_and_submit_job(options):
  current_directory = get_cwd()
  full_dir = join(current_directory, options["dirname"])
  os.makedirs(full_dir, exist_ok = True)
  call_script = write_script(options)
  subprocess.check_call(call_script, shell = True)
