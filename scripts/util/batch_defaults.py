# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from machines import get_machine

(machine, sys, arch) = get_machine()

####################################################
#  Script defaults
####################################################
# https://hpc.llnl.gov/documentation/tutorials/using-lc-s-sierra-systems#Running

# slurm defaults
sops = {}
sops["nodes"] = "1"
sops["ntasks"] = "1"
sops["cpus-per-task"] = "1"
sops["ntasks-per-node"] = "1"
sops["output"] = "slurm.results"
sops["time"] = "2:00:00"
sops["open-mode"] = "truncate"

# Flux
fops = {}
fops["nodes"] = "1"

####################################################
#  Defaults
####################################################
if ('tuolumne' in machine) or ('rzadams' in machine):
    script_options  = sops
if ('dane' in machine) or ('rzhound' in machine) or ('rzwhippet' in machine):
    script_options  = sops

options = {}
options["script"] = script_options
options["actions"] = []
    