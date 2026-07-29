# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import numpy as np
import subprocess
from matplotlib import pyplot as plt

################################################################################
# Compile and run the code
################################################################################
num1dpts = 300
endtime = 2.0
execute = "make example_explicit -j 10 && FILE_DUMP=true ./example_explicit*.out "
parameters = f" -n {num1dpts} -c 1.0 -d 0.0 -s 0.0 -D 2 -T {endtime}"

files = ["solution_unrolled.raw", "solution_full.raw", "initial_value_unrolled.raw"]
for file in files:
   if os.path.exists(file):
      os.remove(file)

os.system(execute + parameters)

################################################################################
# Collect solution data
################################################################################

data = np.loadtxt("initial_value_unrolled.raw")
initial_condition = np.reshape(data[:, 2], [num1dpts, num1dpts])

data = np.loadtxt("solution_unrolled.raw")
solution = np.reshape(data[:, 2], [num1dpts, num1dpts])

data = np.loadtxt("solution_full.raw")
conventional_solution = np.reshape(data[:, 2], [num1dpts, num1dpts])

################################################################################
# Plots
################################################################################

plt.figure()
plt.pcolormesh(initial_condition, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.title("Initial Condition")

plt.figure()
plt.pcolormesh(solution, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.title("Solution")

plt.figure()
plt.pcolormesh(conventional_solution, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.title("Conventional solution")

error  = solution - conventional_solution
plt.figure()
plt.pcolormesh(error)
plt.colorbar()
plt.title("Difference")

plt.show()
