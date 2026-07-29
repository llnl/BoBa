# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import numpy as np
import subprocess
from matplotlib import pyplot as plt

################################################################################
# Compile and run the code
################################################################################

numsteps = 120
num1dpts = 200
dt = 0.005

make = "make clean && make example_implicit -j "
os.system(make)

execute = "SOLVER=bicgstab FILE_DUMP=true ./example_implicit*.out "
parameters = f" --dimension 2"
parameters += f" --boundary-conditions 0"
parameters += f" --advection 1.0"
parameters += f" --diffusion 0.0"
parameters += f" --resolution {num1dpts}"
parameters += f" --timesteps {numsteps}"
parameters += f" --timestep {dt}"

files = ["initial_value_unrolled.raw", "solution_unrolled.raw", "exact_tensor.raw", "conventional_solution.raw"]
for file in files:
   if os.path.exists(file):
      os.remove(file)

os.system(execute + parameters)

################################################################################
# crawl through log.out file and grab printed slices
################################################################################

data = np.loadtxt("dump_initial_value_unrolled.raw")
initial_condition = np.reshape(data[1:, 2], [num1dpts, num1dpts])

data = np.loadtxt("dump_solution_unrolled.raw")
solution = np.reshape(data[1:, 2], [num1dpts, num1dpts])

data = np.loadtxt("dump_exact_tensor.raw")
exact_solution = np.reshape(data[1:, 2], [num1dpts, num1dpts])

data = np.loadtxt("dump_conventional_solution.raw")
conventional_solution = np.reshape(data[1:, 2], [num1dpts, num1dpts])

################################################################################
################################################################################
def error_bound(error):
   return np.amax(np.abs(error))

fig, axs = plt.subplots(2, 3)
fig.set_figheight(8)
fig.set_figwidth(15)

fig.suptitle('Implicit Advection Example')

plot = axs[0, 0].pcolormesh(solution, vmin=0.0, vmax=1.0)
plt.colorbar(plot, ax=axs[0, 0])
axs[0, 0].set_title("TT Solution")

plot = axs[0, 1].pcolormesh(conventional_solution, vmin=0.0, vmax=1.0)
plt.colorbar(plot, ax=axs[0, 1])
axs[0, 1].set_title("Conventional solution")

plot = axs[0, 2].pcolormesh(exact_solution)
plt.colorbar(plot, ax=axs[0, 2])
axs[0, 2].set_title("Exact solution")

error = solution - exact_solution
bound = error_bound(error)
plot = axs[1, 0].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 0])
axs[1, 0].set_title("Tensor train vs. exact")

error  = conventional_solution - exact_solution
bound = error_bound(error)
plot = axs[1, 1].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 1])
axs[1, 1].set_title("Conventional vs. exact (Discretization error)")

error  = solution - conventional_solution
bound = error_bound(error)
plot = axs[1, 2].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 2])
axs[1, 2].set_title("Tensor train vs. conventional (Estimation error)")

plt.show()
