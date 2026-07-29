# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import numpy as np
import subprocess
from matplotlib import pyplot as plt

#
# How to run this - (do this from the top level dir in BoBa)
# python3 examples/exercises/example_implicit_block/run_and_plot.py
#

resolution = 200
dt = 0.001
timesteps = resolution

execute = "make example_implicit_block -j 2 && ./example_implicit_block*.out "
parameters = f" --resolution {resolution}"
parameters += f" --timesteps {timesteps}"
parameters += f" --timestep {dt}"

files = ["exact_solution.dump", "tt_solution.dump", "full_solution.dump"]
for file in files:
   if os.path.exists(file):
      os.remove(file)

os.system(execute + parameters)

################################################################################

data = np.loadtxt("tt_solution.dump")
solution = np.reshape(data, [resolution, resolution])

data = np.loadtxt("exact_solution.dump")
exact_solution = np.reshape(data, [resolution, resolution])

data = np.loadtxt("full_solution.dump")
full_solution = np.reshape(data, [resolution, resolution])

################################################################################
def error_bound(error):
   return np.amax(np.abs(error))

fig, axs = plt.subplots(2, 3)
fig.set_figheight(8)
fig.set_figwidth(15)

fig.suptitle('Implicit Block Advection Example')

plot = axs[0, 0].pcolormesh(solution, vmin=0.0, vmax=1.0)
plt.colorbar(plot, ax=axs[0, 0])
axs[0, 0].set_title("TT Solution")

plot = axs[0, 1].pcolormesh(full_solution, vmin=0.0, vmax=1.0)
plt.colorbar(plot, ax=axs[0, 1])
axs[0, 1].set_title("Full solution")

plot = axs[0, 2].pcolormesh(exact_solution)
plt.colorbar(plot, ax=axs[0, 2])
axs[0, 2].set_title("Exact solution")

error = solution - exact_solution
bound = error_bound(error)
plot = axs[1, 0].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 0])
axs[1, 0].set_title("Tensor train vs. exact")

error  = full_solution - exact_solution
bound = error_bound(error)
plot = axs[1, 1].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 1])
axs[1, 1].set_title("Full vs. exact (Discretization error)")

error  = solution - full_solution
bound = error_bound(error)
plot = axs[1, 2].pcolormesh(error, vmin=-bound, vmax=bound)
plt.colorbar(plot, ax=axs[1, 2])
axs[1, 2].set_title("Tensor train vs. Full (Estimation error)")

plt.show()
