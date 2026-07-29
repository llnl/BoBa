# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np
import glob

# BoBa file format
# See print_file in Tensor.hpp

def read_boba_file(filename):

    data = np.loadtxt(filename)
    (rows, cols) = np.shape(data)
    dimensions = cols - 1
    size = rows - 1
    sizes = data[0, 0:dimensions].astype(int)

    assert np.prod(sizes) == size, "File information is not self-consistent"

    index = data[1:, 0:dimensions].astype(int)
    tensor_data = data[1:, dimensions]
    tensor = np.zeros(sizes)

    for i in range(size):
        mid = tuple(index[i, :dimensions])
        value = tensor_data[i]
        tensor[mid] = value

    return tensor
