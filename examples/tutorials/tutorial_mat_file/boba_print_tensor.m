% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


function [] = boba_print_tensor(tensor, tensor_name)
%BOBA_PRINT_TENSOR Print a tensor in boba_print format
%
%   boba_print_tensor(tensor, tensor_name) writes <tensor> in
%   zero-indexed boba_print format to <tensor_name>.raw file.

fileID = fopen(strcat(tensor_name , '.raw'), 'w');

sizes = size(tensor);
fprintf(fileID, '%d ', sizes);
fprintf(fileID, '0\n');

numelem = numel(tensor);
for i = 1 : numelem
    mid = multiindex(i, sizes);
    fprintf(fileID, '%d ', mid);
    fprintf(fileID, '%.17e\n', tensor(i));
end

fclose(fileID);

end

function mid = multiindex(lid, sizes)

ndim = numel(sizes);
mid = zeros(1, ndim);

lid_copy = lid - 1;
for d = 1 : ndim
  mid(d) = mod(lid_copy, sizes(d));
  lid_copy = floor(lid_copy / sizes(d));
end

end
