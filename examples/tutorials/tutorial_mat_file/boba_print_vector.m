% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


function [] = boba_print_vector(vector, vector_name)
%BOBA_PRINT_VECTOR Print a vector in boba_print format
%
%   boba_print_vector(vector, vector_name) writes <vector> in
%   zero-indexed boba_print format to <vector_name>.raw file.

fileID = fopen(strcat(vector_name , '.raw'), 'w');

n = length(vector);
fprintf(fileID, '%d 0\n', n);

for i = 1 : n
    fprintf(fileID, '%d %.17e\n', i - 1, vector(i));
end

fclose(fileID);

end
