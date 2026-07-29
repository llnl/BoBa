% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

% create the MAT files
step_1_generate_objects;

%
% Example 1: load a tensor
%

% Load the file
file_struct = load('x.mat');

% In the file there will be a tensor called "x"
extracted_tensor = file_struct.x;

% Save the file using boba_print
boba_print_tensor(extracted_tensor, 'x');

%
% Example 2: load a tensor train
%

% Load the file
file_struct_tt = load('A_tt_cores.mat');

% In the file there will be a cell array called "A_tt_cores"
extracted_cores = file_struct_tt.A_tt_cores;

% Save the cores one by one to files using boba_print
for d = 1 : numel(extracted_cores)
    boba_print_tensor(extracted_cores{d}, strcat('A_tt_core_', num2str(d - 1)));
end
