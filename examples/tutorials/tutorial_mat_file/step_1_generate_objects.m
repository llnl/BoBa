% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


%
% Generate data to be read by BOBA.
%

setup_paths;

d = 5;
n = 3;

% create full vector by uniformly sampling sin function
t = linspace(0.0, 2.0 * pi, n^d);

% convert to 5D tensor
x = reshape(sin(t), n * ones(1, d));

% convert to TT format
x_tt = tt_tensor(x, 1.0e-14);

% create midpoint forward difference matrix
e = ones(n^d, 1);
A = full(spdiags([-e, e], [-1, 1], n^d, n^d));

% convert to TTM format
% - unlike tt_tensor, we do not to reshape the matrix into a (5+5)-D
%   tensor beforehand; we can just specify the row and column dimension
%   factorizations directly in the constructor
A_tt = tt_matrix(A, 1.0e-14, n * ones(1, d), n * ones(1, d));

% extract the cores as cell array
x_tt_cores = core2cell(x_tt);
A_tt_cores = core2cell(A_tt);

% save the cell arrays
% - at present, only one set of cores can be saved per file, and the
%   name of the variable should match the name of the file
save x.mat x;
save x_tt_cores.mat x_tt_cores;
save A_tt_cores.mat A_tt_cores;
