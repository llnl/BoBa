% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


%
% Verify MATLAB and BOBA computed values match up.
%

% repeat data generation from step 1
setup_paths;

d = 5;
n = 3;

t = linspace(0.0, 2.0 * pi, n^d);

x = reshape(sin(t), n * ones(1, d));

x_tt = tt_tensor(x, 1.0e-14);

e = ones(n^d, 1);
A = full(spdiags([-e, e], [-1, 1], n^d, n^d));

A_tt = tt_matrix(A, 1.0e-14, n * ones(1, d), n * ones(1, d));

% also create the negative Laplacian operator and the z vector from step 2
sizes = [4, 5, 8, 4];

N = prod(sizes);
delta = 1.0 / (N + 1);

e = ones(N, 1) / (delta * delta);
B = full(spdiags([-e, 2 * e, -e], [-1, 0, 1], N, N));
B_tt_mat = tt_matrix(B, 1.0e-14, sizes, sizes);

t = sin(2.0 * pi * (1 : N)' * delta);
z_mat = B_tt_mat * t;

% read y = A * x computed in BOBA
load y_tt_cores.mat;
y_tt = cell2core(tt_tensor, y_tt_cores);

fprintf('error in y is %e\n', norm(y_tt - A_tt * x_tt));

% read negative Laplacian B computed in BOBA
load B_tt_cores.mat;
B_tt = cell2core(tt_matrix, B_tt_cores);

fprintf('error in B is %e\n', norm(B_tt - B_tt_mat));

% read the z vector in BOBA
load z.mat

fprintf('error in z is %e\n', norm(z - z_mat));
