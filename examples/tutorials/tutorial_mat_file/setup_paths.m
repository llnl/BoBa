% SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


if ~exist('tt_toolbox_root', 'var')
    error('variable `tt_toolbox_root` must point to the root of tt-toolbox code');
end

addpath(tt_toolbox_root);
addpath(fullfile(tt_toolbox_root, 'core'));
