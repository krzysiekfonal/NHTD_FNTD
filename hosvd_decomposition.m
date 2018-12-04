function [ Xtt ] = hosvd_decomposition( X, ranks, alg, varargin)

% Parameters
X = tensor(X);
dim = size(X);
N = ndims(X);
Xtt = cell(N+1, 1);
U = cell(N,1);
core = X;
T = [];

% Main loop
for n = 1:N
    % Eigendecomposition
    U{n} = nvecs(X,n,ranks(n));
    core = ttm(core, U{n}',n);
end
T = ttensor(core, U);

for n = 1:N
    Xtt{n} = T.U{n};
end
Xtt{N+1} = double(T.core);


end

