function [ Xtt ] = ntd_decomposition( X, ranks, alg, varargin)

N = ndims(X);
Xtt = cell(N+1, 1);

% compute matrices of leaves
switch alg
    case 'ntd_als'
        T = ntd_als(tensor(X), ranks(1:N), varargin);    
    case 'ntd_hals'
        T = ntd_hals_phan(tensor(X), ranks(1:N), varargin);
end

for n = 1:N
    Xtt{n} = T.U{n};
end
Xtt{N+1} = double(T.core);

end

