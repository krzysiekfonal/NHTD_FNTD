function [ Xtt ] = nht_decomposition( X, ranks, alg, varargin )

Xtt = cell(size(ranks,2),1);
dim = size(X);
N = ndims(X);

% compute matrices of leaves
for n = 1:N
    W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
    [Xtt{n}, B] = lrmf(W, alg, ranks(n), varargin);
    X = reshape(B, [ranks(n:-1:1) dim(n+1:N)]);
    dim(n) = ranks(n);
end
X = permute(X,[4 3 2 1]);
% TODO make it mor felxible - currently implemented for 4 dims only
S = reshape(double(X), ranks(1) * ranks(2), ranks(3) * ranks(4));
[Xtt{5}, B] = lrmf(S, alg, ranks(5), varargin);
Xtt{5} = reshape(Xtt{5}, ranks(1), ranks(2), ranks(5));

S = permute(B, [2 1]);
[Xtt{6}, B] = lrmf(S, alg, ranks(6), varargin);
Xtt{6} = reshape(Xtt{6}, ranks(3), ranks(4), ranks(6));

Xtt{7} = permute(B, [2 1]);
end

