function [ Xtt ] = ht_decomposition( X, ranks, alg, varargin )

ranks = cell2mat([ranks{:}]);
Xtt = cell(size(ranks,1),1);
dim = size(X);
N = ndims(X);
X_ = X;

% compute matrices of leaves
for n = 1:N
    W = reshape(permute(double(X),[n [1:n-1,n+1:N]]),dim(n), prod(dim)/dim(n));
    Xtt{n} = lrmf(W, alg, ranks(n), varargin);
    X_ = tensor_contraction(X_, Xtt{n}, 1, 1);
end
X = X_;
% TODO make it mor felxible - currently implemented for 4 dims only
S = reshape(double(X), ranks(1) * ranks(2), ranks(3) * ranks(4));
Xtt{5} = lrmf(S, alg, ranks(5), varargin);
X_ = Xtt{5}' * S;
Xtt{5} = reshape(Xtt{5}, ranks(1), ranks(2), ranks(5));

S = reshape(permute(double(X), [3 4 1 2]), ranks(3) * ranks(4), ranks(1) * ranks(2));
Xtt{6} = lrmf(S, alg, ranks(6), varargin);
X_ = Xtt{6}' * permute(X_, [2 1]);
Xtt{6} = reshape(Xtt{6}, ranks(3), ranks(4), ranks(6));

Xtt{7} = permute(X_, [2 1]);
end

