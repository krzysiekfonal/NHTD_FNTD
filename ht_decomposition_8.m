function [ Xtt ] = ht_decomposition( X, ranks, alg, varargin )

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
S = reshape(X, ranks(1) * ranks(2), prod(ranks(3:8)));
Xtt{9} = lrmf(S, alg, ranks(9), varargin);
X_ = Xtt{9}' * S;
Xtt{9} = reshape(Xtt{9}, ranks(1), ranks(2), ranks(9));

S = reshape(permute(X, [3:4 1:2 5:8]), ranks(3) * ranks(4), prod(ranks(1:2)) * prod(ranks(5:8)));
Xtt{10} = lrmf(S, alg, ranks(10), varargin);
X_ = Xtt{10}' * reshape(X_', ranks(3) * ranks(4), prod(ranks(5:8)) * ranks(9));
Xtt{10} = reshape(Xtt{10}, ranks(3), ranks(4), ranks(10));

S = reshape(permute(X, [5:6 1:4 7:8]), ranks(5) * ranks(6), prod(ranks(1:4)) * prod(ranks(7:8)));
Xtt{11} = lrmf(S, alg, ranks(11), varargin);
X_ = Xtt{11}' * reshape(X_', ranks(5) * ranks(6), prod(ranks(7:8)) * prod(ranks(9:10)));
Xtt{11} = reshape(Xtt{11}, ranks(5), ranks(6), ranks(11));

S = reshape(permute(X, [7:8 1:6]), ranks(7) * ranks(8), prod(ranks(1:6)));
Xtt{12} = lrmf(S, alg, ranks(12), varargin);
X_ = Xtt{12}' * reshape(X_', ranks(7) * ranks(8), prod(ranks(9:11)));
Xtt{12} = reshape(Xtt{12}, ranks(7), ranks(8), ranks(12));

X = reshape(X_', ranks(9), ranks(10), ranks(11), ranks(12));

S = reshape(X, ranks(9) * ranks(10), ranks(11) * ranks(12));
Xtt{13} = lrmf(S, alg, ranks(13), varargin);
X_ = Xtt{13}' * S;
Xtt{13} = reshape(Xtt{13}, ranks(9), ranks(10), ranks(13));

S = reshape(permute(X, [3 4 1 2]), ranks(11) * ranks(12), ranks(9) * ranks(10));
Xtt{14} = lrmf(S, alg, ranks(14), varargin);
X_ = Xtt{14}' * X_';
Xtt{14} = reshape(Xtt{14}, ranks(11), ranks(12), ranks(14));

Xtt{15} = X_';
end

