function [ Xtt ] = nht_decomposition_8( X, ranks, alg, varargin )

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
X = permute(X,8:-1:1);
% TODO make it mor felxible - currently implemented for 4 dims only
S = reshape(double(X), ranks(1) * ranks(2), prod(ranks(3:8)));
[Xtt{9}, B] = lrmf(S, alg, ranks(9), varargin);
Xtt{9} = reshape(Xtt{9}, ranks(1), ranks(2), ranks(9));

S = reshape(B', ranks(3) * ranks(4), prod(ranks(5:8)) * ranks(9));
[Xtt{10}, B] = lrmf(S, alg, ranks(10), varargin);
Xtt{10} = reshape(Xtt{10}, ranks(3), ranks(4), ranks(10));

S = reshape(B', ranks(5) * ranks(6), prod(ranks(7:8)) * prod(ranks(9:10)));
[Xtt{11}, B] = lrmf(S, alg, ranks(11), varargin);
Xtt{11} = reshape(Xtt{11}, ranks(5), ranks(6), ranks(11));

S = reshape(B', ranks(7) * ranks(8), prod(ranks(9:11)));
[Xtt{12}, B] = lrmf(S, alg, ranks(12), varargin);
Xtt{12} = reshape(Xtt{12}, ranks(7), ranks(8), ranks(12));

S = reshape(B', ranks(9) * ranks(10), ranks(11) * ranks(12));
[Xtt{13}, B] = lrmf(S, alg, ranks(13), varargin);
Xtt{13} = reshape(Xtt{13}, ranks(9), ranks(10), ranks(13));

S = B';
[Xtt{14}, B] = lrmf(S, alg, ranks(14), varargin);
Xtt{14} = reshape(Xtt{14}, ranks(11), ranks(12), ranks(14));

Xtt{15} = B';

end

