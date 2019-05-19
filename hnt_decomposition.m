function Xht = hnt_decomposition( G, R, L, alg, varargin )
%HNT_DECOMPOSITION Summary of this function goes here
%   Detailed explanation goes here

Xht = cell(L,1);
dims = size(G);
R = [{dims}, R]; % add 0-level in R-tree (because matlab index from 1
                 % levels need to be shifted, so 2 means original 1, 1
                 % means original 0 i.e. ranks from input tensor, not core
                 % tensor

for l=1:L-1
    Xht{l} = cell(2^(L-l));
    for nl = 1:2^(L-l)
        W = pair_mtx(G, nl);
        [A B] = lrmf(W, alg, ranks(n), varargin);
        Xht{l}{nl} = reshape(A, R{l}{2*nl-1}, R{l}{2*nl}, R{l+1}{nl});
        G = reshape(B, [R{l+1}{nl} R{l+1}{1:nl-1} R{l}{(2*nl+1):((2^L-l+1)}];
    end
    N = ndims(G);
    G = permute(G, [N:1]);
end

end

