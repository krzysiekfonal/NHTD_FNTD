function [ results ] = launch_alg_clustering(X, X_orig, k, groups, ranks, alg, varargin)

switch (alg)
    %HT ALGS
    case 1
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        Yht = nht_decomposition(X, ranks, 'als', varargin{1:end});
        Ycl = ht_projection(Yht, 1);
        Y = ht_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 2
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        Yht = nht_decomposition(X, ranks, 'hals', varargin{1:end});
        Ycl = ht_projection(Yht, 1);
        Y = ht_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 3
        Yht = nht_decomposition(X, ranks, 'xray', varargin{1:end});
        Ycl = ht_projection(Yht, 4);
        Y = ht_to_tensor(Yht);
    case 4
        Yht = ht_decomposition(X, ranks, 'left_svd_qr', varargin{1:end});
        Ycl = ht_projection(Yht, 4);
        Y = ht_to_tensor(Yht);        
        
    %T ALGS
    case 5
        Yht = ntd_decomposition(X, ranks, 'ntd_als', varargin{1:end});
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);        
    case 6
        Yht = ntd_decomposition(X, ranks, 'ntd_hals', varargin{1:end});
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);   

    case 8
        Yht = hosvd_decomposition(X, ranks, 'hosvd', varargin{1:end});
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);        
    
    %FT ALGS
    case 9
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        Yht = fnt_decomposition(X, ranks, 'als', varargin{1:end});
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 10
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        Yht = fnt_decomposition(X, ranks, 'hals', varargin{1:end});
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 11
        Yht = fnt_decomposition(X, ranks, 'xray', varargin{1:end});
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);
    case 12
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        Yht = fnt_decomposition(X, ranks, 'svd', varargin{1:end});
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
end

results = {};

tic
idx = kmeans(Ycl, k);
results.et = toc;
BM = bestMap(groups, idx');
results.acc = length(find(groups == BM))/length(groups);
results.acc = 0;
results.res = calc_res(X_orig, Y);

end

