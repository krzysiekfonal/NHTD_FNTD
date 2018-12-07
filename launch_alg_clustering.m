function [ results ] = launch_alg_clustering(X, X_orig, k, groups, ranks, alg, varargin)
et=0;
switch (alg)
    %HT ALGS
    case 1
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        tic
        Yht = nht_decomposition(X, ranks, 'als', varargin{1:end});
        et = toc;
        Ycl = ht_projection(Yht, 1);
        Y = ht_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 2
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        tic
        Yht = nht_decomposition(X, ranks, 'hals', varargin{1:end});
        et = toc;
        Ycl = ht_projection(Yht, 1);
        Y = ht_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 3
        tic
        Yht = nht_decomposition(X, ranks, 'xray', varargin{1:end});
        et = toc;
        Ycl = ht_projection(Yht, 4);
        Y = ht_to_tensor(Yht);
    case 4
        tic
        Yht = ht_decomposition(X, ranks, 'left_svd_qr', varargin{1:end});
        et = toc;
        Ycl = ht_projection(Yht, 4);
        Y = ht_to_tensor(Yht);        
        
    %T ALGS
    case 5
        tic
        Yht = ntd_decomposition(X, ranks, 'ntd_als', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);        
    case 6
        tic
        Yht = ntd_decomposition(X, ranks, 'ntd_hals', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);   

    case 8
        tic
        Yht = hosvd_decomposition(X, ranks, 'hosvd', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);        
    
    %FT ALGS
    case 9
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        tic
        Yht = fnt_decomposition(X, ranks, 'als', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 10
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        tic
        Yht = fnt_decomposition(X, ranks, 'hals', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
    case 11
        tic
        Yht = fnt_decomposition(X, ranks, 'xray', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 4);
        Y = tucker_to_tensor(Yht);
    case 12
        X = permute(X, [4 1 2 3]);
        ranks = ranks([4 1 2 3 5 6 7]);
        tic
        Yht = fnt_decomposition(X, ranks, 'svd', varargin{1:end});
        et = toc;
        Ycl = t_projection(Yht, 1);
        Y = tucker_to_tensor(Yht);
        Y = permute(Y, [2 3 4 1]);
end

results = {};
results.et = et;
tic
idx = kmeans(Ycl, k);
results.etcl = toc;
BM = bestMap(groups, idx');
results.acc = length(find(groups == BM))/length(groups);
results.res = calc_res(X_orig, Y);

end

