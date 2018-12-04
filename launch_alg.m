function [ results ] = launch_alg( MC, X, X_orig, U, ranks, alg, varargin)

% init results
N = ndims(X);
results.res = 0;
results.sir = zeros(1, N);
results.msir = 0;
results.et = 0;
results.alg = alg;

for i=1:MC
    tic
    switch (alg)
        %HT ALGS
        case 1
            Yht = nht_decomposition(X, ranks, 'als', varargin{1:end});
            Y = ht_to_tensor(Yht);
        case 2
            Yht = nht_decomposition(X, ranks, 'hals', varargin{1:end});
            Y = ht_to_tensor(Yht);
        case 3
            Yht = nht_decomposition(X, ranks, 'xray', varargin{1:end});
            Y = ht_to_tensor(Yht);
        case 4
            Yht = ht_decomposition(X, ranks, 'left_svd_qr', varargin{1:end});
            Y = ht_to_tensor(Yht);        
        case 5
            Yht = ntd_decomposition(X, ranks, 'ntd_als', varargin{1:end});
            Y = tucker_to_tensor(Yht);        
        case 6
            Yht = ntd_decomposition(X, ranks, 'ntd_hals', varargin{1:end});
            Y = tucker_to_tensor(Yht);   
        
        case 8
            Yht = hosvd_decomposition(X, ranks, 'hosvd', varargin{1:end});
            Y = tucker_to_tensor(Yht);        
        case 9
            Yht = fnt_decomposition(X, ranks, 'als', varargin{1:end});
            Y = tucker_to_tensor(Yht);
        case 10
            Yht = fnt_decomposition(X, ranks, 'hals', varargin{1:end});
            Y = tucker_to_tensor(Yht);
        case 11
            Yht = fnt_decomposition(X, ranks, 'xray', varargin{1:end});
            Y = tucker_to_tensor(Yht);
        case 12
            Yht = fnt_decomposition(X, ranks, 'svd', varargin{1:end});
            Y = tucker_to_tensor(Yht);
    end
    % calculate results
    results.et = results.et + toc;
    results.res = results.res + calc_res(X_orig, Y);
    if ~isempty(U)
        SIR = zeros(1,N);
        for n=1:N
            SIR(n) = mean(CalcSIR(U{i}, Yht{i}));
            results.sir(n) = results.sir(n) + SIR(n);
        end
        results.msir = results.msir + mean(SIR(1:4));
    end
end
results.res = results.res / MC;
for n=1:N
    results.sir(n) = results.sir(n) / MC;
end
results.msir = results.msir / MC;
results.et = results.et / MC;

end

