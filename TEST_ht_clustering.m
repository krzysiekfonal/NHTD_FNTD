clear
benchmark = 1; %1 - coil_100, 2 - film4
% 1 - NHTD(als), 2 - NHTD(hals), 3 - NHTD(xray), 4 - HTD(left_svd_qr)
% 5 - NTD(als), 6 - NTD(hals), 7 - NTD(xray), 8 - HO-SVD
% 9 - FNTD(als), 10 - FNTD(hals), 11 - FNTD(xray), 12 - FTD(svd)
algs = [1 2 3 4 5 6 8 9 10 11 12];
MC = 5;
n_algs = size(algs, 2);
SNR = 0;

%% setup test params
tol = 1e-4;
maxiters = 30;

%% prepare data
% Xht = generate_ht_tensors(dims, ranks);
% X_ = ht_to_tensor(Xht);
if benchmark == 1
    load('data/coil_100.mat')
    X_ = Y;
    %groups = groups(1:101);
    ranks = [6 6 3 50 6 6 1];
    k = 99;
elseif benchmark == 2
    load('data/film4.mat')
    X_ = X0;
    ranks = [6 8 3 50 6 6 1];
    k = 3;
    groups = inx;
end

if SNR ~= 0
    Nt = randn(size(X_)); 
    tau = (norm(X_(:),'fro')/norm(Nt(:), 'fro'))*10^(-SNR/20);
    X = X_ + tau*Nt;
    else
    X = X_;
end    

%% launch tests
results = cell(n_algs + 1,1);
for i=1:n_algs
    results{i} = {};
    results{i}.res = zeros(MC,1);
    results{i}.et = zeros(MC,1);
    results{i}.acc = zeros(MC,1);
    results{i}.alg = algs(i);
end

for mc=1:MC
    for i=1:n_algs
        r = launch_alg_clustering(X, X_, k, groups, ranks, algs(i),...
                                 'tol', tol, 'maxiters', maxiters);
        results{i}.res(mc) = r.res;
        results{i}.acc(mc) = r.acc;
        results{i}.et(mc) = r.et;
    end
    %launch kmeans for original data
    dims = size(X);
    Ycl = permute(reshape(X, dims(1)*dims(2)*dims(3), dims(4)), [2 1]);
    results{n_algs + 1}.res(mc) = 0;
    tic
    idx = kmeans(Ycl, k);
    results{n_algs + 1}.et(mc) = toc;
    BM = bestMap(groups, idx');
    results{n_algs + 1}.acc(mc) = length(find(groups == BM))/length(groups);
end

for i=1:n_algs
    results{i}.res_mean = mean(results{i}.res);
    results{i}.acc_mean = mean(results{i}.acc);
    results{i}.et_mean = mean(results{i}.et);
    results{i}.etcl_mean = mean(results{i}.etcl);
    results{i}.res_std = std(results{i}.res);
    results{i}.acc_std = std(results{i}.acc);
    results{i}.et_std = std(results{i}.et);
    results{i}.etcl_std = std(results{i}.etcl);
end
