clear
benchmark = 3; %1 - test data, 2 - coil_100, 3 - film4
% 1 - NHTD(als), 2 - NHTD(hals), 3 - NHTD(xray), 4 - HTD(left_svd_qr)
% 5 - NTD(als), 6 - NTD(hals), 7 - NTD(xray), 8 - HO-SVD
% 9 - FNTD(als), 10 - FNTD(hals), 11 - FNTD(xray), 12 - FTD(svd)
algs = [1 2 3 4 5 6 8 9 10 11 12];
SNR = 0;
MC = 1;
n_algs = size(algs, 2);
results = cell(n_algs, 1);

%% setup test params
tol = 1e-4;
maxiters = 30;

%% prepare data
% Xht = generate_ht_tensors(dims, ranks);
% X_ = ht_to_tensor(Xht);

if benchmark == 1
    % dims ranks for this benchmark
    dims = [ 20 30 40 60];
    ranks = [3 4 4 5 3 3 1];
    % Generate core tensors and factor matirces
    Core = max(0,randn(ranks(1:4)));
    U = cell(1,4);
    U{1} = max(0,randn(dims(1), ranks(1)));
    U{2} = max(0,randn(dims(2), ranks(2)));
    U{3} = max(0,randn(dims(3), ranks(3)));
    U{4} = max(0,randn(dims(4), ranks(4)));
    X_ = tensor_contraction(...
        tensor_contraction(...
        tensor_contraction(...
        tensor_contraction(Core, U{1}, 1, 2),...
        U{2}, 1, 2),...
        U{3}, 1, 2),...
        U{4}, 1, 2);
elseif benchmark == 2
    U = [];
    load('data/coil_100.mat')
    X_ = Y;
    ranks = [6 6 3 50 6 6 1];
elseif benchmark == 3
    U = [];
    load('data/film4.mat')
    X_ = X0;
    ranks = [6 8 3 50 6 6 1];
end

N = size(ranks,2);

if SNR ~= 0
    Nt = randn(size(X_)); 
    tau = (norm(X_(:),'fro')/norm(Nt(:), 'fro'))*10^(-SNR/20);
    X = X_ + tau*Nt;
else
    X = X_;
end    


%% launch tests
for i=1:n_algs
    results{i} = launch_alg(MC, X, X_, U, ranks, algs(i),...
        'tol', tol, 'maxiters', maxiters);
    results{i}.SNR = SNR;
end
