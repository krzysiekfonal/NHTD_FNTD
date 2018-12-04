function [y_test_hat,info_legend] = classify_fun(xtrain,ytrain,xtest,algorithm,param)

global info_legend

y_test_hat = [];
J = param.rank;

%% Dimensionality Reduction
switch algorithm.DimRed
    
    case 1 % no dimensionality reduction
        
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'no';  
         
    case 2 % PCA
        
         tic
         xs_train = bsxfun(@minus,xtrain,mean(xtrain,1));
         xs_test = bsxfun(@minus,xtest,mean(xtrain,1));
       
         C=(xs_train'*xs_train)/(size(xs_train,1)-1);
         [Vselect,D]=eigs(C,J);
                  
         xtrain = xs_train*Vselect;
         xtest = xs_test*Vselect;
         
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'PCA';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);  
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
    
    case 3 % MUE        
        
         tic
         Ainit = rand(size(xtrain,1),J); Xinit = rand(J,size(xtrain,2));     
         [xtrain,F_train,res] = nmf_mue(xtrain,Ainit,Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         Xt = rand(J,size(xtest,1));
         xtest = (mue(F_train',xtest',Xt,300))';
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'MUE';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);       
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
         
    case 4 % HALS        
        
         tic
         Ainit = rand(size(xtrain,1),J); Xinit = rand(J,size(xtrain,2));     
         [xtrain,F_train,res] = nmf_fast_hals(xtrain,Ainit,Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         xtest = fast_hals_A(xtest,F_train,rand(size(xtest,1),J),300);       
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HALS';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          
    case 5 % GNMF
        
          tic
          Ainit = rand(size(xtrain,2),J); Xinit = rand(J,size(xtrain,1));     
         [L,W,W1,W2] = fun_correlation(xtrain',ytrain,param.structure_parameters);    
         [Ar,Xr,res] = gnmf(xtrain',Ainit,Xinit,W,param.structure_parameters.gamma,1e-12,param.tol,param.MaxIter,param.show_inx);
         
          Xt = rand(J,size(xtest,1));
          Xt = mue(Ar,xtest',Xt,300);
          xtrain = Xr';
          xtest = Xt';          
          
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'GNMF';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);          
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
         
    case 6 % MD2
           
          tic
          Ainit = rand(size(xtrain,2),J); Xinit = rand(J,size(xtrain,1));  
         [L,W,W1,W2] = fun_correlation(xtrain',ytrain,param.structure_parameters);    
         [Ar,Xr,res] = md2_nmf(xtrain',Ainit,Xinit,W1,W2,...
             param.structure_parameters.alphaA,param.structure_parameters.alphaX,param.structure_parameters.gamma,...
             1e-12,param.tol,param.MaxIter,param.show_inx);
         
          Xt = rand(J,size(xtest,1));
          Xt = mue(Ar,xtest',Xt,300);
          xtrain = Xr';
          xtest = Xt';          
          
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'MD2';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
         
    case 7 % SPG
        
          tic
          Ainit = rand(size(xtrain,2),J); Xinit = rand(J,size(xtrain,1));  
         [L,W,W1,W2] = fun_correlation(xtrain',ytrain,param.structure_parameters);    
         [Ar,Xr,res] = nmf_spg_reg(xtrain',Ainit,Xinit,L,1e-12,param.tol,param.MaxIter,param.show_inx);
          
          Xt = rand(J,size(xtest,1));
          Xt = spg_method(Ar,xtest',Xt,300);
          xtrain = Xr';
          xtest = Xt';          
          
         info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'SPG';  
         info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);
         info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
         
    case 8 % SimplexMax
        
          tic
         [F_train,K] = SimplexMax_method(xtrain',J);
          xtrain = fast_hals_A(xtrain,F_train',rand(size(xtrain,1),J),300);        
          xtest = fast_hals_A(xtest,F_train',rand(size(xtest,1),J),300);  
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'SimplexMax'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
    
    case 9 % SNPA
        
          tic
          [W_train,H_train,~] = SNPA(xtrain',J,1);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'SNPA'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 10 % SPA
        
          tic
          [W_train,H_train,~,~] = SPA(xtrain',J,1);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'SPA'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 11 % XRAY (rand)
        
          tic
          [W_train,H_train,~] = Xray(xtrain',J,1);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'XRAY(rand)'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 12 % XRAY (dist)
        
          tic
          [W_train,H_train,~] = Xray(xtrain',J,3);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'XRAY(D)'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 13 % XRAY (max)
        
          tic
          [W_train,H_train,~] = Xray(xtrain',J,2);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'XRAY(max)'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 14 % XRAY-Greedy
        
          tic
          [W_train,H_train,~] = Xray(xtrain',J,4);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'XRAY(G)'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 15 % CNMF
        
          tic
          [W_train,H_train,~] = CNMF(xtrain',J);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'CNMF'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 16 % CHNMF
        
          tic
          [W_train,H_train] = chnmf(xtrain',J,0);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'CHNMF'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
        
    case 17 % HCHNMF
        
          tic
          [W_train,H_train,~] = hchnmf(xtrain',J,0,param.structure_parameters.l_min);
          xtrain = H_train';
          xtest = fast_hals_A(xtest,W_train',rand(size(xtest,1),J),300);            
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HCHNMF'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);      
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc; 
        
    case 18 % Micromodel means
        
          tic
          [F_train,X_train] = nmf_micro_model_means(xtrain',param.structure_parameters);
          xtrain = X_train';
          xtest = fast_hals_A(xtest,F_train',rand(size(xtest,1),size(xtrain,2)),300);  
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'MicroModels'; 
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);          
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
     
    case 19 % Micromodel classes (FM,m-mean)
        
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          param.structure_parameters.clustering = 1; % FastMap
          param.structure_parameters.leaf = 1; % modified mean
          
          [xtrain, xtest] = nmf_micro_model_class(xtrain',ytrain',xtest',param);
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'C-MM(1)';      
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);    
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
             
    case 20 % Micromodel classes (FM,mean)
        
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          param.structure_parameters.clustering = 1; % FastMap
          param.structure_parameters.leaf = 2; % mean
          
          [xtrain, xtest] = nmf_micro_model_class(xtrain',ytrain',xtest',param);
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'C-MM(2)';      
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);    
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
 
    case 21 % Micromodel classes (FM,median)
        
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          param.structure_parameters.clustering = 1; % FastMap
          param.structure_parameters.leaf = 3; % median
          
          [xtrain, xtest] = nmf_micro_model_class(xtrain',ytrain',xtest',param);
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'C-MM(3)';      
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);    
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
                    
    case 22 % Micromodel classes (k-means)
        
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          param.structure_parameters.clustering = 2; % FastMap
          param.structure_parameters.leaf = 1; % modified mean
          
          [xtrain, xtest] = nmf_micro_model_class(xtrain',ytrain',xtest',param);
          info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'C-MM(4)';      
          info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);    
          info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
          
    case 23 % HO-SVD
        
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = tensor(reshape(xtrain,[size(xtrain,1) param.size(2:end)]));
             Tr = hosvd_n_way(Yr,param.rank_tensor);
             DimYt = size(xtest);
             Yt_1 = reshape(xtest,[DimYt(1) param.size(2:end)]);
          %   G_1 = reshape(Tr.core,[param.rank_tensor(1),prod(param.rank_tensor)/param.rank_tensor(1)]);
          %   xtest = xtest*pinv(double(G_1)*kron(Tr.U{3},Tr.U{2})');
               
             AtA = cellfun(@(x)  x'*x, Tr.U,'uni',0);
             Tt = tensor(Yt_1);
              
             YtAn = tenmat(ttm(Tt,Tr.U,-1,'t'),1); 
             Gn = tenmat(Tr.core,1); % unfolding along n-mode
             YtAnG = YtAn * Gn';
             GtAn = tenmat(full(ttm(Tr.core,AtA,-1)),1);
             B = Gn * GtAn';
             xtest = double(YtAnG)/double(B);
             xtrain = Tr.U{1};

             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HO-SVD';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);   
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end
             
    case 24 % HO-XRAY
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = tensor(reshape(xtrain,[size(xtrain,1) param.size(2:end)]));
                          
             opts.MaxIter = 20; % in FHALS-CPD
             opts.model = 1;          
             opts.selection_i = 2; % 1- rand, 2 - max, 3 - dist
             [U,Kx] = XRAY_CPD(Yr,param.rank,opts);          
           
             xtest = xtest/double(tenmat(U{1},1));
             xtrain = U{2};

             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HO-XRAY';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);                   
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end          
          
    case 25 % HO-MM
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = tensor(reshape(xtrain,[size(xtrain,1) param.size(2:end)]));    
             [F_train,X_train] = micro_model_means_n_way(Yr,param.structure_parameters);
             xtrain = X_train';
             xtest = fast_hals_A(xtest,F_train',rand(size(xtest,1),size(xtrain,2)),300);  
          
             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HO-MM';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);     
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end
          
    case 26 % HO-CMM
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = tensor(reshape(xtrain,[size(xtrain,1) param.size(2:end)]));    
             Yt = tensor(reshape(xtest,[size(xtest,1) param.size(2:end)]));    
                          
             [xtrain, xtest] = micro_model_class_n_way(Yr,ytrain,Yt,param.structure_parameters);
             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HO-CMM';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);     
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end 
          
    case 27 % HO-CMM-SVD
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = tensor(reshape(xtrain,[size(xtrain,1) param.size(2:end)]));    
             Yt = tensor(reshape(xtest,[size(xtest,1) param.size(2:end)]));    
                          
             [xtrain, xtest, ytrain] = micro_model_class_n_way_hosvd(Yr,ytrain,Yt,param);
             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HO-CMM-SVD';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);     
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end    
          
    case 28 % HALS-NTF
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = reshape(xtrain,[size(xtrain,1) param.size(2:end)]);    
             [U,res] = fast_hals(Yr, param.structure_parameters.J, param.MaxIter);
             xtrain = U{1};
             F_train = kr(U{4},kr(U{3},U{2}));    
           %  F_train = kr(U{3},U{2});         
             xtest = fast_hals_A(xtest,F_train',rand(size(xtest,1),size(xtrain,2)),300);  
          
             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'HALS-NTF';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);     
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end    
          
    case 29 % HALS-NTF
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Yr = reshape(xtrain,[size(xtrain,1) param.size(2:end)]);    
             [U,res] = fast_hals_r(Yr, param.structure_parameters.J, param.MaxIter);
             xtrain = U{1};
         %    F_train = kr(U{3},U{2});         
             F_train = kr(U{4},kr(U{3},U{2}));                      
             xtest = fast_hals_A(xtest,F_train',rand(size(xtest,1),size(xtrain,2)),300);  
          
             info_legend(algorithm.DimRed,algorithm.Classifier).DimRed = 'R-HALS-NTF';      
             info_legend(algorithm.DimRed,algorithm.Classifier).rank = size(xtrain,2);     
             info_legend(algorithm.DimRed,algorithm.Classifier).time_DimRed = toc;
              
          end        
          
end % switch    


%% Classifier
switch algorithm.Classifier
    
    case 1 % KNN-E
        
        tic
        Mdl = fitcknn(xtrain,ytrain,'NumNeighbors',1);
        y_test_hat = predict(Mdl, xtest);
        info_legend(algorithm.DimRed,algorithm.Classifier).time_classifier = toc;
        info_legend(algorithm.DimRed,algorithm.Classifier).classifier = 'KNN-E';  
    
    case 2 % KNN-C
        
        tic
        Mdl = fitcknn(xtrain,ytrain,'NumNeighbors',1,'Distance','cosine');
        y_test_hat = predict(Mdl, xtest);
        info_legend(algorithm.DimRed,algorithm.Classifier).time_classifier = toc;
        info_legend(algorithm.DimRed,algorithm.Classifier).classifier = 'KNN-C';            
            
    case 3 % SVM
        
        tic
        y_test_hat = mSVM_fun(xtrain',ytrain,xtest');
        info_legend(algorithm.DimRed,algorithm.Classifier).time_classifier = toc;
        info_legend(algorithm.DimRed,algorithm.Classifier).classifier = 'SVM';               
        
    case 4 % NB
        
       % Md2 = fitcnb(xtrain,ytrain);
       % y_test_hat = predict(Md3, xtest);        
        tic
        y_test_hat = bayess_fun(xtrain',ytrain,xtest');
        info_legend(algorithm.DimRed,algorithm.Classifier).time_classifier = toc;
        info_legend(algorithm.DimRed,algorithm.Classifier).classifier = 'NB';       
        
    case 5 % LDA
    
        tic
        y_test_hat = classify(xtest,xtrain,ytrain);
        info_legend(algorithm.DimRed,algorithm.Classifier).time_classifier = toc;
        info_legend(algorithm.DimRed,algorithm.Classifier).classifier = 'LDA';       
        
end % switch classifier


end