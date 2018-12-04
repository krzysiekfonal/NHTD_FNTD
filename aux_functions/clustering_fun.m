function [groups_estim,info_legend] = clustering_fun(Y,algorithm,param)

groups_estim = [];
J = param.rank;
info_legend = struct;

%% Algorithms
switch algorithm.clustering
    
    case 1 % k-means
        
         % unfolding
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding 
         else
            Y_1 = Y; 
         end
               
         tic
         groups_estim = kmeans(Y_1,J); 
         info_legend(algorithm.clustering).clustering = 'kmeans';  
         info_legend(algorithm.clustering).time = toc;
             
    case 2 % PCA
        
         tic
         xs_train = bsxfun(@minus,xtrain,mean(xtrain,1));
         xs_test = bsxfun(@minus,xtest,mean(xtrain,1));
       
         C=(xs_train'*xs_train)/(size(xs_train,1)-1);
         [Vselect,D]=eigs(C,J);
                  
         xtrain = xs_train*Vselect;
         xtest = xs_test*Vselect;
         
    
    case 3 % MUE        
        
         % unfolding
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
        
         tic
         [A,X,res] = nmf_mue(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
          A = norm_fun(A,param.normalization,0); % normalization
         [~,groups_estim] = max(A,[],2);
         info_legend(algorithm.clustering).clustering = 'MUE';  
         info_legend(algorithm.clustering).time = toc;
         
    case 4 % HALS        
                
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
        [A,X,res] = nmf_fast_hals(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         A = norm_fun(A,param.normalization,0); % normalization
         [~,groups_estim] = max(A,[],2);
         info_legend(algorithm.clustering).clustering = 'HALS';  
         info_legend(algorithm.clustering).time = toc;         
         
   %      info_legend(algorithm.clustering).clustering 
          
    case 5 % GNMF
        

         
    case 6 % MD2
           
         
         
    case 7 % SPG
        
         
         
    case 8 % SimplexMax
                
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic        
         [A,K] = SimplexMax_method(Y_1',J);
          X = fast_hals_inner(Y_1(K,:)',Y_1',rand(J,size(Y_1,1)),1e-6,300);          
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'SM';  
         info_legend(algorithm.clustering).time = toc;            
             
    
    case 9 % SNPA
                
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = SNPA(Y_1',J,0);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'SNPA';  
         info_legend(algorithm.clustering).time = toc;            
        
    case 10 % SPA
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = SPA(Y_1',J,1);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'SPA';  
         info_legend(algorithm.clustering).time = toc;                    
          
    case 11 % XRAY (rand)
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = Xray(Y_1',J,1);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'Xray(rand)';  
         info_legend(algorithm.clustering).time = toc;                    
      
        
    case 12 % XRAY (dist)
        
          if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = Xray(Y_1',J,3);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'Xray(dist)';  
         info_legend(algorithm.clustering).time = toc;                        
     
        
    case 13 % XRAY (max)
        
          if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = Xray(Y_1',J,2);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'Xray(max)';  
         info_legend(algorithm.clustering).time = toc;                       
      
        
    case 14 % XRAY-Greedy
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = Xray(Y_1',J,4);
         X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'Xray(G)';  
         info_legend(algorithm.clustering).time = toc;                        
         
        
    case 15 % CNMF
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = CNMF(Y_1',J);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'CNMF';  
         info_legend(algorithm.clustering).time = toc;                         
                 
    case 16 % CHNMF
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X] = chnmf(Y_1',J,0);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'CHMF';  
         info_legend(algorithm.clustering).time = toc;           
      
    case 17 % HCHNMF
                
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [A,X,~] = hchnmf(Y_1',J,0,param.structure_parameters.l_min);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);
         info_legend(algorithm.clustering).clustering = 'HCHNMF';  
         info_legend(algorithm.clustering).time = toc;                
        
    case 18 % Micromodel means
        
         if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,1); % normalization        
                          
         tic
         [A,X] = nmf_micro_model_means(Y_1',param);
          X = norm_fun(X,param.normalization,1); % normalization
         [~,groups_estim] = max(X,[],1);               
         info_legend(algorithm.clustering).clustering = 'MM';  
         info_legend(algorithm.clustering).time = toc;                        
     
    case 19 % Micromodel classes
        
          
             
    case 20 % HO-SVD
        
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
             return
          else
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             Tr = hosvd_n_way(Y,param.rank_tensor);
             A = norm_fun(Tr.U{1},param.normalization,0); % normalization  
             [~,groups_estim] = max(A,[],2);
             info_legend(algorithm.clustering).clustering = 'HOSVD';  
             info_legend(algorithm.clustering).time = toc;                   
          end
             
    case 21 % HO-XRAY
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
                                       
             opts.MaxIter = 20; % in FHALS-CPD
             opts.model = 1;          
             opts.selection_i = 2; % 1- rand, 2 - max, 3 - dist
             
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             [U,Kx] = XRAY_CPD(Y,param.rank,opts);          
           
             A = norm_fun(U{2},param.normalization,0); % normalization  
             [~,groups_estim] = max(A,[],2);
             info_legend(algorithm.clustering).clustering = 'HO-XRAY';  
             info_legend(algorithm.clustering).time = toc;          
              
          end          
          
    case 22 % HO-MM
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             [A,X] = micro_model_means_n_way(Y,param.structure_parameters);         
              X = norm_fun(X,param.normalization,1); % normalization  
             [~,groups_estim] = max(X,[],1); 
             info_legend(algorithm.clustering).clustering = 'HO-MM';  
             info_legend(algorithm.clustering).time = toc;                
              
          end
          
    case 23 % HO-CMM
 
         
        
          
    case 24 % HO-CMM-SVD
 
         
          
end % switch    



end