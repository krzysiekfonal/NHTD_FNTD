function [U,res,info_legend] = bss_fun(Y,algorithm,param)

U = cell([ndims(Y) 1]);
res = [];
J = param.rank;
info_legend = struct;

%% Algorithms
switch algorithm.clustering
    
    case 1 % k-means
        
             
    case 2 % PCA
        
         
    
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
         [U{1},X,res] = nmf_mue(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         U{2} = X';
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
         [U{1},X,res] = nmf_fast_hals(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         U{2} = X';
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
         Y_1 = norm_fun(Y_1,param.normalization,1); % normalization        
                          
         tic        
         [U{1},K] = SimplexMax_method(Y_1,J);
         X = fast_hals_inner(Y_1(:,K),Y_1,rand(J,size(Y_1,2)),1e-6,300);          
         U{2} = X';
         U{1} = (norm_fun(U{1},param.normalization,1)); % normalization
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
         [U{1},X,~] = SNPA(Y_1,J,0);
         U{2} = X';
         U{1} = (norm_fun(U{1},param.normalization,1)); % normalization
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
         [U{1},X,~] = SPA(Y_1,J,1);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = Xray(Y_1,J,1);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = Xray(Y_1,J,3);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = Xray(Y_1,J,2);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = Xray(Y_1,J,4);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = CNMF(Y_1,J);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X] = chnmf(Y_1,J,0);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X,~] = hchnmf(Y_1,J,0,param.structure_parameters.l_min);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization
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
         [U{1},X] = nmf_micro_model_means_new(Y_1,param);
         U{2} = X';
         U{1} = norm_fun(U{1},param.normalization,1); % normalization        
         info_legend(algorithm.clustering).clustering = 'MM';  
         info_legend(algorithm.clustering).time = toc;                        
     
    case 19 % HO-SM
        
        tic
        if length(param.size) < 3
           disp('Number of modes is too small - no dimensionality reduction is used');
        else
           Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization      
           [A_final_1,K_final] = SimplexMax_n_way(Y,J);
           A_final = Y(K_final,:,:);
           A_final_mtx = double((tenmat(A_final,1)))';
           U = cp_sm_2D_special_SM(Y,A_final_mtx);
           B = kr(U{3},U{2});
             
           % Macierz koduj¹ca X
           if param.structure_parameters.inx_X
              Ym = double(tenmat(Y,1))';
            %  [X,G,iter] = fast_hals_inner(B,Ym,rand(J,size(Ym,2)),1e-12,1000);
               X = max(0,B\Ym);
           end
           U{1} = X';
           
        %    U{1} = (norm_fun(U{1},param.normalization,1)); % normalization
            info_legend(algorithm.clustering).clustering = 'HO-SM';  
            info_legend(algorithm.clustering).time = toc; 
        end
             
    case 20 % HO-SVD
        
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
             return
          else
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             Tr = hosvd_n_way(Y,param.rank_tensor);
             U{1} = norm_fun(Tr.U{1},param.normalization,0); % normalization 
             for n = 2:length(param.size)
                 U{n} = Tr.U{n}'
             end
             info_legend(algorithm.clustering).clustering = 'HOSVD';  
             info_legend(algorithm.clustering).time = toc;                   
          end
             
    case 21 % HO-XRAY
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
                                       
             opts.MaxIter = 20; % in FHALS-CPD
             opts.model = 2;          
             opts.selection_i = 2; % 1- rand, 2 - max, 3 - dist
             
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             [U,Kx] = XRAY_CPD(Y,param.rank,opts);          
           
             U{2} = norm_fun(U{2},param.normalization,0); % normalization  
             info_legend(algorithm.clustering).clustering = 'HO-XRAY';  
             info_legend(algorithm.clustering).time = toc;          
              
          end          
          
    case 22 % HO-MM
 
          tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Y = max(0,Y); 
             Y = norm_fun_n_way(tensor(Y),param.normalization); % normalization   
             U = micro_model_means_n_way(Y,param.structure_parameters);         
             U{2} = norm_fun(U{2},param.normalization,1); % normalization  
             info_legend(algorithm.clustering).clustering = 'HO-MM';  
             info_legend(algorithm.clustering).time = toc;                
              
          end
          
    case 23 % HALS-NTF
 
         tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Y = max(0,Y); 
             Y = norm_fun_n_way(Y,param.normalization); % normalization   
             [U,res] = fast_hals(Y, J, param.MaxIter);
          end
          info_legend(algorithm.clustering).clustering = 'HALS-NTF';  
          info_legend(algorithm.clustering).time = toc;   
          
      case 24  % R-HALS-NTF
 
         tic
          if length(param.size) < 3
             disp('Number of modes is too small - no dimensionality reduction is used');
          else
             Y = max(0,Y); 
             Y = norm_fun_n_way(Y,param.normalization); % normalization   
             [U,res] = fast_hals_r(Y, J, param.MaxIter);
          end
          info_legend(algorithm.clustering).clustering = 'R-HALS-NTF';  
          info_legend(algorithm.clustering).time = toc;             
                                 
      case 25 % RK
 
        if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [U{1},X,res] = nmf_rk(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         U{2} = X';
         info_legend(algorithm.clustering).clustering = 'RK';  
         info_legend(algorithm.clustering).time = toc;         
          
    case 26 % RK-ADMM
 
        if length(param.size) > 2
            DimY = size(Y); 
            Y_1 = reshape(Y,[DimY(1) prod(DimY)/DimY(1)]); % unfolding
         else
            Y_1 = Y; 
         end
         Y_1 = norm_fun(Y_1,param.normalization,0); % normalization        
                          
         tic
         [U{1},X,res] = nmf_rk_admm(Y_1,param.Ainit,param.Xinit,1e-12,param.tol,param.MaxIter,param.show_inx);
         U{2} = X';
         info_legend(algorithm.clustering).clustering = 'RK-ADMM';  
         info_legend(algorithm.clustering).time = toc;         
          
        
          
end % switch    



end