function param = parameters_clustering(Y,groups,benchmark)

C = length(unique(groups)); % true number of clusters

param = struct;
param.C = C; % true number of clusters
param.size = size(Y); % size of the benchmark

% General parameters for other methods
alphaA = 1e-5;
alphaX = 1e-3;
lambda_reg = 1e-8;
sigma2_W = 1e7;
xi = 1e-4;

% Data-orianted parameters
switch benchmark
    
       case 1 % MNIST
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
    
       case 2 % Movie (IWANN 2017)
            
            J = C; % rank of factorization
            Jt = [C 20 20 3]; % 4D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index       
            
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X     
            
       case 3 % Smartphone datasets
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 4 % DriveFace
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X     
             
       case 5 % ORL facial images
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X    
            
       case 6 % Emotions-Wavelets
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X    
            
       case 7 % Digits (student)
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X         
                        
       case 8 % Textures
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X        
            
       case 9 % Musical instruments
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-5; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 10 % TDT
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 100; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=100; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X   
           
       case 11 % Citeseer
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X    
            
       case 12 % Cora
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X     
            
       case 13 % Wiki
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 14 % Reuters
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 15 % 20NewsHome
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 16 % TDT2_all
        
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
           
       case 17 % RCV1_4Class
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
           
       case 18 % tae
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 19 % seeds
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 20 % heart
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 21 % wpbc
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 22 % wine
           
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 23 % coil20
               
            J = C; % rank of factorization
            Jt = [C 13 13]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=50; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 24 % cifar10
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 25 % cifar100
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 26 % umist
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 27 % alphadigits
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 28 % usps
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 29 % NORB
            
            J = C; % rank of factorization
            Jt = [C 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=30; % minimalny rozmiar liœcia 
            J_inner = 10; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 2; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X   
    
end % benchmark

param.rank = J;
param.MaxIter = MaxIter;
param.tol = Tol;
param.show_inx = show_comments; 
param.rank_tensor = Jt;
param.normalization = Norm_inx; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm

% Geometric methods
param.structure_parameters.J = J; % rank
param.structure_parameters.J_inner = J_inner; % number of extreme rays in each leaf
param.structure_parameters.l_min = l_min; % number of samples in each leaf
param.structure_parameters.show = show_vizualization;
param.structure_parameters.Aw = [];
param.structure_parameters.extreme_rays_leaves = extreme_rays_leaves;
param.structure_parameters.extreme_rays_means = extreme_rays_means;
param.structure_parameters.inx_X = inx_X;

end
