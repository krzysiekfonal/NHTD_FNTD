function param = parameters_classification(Y,groups,benchmark)

C = length(unique(groups)); % true number of clusters

param = struct;
param.C = C; % true number of clusters
param.size = size(Y); % size of the benchmark

% Data-orianted parameters
switch benchmark
    
       case 1 % MNIST
           
            J = 25; % rank of factorization
            Jt = [J 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 20; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
    
       case 2 % Movie (IWANN 2017)
            
            J = 30; % rank of factorization
            Jt = [J 10 10 3]; % 4D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 40; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index       
            
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X     
            
       case 3 % Smartphone datasets
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 25; % rank of factorization
            Jt = [J 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 40; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X    
            
       case 6 % Emotions-Wavelets
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [10 20 20]; % 3D
         
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
           
            J = 12; % rank of factorization
            Jt = [10 20 20]; % 3D
         
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
            
       case 10 % TDT
           
            J = 10; % rank of factorization
            Jt = [10 20 20]; % 3D
         
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
           
       case 11 % Citeseer
           
            J = 10; % rank of factorization
            Jt = [10 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 100; % maximal number of iteration
            Tol = 1e-8; % tolerance for residual error
            show_comments = 1; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 16 % TDT2_all
        
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 100; % maximal number of iteration
            Tol = 1e-8; % tolerance for residual error
            show_comments = 1; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
           
       case 18 % tae
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % basic rank of factorization
            Jt = [J 20 20]; % 3D
         
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
           
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
               
            J = 75; % rank of factorization
            Jt = [J 13 13]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 200; % maximal number of iteration
            Tol = 1e-4; % tolerance for residual error
            show_comments = 0; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 5; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 24 % cifar10
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
            Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
            MaxIter = 100; % maximal number of iteration
            Tol = 1e-8; % tolerance for residual error
            show_comments = 1; % show index           
            show_vizualization = 0; % show index           
                      
            l_min=10; % minimalny rozmiar liœcia 
            J_inner = 3; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
            extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
            extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
            inx_X = 1; % 1 - computed X, 0 - no computation of X
            
       case 25 % cifar100
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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
            
            J = 10; % rank of factorization
            Jt = [J 20 20]; % 3D
         
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

% Parameters for discriminat methods
gamma_DNMF = 1e-5;
delta_DNMF = 1e-7;
gamma = 1e-6; % GNMF

alphaA = 1e-5;
alphaX = 1e-3;
lambda_reg = 1e-8;

k_NN = 5;
k_NN_w = 8;
k_NN_b = 20;
sigma2_W = 1e7;
xi = 1e-4;

% 1- correlation, 2 - binary kNN, 3 - Euclidean distance, 4 - cosine
% 5 - binary kNN L1 L2, 6 - Euclidean kNN L1 L2
correlation = 3; 

% 0 - L, 1 - DNMF, 2 - eig
combination = 0;

% Geometric methods
param.structure_parameters.J = J; % rank
param.structure_parameters.J_inner = J_inner; % number of extreme rays in each leaf
param.structure_parameters.l_min = l_min; % number of samples in each leaf
param.structure_parameters.show = show_vizualization;
param.structure_parameters.Aw = [];
param.structure_parameters.extreme_rays_leaves = extreme_rays_leaves;
param.structure_parameters.extreme_rays_means = extreme_rays_means;
param.structure_parameters.inx_X = inx_X;

% Discriminant methods
param.structure_parameters.correlation = correlation;
param.structure_parameters.combination = combination;
param.structure_parameters.gamma = gamma;
param.structure_parameters.gamma_DNMF = gamma_DNMF;
param.structure_parameters.delta_DNMF = delta_DNMF;
param.structure_parameters.alphaA = alphaA;
param.structure_parameters.alphaX = alphaX;
param.structure_parameters.lambda_reg = lambda_reg;
param.structure_parameters.k_NN = k_NN;
param.structure_parameters.k_NN_w = k_NN_w;
param.structure_parameters.k_NN_b = k_NN_b;
param.structure_parameters.sigma2_W = sigma2_W;
param.structure_parameters.xi = xi;

end
