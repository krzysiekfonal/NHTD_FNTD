function param = parameters_bss(Y,Structure_data)

param = struct;
param.size = size(Y); % size of the benchmark

% General parameters for other methods
alphaA = 1e-5;
alphaX = 1e-3;
lambda_reg = 1e-8;
sigma2_W = 1e7;
xi = 1e-4;

% Data-orianted parameters
if length(Structure_data.J) == 1
   J = Structure_data.J; % rank of factorization
   Jt = J;
else
   Jt = Structure_data.J; % 3D
   J = Jt(1);
end
         
% Parameters for geometric methods
Norm_inx = 0; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm
MaxIter = 25; % maximal number of iteration
Tol = 1e-5; % tolerance for residual error
show_comments = 0; % show index           
show_vizualization = 0; % show index           
                      
l_min=20; % minimalny rozmiar liœcia 
J_inner = 5; % zadana liczba promieni ekstremalnych w ka¿dym z liœci 
extreme_rays_leaves = 1; % 1- chcnmf, 2 - simplex
extreme_rays_means = 2 ; % 1- distance-sum, 2 - simplex
inx_X = 1; % 1 - computed X, 0 - no computation of X
 
param.rank = J;
param.rank_tensor = Jt;
param.MaxIter = MaxIter;
param.tol = Tol;
param.show_inx = show_comments; 
param.normalization = Norm_inx; % 0 - no normalization, 1 - L1-norm, 2 - L2-norm

% Geometric methods
param.structure_parameters.J = J; % rank
param.structure_parameters.J_inner = J_inner; % number of extreme rays in each leaf
param.structure_parameters.l_min = l_min; % number of samples in each leaf
param.structure_parameters.show = show_vizualization;
%param.structure_parameters.Aw = [];
param.structure_parameters.extreme_rays_leaves = extreme_rays_leaves;
param.structure_parameters.extreme_rays_means = extreme_rays_means;
param.structure_parameters.inx_X = inx_X;

end
