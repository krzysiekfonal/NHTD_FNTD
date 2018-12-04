% Main script for testing classification algorithms
clear all; close 

% Paths
addpath('Classifiers');
addpath('CHNMF');
addpath('NMF');
addpath('NTF');
addpath('snpa_v3');

% Paths to data
addpath('movies');
addpath('MNIST');
addpath('Textual');
addpath('Smartphone');
addpath('DriveFace');
addpath('Various_datasets');

global info_legend

%% Data uploading
% 1 - MNIST, 2- movie (IWANN 2017), 3 - Smartphones, 4 - DriveFaces, 
% 5 - ORL, 6 - Emotions-wavelets, 7 - digits (student)
% 8 - Textures, 9 - Musical instruments
% 10 - TDT, 11 - Citeseer, 12 - Cora, 13 - Wiki, 14 - Reuters
% 15 - 20NewsHome, 16 - TDT 2, 17 -  RCV1_4Class,
% 18 - tae, 19 - seeds, 20 - heart, 21 - wpbc, 22 - wine,
% 23 - coil20, 24 - cifar10, 25 - cifar100, 26 - umist, 
% 27 - alphadigits, 28 - usps, 29 - NORB
benchmark = 5;

% =======================================================================
% 1 - k-means, 2 - , 3 - MUE, 4 - HALS
% 5 - , 6 - , 7 - 
% 8 - SimplexMax, 9 - SNPA, 10 - SPA, 
% 11 - XRAY (rand), 12 - XRAY (dist), 13 - XRAY (max), 14 - XRAY (Greedy)
% 15 - CNMF, 16 - CHNMF, 17 - HCHNMF
% 18 - MM, 19 - , 20 - HO-SVD(1), 21 - HO-XRAY, 
% 22 - HO-MM, 23 - , 24- 
% 
Methods = [4];

% ========================================================================
% Visualization
visualization = 1;

% INFO
% ========================================================================
% Y (I_1 x I_2 x ... I_N): I_1 - number of samples, I_2 x ... x I_N - attributes
[Y,groups] = benchmarks_load(benchmark);

%% Preprocessing
Y = full(Y);

%% Basic parameters
MCRuns = 10; % number of MC runs
param = parameters_clustering(Y,groups,benchmark);
J = param.rank;

%% Clustering algorithms
%% =========================================================================
ET = zeros([MCRuns,max(Methods)]);
Accur_ratio = zeros([MCRuns,max(Methods)]);
Purity = zeros([MCRuns,max(Methods)]);
MI = zeros([MCRuns,max(Methods)]);

% MC runs
%% =======================================================================
for k = 1:MCRuns
    
    % Initialization
    ET_k = zeros([max(Methods),1]); 
    Acc_k = zeros([max(Methods),1]); 
    Purity_k = zeros([max(Methods),1]); 
    MIhat_k = zeros([max(Methods),1]); 
    algorithm = struct;
    
    if length(param.size) > 2
       param.Ainit = rand(size(Y,1),J);
       param.Xinit = rand(J,prod(param.size)/size(Y,1));        
    else
       param.Ainit = rand(size(Y,1),J);
       param.Xinit = rand(J,size(Y,2));
    end
      
    for w = Methods
  
        disp(['MC run: ',num2str(k),', Method: ',num2str(w)]);
        algorithm.clustering = w;
              
        % Algorithm
        t_start = cputime;
        groups_estim  = clustering_fun(Y,algorithm,param);
        ET_k(w) = cputime - t_start; % elasped time for one algorithm
           
        if ~isempty(groups_estim)
            % ===========================================================
            BM_ratio = bestMap(groups,groups_estim);
            %=============  evaluate AC: accuracy ==============
            Acc_k(w) = length(find(groups == BM_ratio))/length(groups);
            %=============  evaluate MIhat: nomalized mutual information =================
            MIhat_k(w) = MutualInfo(groups,BM_ratio);
            
            % Purity(w) = AccMeasure(groups,groups_estim);
            Purity_k(w) = purity_clustering(groups_estim,groups,length(unique(groups)));
        end
                
    end % for w
    
    ET(k,:) = ET_k;  
    Accur_ratio(k,:) = Acc_k; 
    Purity(k,:) = Purity_k; 
    MI(k,:) = MIhat_k;
    
end % for k

 
%% Vizualization
 if visualization==1
     
     Accur_ratio(:,~sum(Accur_ratio)) = [];
     Purity(:,~sum(Purity)) = [];
     MI(:,~sum(MI)) = [];
     ET(:,~sum(ET)) = [];
     legend_tmp = char(info_legend.clustering);
     legend_tmp(setdiff(1:max(Methods),Methods),:) = [];
    
     figure
     subplot(3,1,1) % Accuracy
     boxplot(MI)
     xlabel('Algorytmy','FontName','Times New Roman','FontSize',24)
     ylabel('MI','FontName','Times New Roman','FontSize',24)
     set(gca,'FontName','Times New Roman','FontSize',20)
     set(gcf,'Color',[1 1 1])
     set(gca,'XTickLabel',legend_tmp)
     
     subplot(3,1,2) % Accuracy
     boxplot(Purity)
     xlabel('Algorytmy','FontName','Times New Roman','FontSize',24)
     ylabel('Purity','FontName','Times New Roman','FontSize',24)
     set(gca,'FontName','Times New Roman','FontSize',20)
     set(gcf,'Color',[1 1 1])
     set(gca,'XTickLabel',legend_tmp)
        
     subplot(3,1,3) % ET
     boxplot(ET)
     xlabel('Algorytmy','FontName','Times New Roman','FontSize',24)
     ylabel('Elapsed time [sec.]','FontName','Times New Roman','FontSize',24)
     set(gca,'FontName','Times New Roman','FontSize',20)
     set(gcf,'Color',[1 1 1])
     set(gca,'XTickLabel',legend_tmp)
            
end % vizualition


