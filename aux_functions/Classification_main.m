% Main script for testing classification algorithms
clear; close 

% Paths to algorithms
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
% 1 - no dimensionality reduction, 2 - PCA, 3 - MUE, 4 - HALS
% 5 - GNMF, 6 - MD2, 7 - SPG
% 8 - SimplexMax, 9 - SNPA, 10 - SPA, 
% 11 - XRAY (rand), 12 - XRAY (dist), 13 - XRAY (max), 14 - XRAY (Greedy)
% 15 - CNMF, 16 - CHNMF, 17 - HCHNMF
% 18 - MM, 19 - C-MM, 20 - HO-SVD(1), 21 - HO-XRAY, 
% 22 - HO-MM, 23 - HO-CMM, 24- HO-CMM-SVD
% 
Methods = [19];

% ========================================================================
% 1- KNN(E), 2 - KNN(C), 3 - SVM, 4 - NB, 5 - LDA
Classifiers = [2]; 

% Visualization
visualization = 1;

% INFO
% ========================================================================
% Y (I_1 x I_2 x ... I_N): I_1 - number of samples, I_2 x ... x I_N - attributes
[Y,groups] = benchmarks_load(benchmark);

%% Preprocessing
Y = full(Y);

%% Basic parameters
MCRuns = 1; % number of MC runs
NoCV = 5; % n-fold cross-validation
C = length(unique(groups)); % number of classes
param = parameters_classification(Y,groups,benchmark);

%% Classification algorithms
%% =========================================================================
ET = zeros([MCRuns,max(Methods),max(Classifiers)]);
cfMat = zeros([MCRuns,max(Methods),max(Classifiers),C,C]);
mcr_ratio = zeros([MCRuns,max(Methods),max(Classifiers)]);

cp = cvpartition(groups,'kfold',NoCV); % Stratified cross-validation
order = unique(groups);

% MC runs
%% =======================================================================
for k = 1:MCRuns
    
    % Initialization
    ET_k = zeros([max(Methods),max(Classifiers)]); 
    cfMat_k = zeros([max(Methods),max(Classifiers),C,C]);
    mcr_ratio_k = zeros([max(Methods),max(Classifiers)]); 
    algorithm = struct;
      
    for w = Methods
        for v = Classifiers;

            disp(['MC run: ',num2str(k),', Method: ',num2str(w),', Classifier: ',num2str(v)]);

            algorithm.DimRed = w;
            algorithm.Classifier = v;

            t_start = cputime;
            f_class_1 = @(xtrain,ytrain,xtest) classify_fun(xtrain,ytrain,xtest,algorithm,param);
       %     f_class_2 = @(xtrain,ytrain,xtest) cell(classify_fun_outer(xtrain,ytrain,xtest,algorithm,param));
            f_class_3 = @(xtrain,ytrain,xtest,ytest)confusionmat(ytest,feval(f_class_1,xtrain,ytrain,xtest),'order',order);
            CMtx = reshape(sum(crossval(f_class_3,Y,groups,'partition',cp)),C,C);
            ET_k(w,v) = (cputime - t_start)/NoCV; % elasped time for one CV
            cfMat_k(w,v,:,:) = CMtx;
            mcr_ratio_k(w,v) =  100*(sum(sum(CMtx)) - sum(diag(CMtx)))/sum(sum(CMtx)); 
        %    [~,info_legend] = feval(f_class_1,[],[],[]);

        end % for v
    end % for w
    
    ET(k,:,:) = ET_k;  
    cfMat(k,:,:,:,:) = cfMat_k;
    mcr_ratio(k,:,:) = mcr_ratio_k;
    
end % for k

 
%% Vizualization
 if visualization==1
     
     figure
     i = 0;
     for v = Classifiers;
         for w = Methods
            
            i = i + 1;            
            subplot(length(Classifiers),length(Methods),i)
            hintonw((squeeze(mean(cfMat(:,w,v,:,:),1)))')
            title([info_legend(w,v).DimRed,', ',info_legend(w,v).classifier,', MCR = ',num2str( squeeze(mean(mcr_ratio(:,w,v),1))  )])
            ylabel('Output')
            
        end
     end
     set(gcf,'Color',[1 1 1])
            
end % vizualition


