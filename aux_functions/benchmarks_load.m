function [Y,groups] = benchmarks_load(benchmark)

switch benchmark  
    
       case 1 % MNIST
           
            Sx = load('MNIST_train.mat');
            Y = Sx.imgs;
            groups = Sx.labels+1;            
            Y = permute(Y,[3 1 2]);
    
       case 2 % Movie (IWANN 2017)
            
            Sx = load('film4.mat');
            Y = Sx.X0;
            groups = (Sx.inx)';
            Yf = permute(Y,[4 1 2 3]);
            Y = Yf(:,1:4:end,1:4:end,:);     
            
       case 3 % Smartphone datasets
           
            inx_read = 1; % 1 - upload tensors from the mat-file; 
                          % 2 - upload data from the tex-file, and transform them with DWT            
            [Y,groups] = smartphone_data(inx_read);  
             Y = permute(Y,[3 1 2]);
            
       case 4 % DriveFace
           
            inx_read = 1; % 1 - upload tensors from the mat-file; 
                          % 2 - upload data from the tex-file, and transform them with DWT            
            [Y,groups] = DriveFace_data(inx_read);   
             Y = permute(Y,[3 1 2]);        
             
       case 5 % ORL facial images
           
            Sx=load('ORL_tensors');
            Y = double(cat(3,Sx.C_train,Sx.C_test));
            Y = permute(Y,[3 1 2]); 
            groups = [Sx.Class_train_inx; Sx.Class_test_inx];       
            
       case 6 % Emotions-Wavelets
           
            Sx=load('emotion_wavelets_data_base');
            Y = double(cat(3,Sx.C_train,Sx.C_test));
            Y = permute(Y,[3 1 2]); 
            groups = [Sx.Class_train_inx; Sx.Class_test_inx];         
            
       case 7 % Digits (student)
           
            Sx=load('Digits_tensors');
            Y = double(cat(3,Sx.C_train,Sx.C_test));
            Y = permute(Y,[3 1 2]); 
            groups = [Sx.Class_train_inx; Sx.Class_test_inx];          
                        
       case 8 % Textures
           
            Sx=load('Texture_tensors');
            Y = double(cat(3,Sx.C_train,Sx.C_test));
            Y = permute(Y,[3 1 2]); 
            groups = [Sx.Class_train_inx; Sx.Class_test_inx];          
            
       case 9 % Musical instruments
           
            Sx=load('WAVE_tensors_4s_supervised');
            Y = double(cat(3,Sx.C_train,Sx.C_test));
            Y = permute(Y,[3 1 2]); 
            groups = [Sx.Class_train_inx; Sx.Class_test_inx];    
            
       case 10 % TDT
           
            Sx=load('TDT');
            Y = Sx.Y;
            groups = Sx.groups;     
           
       case 11 % Citeseer
           
            Sx=load('Citeseer');
            Y = Sx.Y;
            groups = Sx.groups;     
            
       case 12 % Cora
           
            Sx=load('Cora');
            Y = Sx.Y;
            groups = Sx.groups;     
            
       case 13 % Wiki
           
            Sx=load('Wiki');
            Y = Sx.Y;
            groups = Sx.groups; 
            
       case 14 % Reuters
           
            Sx=load('Reuters21578');
            Y = Sx.fea(:,1:8000)';
            groups = full(Sx.gnd(1:8000));  
            
       case 15 % 20NewsHome
           
            Sx=load('20NewsHome');
            Y = Sx.fea;
            groups = full(Sx.gnd);  
            
       case 16 % TDT2_all
        
            Sx=load('TDT2_all');
            Y = Sx.fea;
            groups = full(Sx.gnd); 
           
       case 17 % RCV1_4Class
           
            Sx=load('RCV1_4Class');
            Y = Sx.fea;
            groups = full(Sx.gnd); 
           
       case 18 % tae
           
            Sx=load('tae');
            Y = Sx.Y;
            groups = Sx.groups;  
            
       case 19 % seeds
           
            Sx=load('seeds');
            Y = Sx.Y;
            groups = Sx.groups; 
            
       case 20 % heart
           
            Sx=load('heart');
            Y = Sx.Y;
            groups = Sx.groups;
            
       case 21 % wpbc
           
            Sx=load('wpbc');
            Y = Sx.Y;
            groups = Sx.groups;
            
       case 22 % wine
           
            Sx=load('wine');
            Y = Sx.Y;
            groups = Sx.groups;
            
       case 23 % coil100
               
            Sx=load('coil_100');
            Y = Sx.Y;
            Y = permute(Y,[4 1 2 3]);
            groups = Sx.groups';
            
       case 24 % cifar10
            
            Sx=load('cifar10');
            Y = Sx.Y;
            groups = Sx.groups+1;
            
       case 25 % cifar100
            
            Sx=load('cifar100');
            Y = Sx.Y;
            groups = Sx.groups+1;
            
       case 26 % umist
            
            Sx=load('umist');
            Y = Sx.Y';
            groups = Sx.groups;   
            
       case 27 % alphadigits
            
            Sx=load('alphadigits');
            Y = Sx.Y;
            groups = Sx.groups';                 
            Y = permute(Y,[3 1 2]);
            
       case 28 % usps
            
            Sx=load('usps');
            Y = Sx.Y';
            groups = Sx.groups;
            
       case 29 % NORB
            
            Sx=load('NORB_small');
            Y = Sx.Y;
            groups = Sx.groups';                 
            Y = permute(Y,[3 1 2]);            
            
end

