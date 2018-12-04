function [Y,Uw] = bss_benchmarks_load(Structure_data)

Y = []; Uw = [];

if isempty(Structure_data.Y)

    N = length(Structure_data.DimY); % number of modes
    for n = 1:N
        if Structure_data.U(n) == 1 % random
           Uw{n} = Sparse_matrix(Structure_data.DimY(n),Structure_data.J,Structure_data.sparsity(n));
 %          Uw{n} = rand([Structure_data.DimY(n) Structure_data.J]);
%            if n == 1
%               Uw{n} = Sparse_matrix(Structure_data.DimY(n),Structure_data.J,Structure_data.sparsity(n));
%        
%            else
%               Uw{n} = max(0,rand([Structure_data.DimY(n) Structure_data.J]));
%            end            
           
        elseif Structure_data.U(n) == 2 % USGS
            
             Ax = load('USGS_pruned_10_deg.mat');
             Awx = Ax.B;
      %       Uw{n} = Awx(:,[2 31 38 49]); % for USGS (ICAISC 2014)
      %       Uw{n} = Awx(:,[2 31 38 49 61]); % for USGS (ICAISC 2014)
             inx_select = randperm(size(Awx,2),Structure_data.J);
             Uw{n} = Awx(:,inx_select); % for USGS (ICAISC 2014)
      
             
        elseif Structure_data.U(n) == 3 % synthetic abundance (Miao)
            
             win = 5;
             pure_pixels = 1;
             [Ux,M,N] = (getSynAbundance(Structure_data.DimY(n),Structure_data.J, win, pure_pixels));
             Uw{n} = Ux';
             
        elseif Structure_data.U(n) == 4 % synthetic abundance (Aggarwal)
            
            S = load('abundance');
            Uw{n} = reshape(S.abundance(:,:,1:Structure_data.J),50^2,Structure_data.J);                 
            
        elseif Structure_data.U(n) == 5 % Sparse10sinc4.mat (ICALAB)
            
            S = load('Sparse10sinc4.mat');
            Uw{n} = S.X(1:Structure_data.J,1:200)';           
        
        elseif Structure_data.U(n) == 6 % Sparse10sinc4.mat (ICALAB)
            
            S = load('Sparse10sinc4.mat');
            Uw{n} = S.X(1:Structure_data.J,300:500)';         
            
        end                       
        Uw{n} = norm_fun(Uw{n},Structure_data.norm(n),1); % normalization        
    end % for n
        
    lambda = ones(Structure_data.J,1); % super-diagonal from the core tensor
    Y = double(ktensor(lambda,Uw));
        
    % Noise
    if ~isempty(Structure_data.SNR)
        Nt = randn(size(Y)); 
        tau = (norm(Y(:))/norm(Nt(:)))*10^(-Structure_data.SNR/20);
       % SNR = 20*log10(norm(Y_true,'fro')/norm(tau*N,'fro'))
        Y = Y + tau*Nt;
    end   
    
else % real data
    
    
    
    
end



















end