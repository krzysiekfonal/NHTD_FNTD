function Y = norm_fun(Y,param_norm)

% normalization 
        switch param_norm

            case 0 % no normalization

                Y = Y;

            case 1 % L1-norm

                N = ndims(Y);
                Y_L1 = double(collapse(Y,[2:N])); % L1-norm along all but first mode
                Y = scale(Y,1./Y_L1,1);
       
            case 2 % L2-norm

                N = ndims(Y);
                Y_L2 = sqrt(double(collapse(power(Y,2),[2:N]))); % L1-norm along all but first mode
                Y = scale(Y,1./Y_L2,1);
             
        end % normalization
        
end
        
        