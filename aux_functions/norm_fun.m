function Y = norm_fun(Y,param_norm,inx_cols)

% inx_cols = 1: column-normalizaation
% inx_cols = 0: row-normalizaation

% normalization 
        switch param_norm

            case 0 % no normalization

                Y = Y;

            case 1 % L1-norm

                if inx_cols
                   Y = bsxfun(@rdivide,Y,sum(Y,1));
                else
                   Y = bsxfun(@ldivide,sum(Y,2),Y);
                end                    

            case 2 % L2-norm

                if inx_cols
                   Y = bsxfun(@rdivide,Y,sqrt(sum(Y.^2,1)));
                else
                   Y = bsxfun(@ldivide,sqrt(sum(Y.^2,2)),Y);    
                end
             
        end % normalization
        
end
        
        