function [A,X,res] = nmf_als(Y,R,tol_obj,MaxIter,show_inx)

A = rand(size(Y,1),R); X = rand(R,size(Y,2));
res(1) = norm(Y - A*X,'fro')/norm(Y,'fro');
lambda = 1e-12;

    for k = 1:MaxIter

        if show_inx & ~mod(k,10)
           disp(['ALS iterations: ',num2str(k)]);
        end

      % Update for X
        X = max(eps,inv(A'*A + lambda*eye(R))*(A'*Y));

      % Normalization
        dX = eps+(sum(X,2));
        X = bsxfun(@ldivide,dX,X);
        A = bsxfun(@times,dX',A);

      % Update for A  
        A = max(eps,Y*X'*inv(X*X' + lambda*eye(R)));

      % Normalization
        dA = eps+(sum(A,1));
        A = bsxfun(@rdivide,A,dA);
        X = bsxfun(@times,dA',X);

        res(k+1) = norm(Y - A*X,'fro')/norm(Y,'fro');
        
      % Stagnation 
        if k > 2
            if abs(res(k) - res(k+1))/res(1) < tol_obj
                disp(['Stagnation of ALS iterations: ',num2str(k+1), ', Residual: ',num2str(res(k+1)), ', Diff_res: ', num2str(res(k) - res(k+1))]);
                break;
            end
        end % if

    end

end
