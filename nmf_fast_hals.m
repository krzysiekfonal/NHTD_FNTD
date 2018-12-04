function [A,X,res] = nmf_fast_hals(Y,R,tol_obj,MaxIter,show_inx)

A = rand(size(Y,1),R); X = rand(R,size(Y,2));
tol_proj = 1e-12;
gradA = A*(X*X') - Y*X'; gradX = (A'*A)*X - A'*Y;
initgrad = norm([gradA; gradX'],'fro');
if show_inx
    fprintf('Init gradient norm %f\n', initgrad); 
end
tolA = max(0.00001,tol_proj)*initgrad; tolX = tolA;
normY = norm(Y,'fro');
res(1) = norm(Y - A*X,'fro')/normY;
k_inner = 10;

    for k = 1:MaxIter

        projnorm = norm(gradA(gradA<0 | A>0)) + norm(gradX(gradX<0 | X>0));
        if show_inx & (k < 10 | ~mod(k,10))
           disp(['FAST-HALS iterations: ',num2str(k), ',  Gradient norm: ',num2str(projnorm )]);
        end

      % Stopping criterion  
        if projnorm < tol_proj*initgrad,
           break;
        end
                
      % Update for X
        [X,gradX,iterX] = fast_hals_inner(A,Y,X,R,tolX,k_inner); 
        if iterX == 1
           tolX = 0.1*tolX;
        end
   
% %       % Normalization
        dX = eps+(sum(X,2));
        X = bsxfun(@ldivide,dX,X);
        A = bsxfun(@times,dX',A);

      % Update for A  
        [At,gradA,iterA] = fast_hals_inner(X',Y',A',R,tolA,k_inner);
        A = At'; gradA = gradA';
        if iterA == 1
           tolA = 0.1*tolA;
        end

% %       % Normalization
        dA = eps+(sum(A,1));
        A = bsxfun(@rdivide,A,dA);
        X = bsxfun(@times,dA',X);
        
        Y_hat = A*X;
        res(k+1) = sqrt(normY^2 + norm(Y_hat, 'fro')^2 - 2*sum(Y(:) .* Y_hat(:)))/normY;
        
       % Stagnation 
        if k > 10
            if abs(res(k) - res(k+1))/res(1) < tol_obj
                disp(['Stagnation of FAST-HALS iterations: ',num2str(k+1), ', Residual: ',num2str(res(k+1)), ', Diff_res: ', num2str(res(k) - res(k+1))]);
               break;
            end
        end % if

    end
    
end

% ========================================================================
function [X,G,iter] = fast_hals_inner(A,Y,X,r,tol,k_inner)

    W = A'*Y; V = A'*A; 
    for iter = 1:k_inner

      % stopping
       G = V*X - W;
       projgrad = norm(G(G < 0 | X > 0));
       if projgrad < tol
          break
       end

        for j = 1:r 
            X(j,:) = max(eps,X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j));  
        end
              
    end % for k
end % function FAST-HALS_inner




