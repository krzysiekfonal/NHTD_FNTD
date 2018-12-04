function [X,G,iter] = fast_hals_inner(A,Y,X,tol,k_inner)

    W = A'*Y; V = A'*A; 
    R = size(A,2);
    
    for iter = 1:k_inner

     % stopping
       G = V*X - W;
       projgrad = norm(G(G < 0 | X > 0));
       if projgrad < tol,
          break
       end

       for j = 1:R 
           X(j,:) = max(eps,X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j));  
  %         X(j,:) = X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j);  
       end
              
    end % for k
    
end % function FAST-HALS_inner
