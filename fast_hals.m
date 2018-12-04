function X = fast_hals(A,Y,J, k_inner)
    X = rand(J,size(Y,2));
    r = size(A,2);
    W = A'*Y; V = A'*A; 
    
    for iter = 1:k_inner

        for j = 1:r 
            X(j,:) = max(eps,X(j,:) + (W(j,:) - V(j,:)*X)/V(j,j));  
        end
              
    end % for k
    
end % function FAST-HALS_inner
