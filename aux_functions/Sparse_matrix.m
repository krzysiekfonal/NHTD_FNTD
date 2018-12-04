function A = Sparse_matrix(I,J,sA)

% Mixing matrix
a_row = 1; a_col = 1;
A= rand(I,J).*max(0,sign(rand(I,J)-sA));
while (~isempty(a_row) | ~isempty(a_col)) & (min(size(A)) > 1)
     % a_row = find(sum(A>0,2)<2); % minimum 2 non-zero entries
      a_row = find(sum(A>0,2)<1);
      if ~isempty(a_row)
          for i = 1:length(a_row)
              A(a_row(i),ceil(rand(1,2)*J)) = abs(rand(1,2) + eps);
          end
      end
    %  a_col = find(sum(A>0,1)<2); % minimum 2 non-zero entries
      a_col = find(sum(A>0,1)<1);
      if ~isempty(a_col)
         for i = 1:length(a_col)
             A(ceil(rand(2,1)*I),a_col(i)) = abs(rand(2,1) + eps);
         end
      end
end % while