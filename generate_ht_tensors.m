function [ Xht ] = generate_ht_tensors( dims, ranks )

N = size(ranks,1);
Xht = cell(N,1);

Xht{1} = sparse_matrix(dims(1), ranks(1), 0.9);
Xht{2} = sparse_matrix(dims(1), ranks(2), 0.9);
Xht{3} = sparse_matrix(dims(1), ranks(3), 0.9);
Xht{4} = sparse_matrix(dims(1), ranks(4), 0.9);

Xht{5} = sparse_matrix(ranks(1) * ranks(2), ranks(5), 0.9);
Xht{5} = reshape(Xht{5}, ranks(1), ranks(2), ranks(5));
Xht{6} = sparse_matrix(ranks(3) * ranks(4), ranks(6), 0.9);
Xht{6} = reshape(Xht{6}, ranks(3), ranks(4), ranks(6));

Xht{7} = sparse_matrix(ranks(5), ranks(6), 0.9);

% Xht{1} = max(0, randn(dims(1), ranks(1)));
% Xht{2} = max(0, randn(dims(2), ranks(2)));
% Xht{3} = max(0, randn(dims(3), ranks(3)));
% Xht{4} = max(0, randn(dims(4), ranks(4)));
% 
% Xht{5} = max(0, rand(ranks(1), ranks(2), ranks(5)));
% Xht{6} = max(0, rand(ranks(3), ranks(4), ranks(6)));
% 
% Xht{7} = max(0, rand(ranks(5), ranks(6)));

end

