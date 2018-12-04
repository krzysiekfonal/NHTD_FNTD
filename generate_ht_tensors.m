function [ Xht ] = generate_ht_tensors( dims, ranks )

N = size(ranks,1);
Xht = cell(N,1);

Xht{1} = max(0, randn(dims(1), ranks(1)));
Xht{2} = max(0, randn(dims(2), ranks(2)));
Xht{3} = max(0, randn(dims(3), ranks(3)));
Xht{4} = max(0, randn(dims(4), ranks(4)));

Xht{5} = max(0, rand(ranks(1), ranks(2), ranks(5)));
Xht{6} = max(0, rand(ranks(3), ranks(4), ranks(6)));

Xht{7} = max(0, rand(ranks(5), ranks(6)));

end

