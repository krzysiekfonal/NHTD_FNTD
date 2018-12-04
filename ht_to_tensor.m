function [ X ] = ht_to_tensor( Xht )

B1 = tensor_contraction(tensor_contraction(Xht{5}, Xht{1}',1,1),...
    Xht{2}', 1, 1);
B2 = tensor_contraction(tensor_contraction(Xht{6}, Xht{3}',1,1),...
     Xht{4}', 1, 1);
X = tensor_contraction(tensor_contraction(Xht{7}, B1, 1, 1),...
    B2, 1, 1);

end

