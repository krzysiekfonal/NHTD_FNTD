function [ X ] = tucker_to_tensor( Xt )

N = size(Xt);
X = Xt{N};
for n=1:N-1
    X = tensor_contraction(X, Xt{n}, 1, 2);
end

end

