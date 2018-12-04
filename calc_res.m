function [ res ] = calc_res( X, Y )

D = Y-X;

normD = sqrt(sum(D(:).^2));
normX = sqrt(sum(X(:).^2));

res = normD / normX;

end

