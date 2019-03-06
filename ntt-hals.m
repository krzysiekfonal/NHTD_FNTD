function [ Ytt ] = ntt-hals( X, ranks, maxiters )
N = ndims(X);
Ytt = generate_core_tensors(N, ranks);
for i=1:maxiters
    for n=N:-1:2
        
    end
end

end

