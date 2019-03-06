function [ Xtt ] = generate_core_tensors( dims, ranks )

N = size(dims,2);
Xtt = cell(N,1);

left_r = ranks(1);
Xtt{1} = max(0, randn(dims(1), left_r));
for n=2:N-1
    right_r = ranks(n);
    Xtt{n} = max(0, randn(left_r, dims(n), right_r));
    left_r = right_r;
end
Xtt{4} = max(0, randn(left_r, dims(N)));

end

