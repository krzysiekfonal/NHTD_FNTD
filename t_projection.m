function [ Ycl ] = t_projection( Yht, mode )

if mode == 1
    Ycl = tensor_contraction(Yht{1}, Yht{5}, 2, 1);
    dims = size(Ycl);
    Ycl = reshape(Ycl, dims(1), dims(2)*dims(3)*dims(4));
elseif mode == 4
    Ycl = tensor_contraction(Yht{4}, Yht{5}, 2, 4);
    dims = size(Ycl);
    Ycl = reshape(Ycl, dims(1), dims(2)*dims(3)*dims(4));
end

end

