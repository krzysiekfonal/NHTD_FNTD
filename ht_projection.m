function [ Ycl ] = ht_projection( Yht, mode )

if mode == 1
    Ycl = tensor_contraction(...
            tensor_contraction(Yht{1}, Yht{5}, 2, 1), Yht{7}, 3, 1);
    dims = size(Ycl);
    Ycl = reshape(Ycl, dims(1), dims(2)*dims(3));
elseif mode == 4
    Ycl = tensor_contraction(...
            tensor_contraction(Yht{4}, Yht{6}, 2, 2), Yht{7}, 3, 2);
    dims = size(Ycl);
    Ycl = reshape(Ycl, dims(1), dims(2)*dims(3));
end

end

