function Purity = purity_clustering(inx_XH,Labels,J)

l = [];
for i = 1:J
    for j = 1:J
        l(i,j) = length(find(inx_XH(find(Labels == i)) == j));        
    end
end
if ~prod(sum(l,1))
    disp('Zero-value columns in L');
end
l_max = max(l,[],1);
Purity = sum(l_max)/length(inx_XH);

end
