function SIR = CalcSIR(A,Aest)

% Sergio Cruces & Andrzej Cichocki 
% A=A*diag(1./(sqrt(sum(A.^2))+eps));
% Aest=Aest*diag(1./(sqrt(sum(Aest.^2))+eps));

A=bsxfun(@rdivide,A,sum(A,1));
Aest=bsxfun(@rdivide,Aest,sum(Aest,1));

for i=1:size(Aest,2)
  [MSE(i),ind]=min(sum(bsxfun(@minus,Aest(:,i),A).^2,1));
  %A(:,ind) = [];
end
SIR=-10*log10(MSE); 
