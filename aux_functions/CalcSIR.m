function SIR = CalcSIR(A,Aest)

% Sergio Cruces & Andrzej Cichocki 
%A=A*diag(1./(sqrt(sum(A.^2))+eps));
%Aest=Aest*diag(1./(sqrt(sum(Aest.^2))+eps));

A=bsxfun(@rdivide,A,eps+sqrt(sum(A.^2,1)));
Aest=bsxfun(@rdivide,Aest,eps+sqrt(sum(Aest.^2,1)));

for i=1:size(Aest,2)
  [MSE(i),ind]=min(sum(bsxfun(@minus,Aest(:,i),A).^2,1));
end
SIR=-10*log10(MSE); 
