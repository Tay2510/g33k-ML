function [y] = SVMClassify(Model, X)

[NTest MTest] = size(X);
Y = Model.yTrain;
l = length(Y);
classNum = length(Model.alphaCell);
F = zeros(NTest, classNum);
K = Model.XTrain*(Model.XTrain)';
C = Model.C;

for k = 1 : classNum
   alpha = Model.alphaCell{k};  
   alpha(alpha < C*0.00001) = 0;
   alpha(alpha > C*0.999999999) = C;
   SVIndex = find(alpha > 0 & alpha < C);      
   SV = Model.XTrain(SVIndex, :);    
   ysv = double(Model.yTrain(SVIndex));          
   svone = zeros(l, 1); 
   svone(SVIndex, 1) = 1;
   b = svone'*(Y - ((alpha.*Y)'*K')')/sum(svone);
   Ki = SV*X';
   temp = Ki'*(alpha(SVIndex).*ysv) + b ;     
   
   F(:, k) = temp;   
end

y = zeros(NTest, 1);
for n = 1 : NTest    
    test = F(n, :);
    [M I] = max(test);
    label = I(1) - 1;
    y(n) = label;
end

end