function [F] = SVMClassify(Model, X)
tic
xTest = X;
[NTest MTest] = size(xTest);
yTest = zeros(NTest, 1);

classNum = length(Model.alphaCell);
epsilon = 0.0001;   
G = (Model.XTrain)*(Model.XTrain)';
F = zeros(NTest, classNum);
for k = 1 : classNum
   alpha = Model.alphaCell{k};   
   SVIndex = find(Model.alphaCell{k} > epsilon & Model.alphaCell{k} < (Model.C - epsilon));   
   SV = Model.XTrain(SVIndex, :); size(SV)
   a = alpha(SVIndex);              size(a)
   y = double(Model.yTrain(SVIndex));    size(y)
   b = mean(1./y - G(SV, :)*(alpha.*y));
   
   for n = 1 : NTest
       xNew = xTest(n, :)';              
       F(n, k) = sum(a.*y.*(SV*xNew)) + b;
   end
end

toc
end