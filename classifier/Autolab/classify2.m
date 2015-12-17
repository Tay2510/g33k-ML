function [y] = classify(Model2, X)

cellSize = 4;
% Extract test HoG
%================================================
xTest = [];
size_XTest = size(X, 1);
for i = 1 : size_XTest
    im = im2single(reshape(X(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    xTest = [xTest; hog(:)'];    
end
%================================================

[NTest, MTest] = size(xTest);

Y = Model2.yTrain;
l = length(Y);
classNum = length(Model2.alphaCell);
F = zeros(NTest, classNum);
K = Model2.XTrain*(Model2.XTrain)';
C = Model2.C;

for k = 1 : classNum
   alpha = Model2.alphaCell{k};  
   alpha(alpha < C*0.00001) = 0;
   alpha(alpha > C*0.999999999) = C;
   SVIndex = find(alpha > 0 & alpha < C);      
   SV = Model2.XTrain(SVIndex, :);    
   ysv = double(Model2.yTrain(SVIndex));          
   svone = zeros(l, 1); 
   svone(SVIndex, 1) = 1;
   b = svone'*(Y - ((alpha.*Y)'*K')')/sum(svone);
   Ki = SV*xTest';
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