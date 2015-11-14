function [cls] = logisticRegressionClassify(xTest, w)
datalength = length(xTest);
cls = zeros(datalength, 1);

for i = 1 : datalength
    row = xTest(i, :);
    Exp = exp(row*w);
    P1 = Exp/(1 + Exp);
    P0 = 1 - P1;
    if P1 > P0
        cls(i) = 1;
    end
end

end
