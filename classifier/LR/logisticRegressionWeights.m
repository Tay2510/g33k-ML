function [w] = logisticRegressionWeights(xTrain, yTrain, w0, nIter)

e = 0.1;
w = w0;
featureDim = length(w0);
datalength = length(yTrain);

% Modify data by inserting column of ones
testColumn = xTrain(:, 1);
if sum(testColumn) ~= datalength
    oneColumn = ones(datalength, 1);
    xTrain = [oneColumn xTrain];
end

for n = 1: nIter
    % Gradient Ascent
    grad = zeros(featureDim, 1);
    for g = 1: featureDim
        gSum = 0;
        for i = 1 : datalength
            row = xTrain(i, :);
            Exp = exp(row*w);
            p = Exp/(1 + Exp);
            gSum = gSum + xTrain(i, g)*(yTrain(i) - p);
        end
        grad(g) = gSum;
    end
    w = w + e*grad;    
end

end
