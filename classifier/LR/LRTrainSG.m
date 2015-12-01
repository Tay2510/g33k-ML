function [w] = LRTrainSG(xTrain, yTrain, w0)
% [Input]
% ==============================================================
% xTrain : training data (N, M + 1)         ; M = #(feature)
% yTrain : feature vector (N, 1)            ; N = #(sample)
% w0 : initial matrix of weights (M + 1, Y) ; Y = #(label)
% ==============================================================
% [Output] : a trained matrix of weights 

% Stochastic Gradient Ascent
e = 0.1;    % eta, a coefficient for gradient step modification         
w = w0;
featureDim = length(w0);        % featureDim = M + 1
datalength = length(yTrain);    % datalength = N

% iteration for all training samples (stochastic gradient ascent)
for i = 1: datalength
    sample = xTrain(i, :);
    label = yTrain(i);
    % iteration for all labels
    for y = 0 : 9
        increment = zeros(featureDim, 1);
        booleanTempVar = 0;
        if label == y
            booleanTempVar = 1;
        end
        % iteration for all features
        for g = 1: featureDim            
            sigP = sigmoidProb(label, sample, w);
            increment(g) = sample(g)*(booleanTempVar - sigP);
            display([num2str(i) '/' num2str(y) '/' num2str(g) ':' num2str(sigP)]);
        end
        w(:, y + 1) = w(:, y + 1) + e*increment;
    end    
end
end
