function [w] = LRTrain(xTrain, yTrain, w0, nIter)
% [Input]
% ==============================================================
% xTrain : training data (N, M + 1)         ; M = #(feature)
% yTrain : feature vector (N, 1)            ; N = #(sample)
% w0 : initial matrix of weights (M + 1, Y) ; Y = #(label)
% nIter : user-defined iteration steps
% ==============================================================
% [Output] : a trained matrix of weights 

% Gradient Ascent
e = 0.1;    % eta, a coefficient for gradient step modification         
w = w0;
featureDim = length(w0);        % featureDim = M + 1
datalength = length(yTrain);    % datalength = N

% iteration for nIter times (or until converge, not implemented though)
for n = 1: nIter   
    % iteration for all labels
    for y = 0 : 9
        increment = zeros(featureDim, 1);
        % iteration for all features
        for g = 1: featureDim
            gSum = 0;
            % iteration for all training samples (batch gradient ascent)
            for i = 1 : datalength
                sample = xTrain(i, :);
                label = yTrain(i);
                booleanTempVar = 0;
                if label == y
                    booleanTempVar = 1;
                end
                gSum = gSum + sample(g)*(booleanTempVar - sigmoidProb(label, sample, w));
                display([num2str(n) '/' num2str(y) '/' num2str(g) '/' num2str(i)])
            end
            increment(g) = gSum;
        end
        w(:, y + 1) = w(:, y + 1) + e*increment;
    end    
end
end
