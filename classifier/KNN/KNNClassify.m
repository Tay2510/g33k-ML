function [cls] = KNNClassify(xTrain, yTrain, xTest, k)
% [Input]
% ==============================================================
% xTrain : training data (N, M)             ; M = #(feature)
% yTrain : feature vector (N, 1)            ; N = #(training sample)
% xTest  : testing data (N', M)             ; N'= #(testing sample) 
% k      : the user-defined number of nearest neighbors
% ==============================================================
% [Output] : a vector of classified labels

[NTrain MTrain] = size(xTrain);
[NTest, MTest] = size(xTest);
cls = zeros(NTest, 1);

% iteration for all testing samples
for i = 1 : NTest    
    % cache for test sample
    testSample = xTest(i, :);
    % cache for measurement score (freq. / avg. dist...etc) for label 0 ~ 9
    measures = zeros(1, 10); 
    % cache for distance
    distances = zeros(1, NTrain);
    
    % iteration for all training samples
    for n = 1 : NTrain
        % cache for reference sample
        refSample = xTrain(n, :);
        distances(n) = EucliDist(testSample, refSample);
    end
    
    [d, index] = sort(distances);
    index_nearest = index(1:k);    
    label_nearest = yTrain(index_nearest, :);   
    for y = 0 : 9
        measures(y + 1) = sum(label_nearest == y);
    end
    labels = find(measures == max(measures));        
    label = labels(1) - 1;
    cls(i) = label;    
end

end

