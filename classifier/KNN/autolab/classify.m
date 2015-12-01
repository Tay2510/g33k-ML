function y = classify(Model, X)

xTrain = Model.xTrain;
yTrain = Model.yTrain;
xTest = X;
k = 10;
[NTrain MTrain] = size(xTrain);
[NTest, MTest] = size(xTest);

% iteration for all testing samples
y = zeros(NTest, 1);
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
        distances(n) = norm(testSample - refSample);
    end
    
    [d, index] = sort(distances);
    index_nearest = index(1:k);    
    label_nearest = yTrain(index_nearest, :);   
    for label = 0 : 9
        measures(label + 1) = sum(label_nearest == label);
    end
    labels = find(measures == max(measures));        
    label = labels(1) - 1;
    y(i) = label;    
end
end