clc
tic
%% Use SGD (no nIter)
xTrain = normc(XTrainFeat);
yTrain = YTrain;
xTest = normc(XTestFeat);
yTest = YTest;

[NTrain MTrain] = size(xTrain);    % N: samples ; M: features
[NTest MTest] = size(xTest);

% Modify data by inserting column of ones
oneColumnTrain = ones(NTrain, 1);
xTrainTemp = [oneColumnTrain xTrain];
oneColumnTest = ones(NTest, 1);
xTestTemp = [oneColumnTest xTest];

% set w0
w0 = rand(MTrain + 1, 10);

% Training (stochastic gradient ascent)
display('Training...');    
weights = LRTrainSG(xTrainTemp, yTrain, w0);

% Classification
display('Classifying...');
YPredict = LRClassify(xTestTemp, weights);

% evalutaion
cMat = confusionmat(yTest, YPredict);
hogAcc = sum(diag(cMat)) / sum(sum(cMat));
display(hogAcc);

toc
