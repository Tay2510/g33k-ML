clc
%% KNN
K = [5 10 15 20];

xTrain = XTrainFeat;
yTrain = YTrain;
xTest = XTestFeat;
yTest = double(YTest);

sizeK = length(K);
hogAccKNNs = zeros(1, sizeK);

for j = 1 : sizeK
    k = K(j);
    yPredict = KNNClassify(xTrain, yTrain, xTest, k);
    cMat = confusionmat(yTest, yPredict);
    score = sum(diag(cMat)) / sum(sum(cMat));
    hogAccKNNs(j) = score;
    display(['k = ' num2str(k) ': ' num2str(score)]);
end
