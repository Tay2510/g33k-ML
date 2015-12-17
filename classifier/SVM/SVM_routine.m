xTrain = XTrainFeat;
yTrain = YTrain;
xTest = XTestFeat;
yTest = double(YTest);

yPredict = SVMClassify(model, xTest);
cMat = confusionmat(yTest, yPredict);
score = sum(diag(cMat)) / sum(sum(cMat))
    
    
    