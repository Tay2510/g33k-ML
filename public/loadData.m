display('Loading data...');

XTrain = [];
YTrain = [];
batchParts = 5;
for n = 1 : batchParts
    path = strcat('small_data_batch_', num2str(n));
    load(path);
    if n ~= batchParts        
        XTrain = [XTrain; data];
        YTrain = [YTrain; labels];
    else
        XTest = data;
        YTest = labels;
    end
end

dXTrain = double(XTrain);
dXTest = double(XTest);