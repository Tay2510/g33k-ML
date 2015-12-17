xTrain = double(XTrainFeat);
yTrain = double(YTrain);
ySVM = yTrain;  
[NTrain MTrain] = size(xTrain);

C = 20*(1/NTrain);          % for non-linear SVM
classNum = 1;            % 0~9


Model.alphaCell = cell(classNum, 1);
Model.XTrain = xTrain;
Model.yTrain = yTrain;
Model.C = C;

Test = cell(classNum, 1);

K = xTrain*xTrain';      % linear kernel

for n = 1 : classNum
    display(n);
    ySVM = (-1)*ones(NTrain, 1);
    label = n - 1;
    [rowIndex colIndex] = find(yTrain == label);
    ySVM(rowIndex) = 1;
    Test{n} = ySVM;    
    
    H = ySVM*(ySVM').*K;    % ???
    
    f = -1*ones(NTrain, 1);
    A = [];
    b = [];
    Aeq = ySVM'; 
    beq = 0;
    lb = zeros(NTrain, 1);
    ub = C*ones(NTrain, 1);

    [ALPHA, VAL] = quadprog(H, f, A, b, Aeq, beq, lb, ub) ;

    Model.alphaCell{n} = ALPHA;
    
end