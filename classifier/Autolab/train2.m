function Model = SVMTrain(X, Y)
cellSize = 4;
% Extract test HoG
%================================================
xTrain = [];
size_XTrain = size(X, 1);
for i = 1 : size_XTrain
    im = im2single(reshape(X(i,:), 32, 32, 3));
    hog = vl_hog(im, cellSize);
    xTrain = [xTrain; hog(:)'];    
end
%================================================
xTrain = double(xTrain);
yTrain = double(Y);
ySVM = yTrain;  
[NTrain MTrain] = size(xTrain);
C = 0.025;
classNum = 10;           % 0~9

K = xTrain*xTrain';      % linear kernel

Model.alphaCell = cell(classNum, 1);
Model.XTrain = xTrain;
Model.yTrain = yTrain;
Model.C = C;

for n = 1 : classNum    
    ySVM = (-1)*ones(NTrain, 1);
    label = n - 1;
    [rowIndex colIndex] = find(yTrain == label);
    ySVM(rowIndex) = 1;
    
    display(length(find(ySVM > 0)));    
    
    H = ySVM*(ySVM').*K;
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


end