function Model = SVMTrain(X, Y)
XTrain = X;
YTrain = Y;
[NTrain MTrain] = size(X);

H = [eye(MTrain)];
H(MTrain + 1, MTrain + 1) = 0;

f = zeros (MTrain + 1 ,1);
A = -[diag(YTrain)*XTrain YTrain];
b = -ones(NTrain ,1) ;

[ALPHA, VAL , EXITFLAG ,OUTPUT , LAMBDA ] = quadprog(H, f, A, b);

Model.alpha = ALPHA;
Model.val = VAL;
Model.flag = EXITFLAG;
Model.output = OUTPUT;
Model.lambda = LAMBDA;

end