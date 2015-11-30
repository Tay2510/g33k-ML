function p = sigmoidProb(y, x, w)
    
% [Input]
% ==============================================================
% y : label (y = 0 ~ 9)
% x : single row data (1, M + 1)   ; M = #(feature)
% w : matrix of weights (M + 1, Y) ; Y = #(class)
% ==============================================================
% [Output] : probability [0, 1]

probSum = 0;
    
for j = 0 : 8
    probSum = probSum + exp(x*w(:, j + 1));
end
    
if y == 9
    p = 1/(1 + probSum);
else
    p = exp(x*w(:, y + 1))/(1 + probSum);
end


end
