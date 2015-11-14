function [p] = sigmoidProb(y, x, w)
    % y : label
    % x : single row data (1, k)
    % w : matrix of weights (k, y)

    probSum = 0
    for i = 1 : 
    prob = 1/(1 + exp(x*w(:, y + 1)));
    if y == 0
        p = 1 - prob;
    else
        p = 1 - prob;
    end
end
