function [cls] = LRClassify(xTest, w)
% [Input]
% ==============================================================
% xTest : training data (N, M + 1)         ; M = #(feature)
% w : matrix of weights (M + 1, Y)         ; Y = #(label)
% ==============================================================
% [Output] : a label vector (N, 1)

[N M] = size(xTest);
datalength = N;
cls = zeros(N, 1);

% iteration for all test samples
for n = 1 : datalength        
    testSample = xTest(n, :);
    clsprob = zeros(1, 10);        
    % iteration for all labels
    for y = 0 : 9        
        clsprob(y + 1) = sigmoidProb(y, testSample, w);
    end
    cls(n) = find(clsprob == max(clsprob)) - 1;
end

end
