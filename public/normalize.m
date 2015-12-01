function [Xnew] = normalize(Xold)
[M N] = size(Xold);
Xnew = zeros(M, N);
for n = 1 : N
    column = Xold(:, n);
    Max = max(column);
    Min = min(column);
    d = Max - Min;
    Xnew(:, n) = (Xold(:, n) - Min)./d;
end
end

