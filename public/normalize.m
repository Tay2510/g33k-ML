function [Xnew] = normalize(Xold)
[M N] = size(Xold);
Xnew = zeros(M, N);
for n = 1 : N
    column = Xold(:, n);
    mu = mean(column);
    sigma = sqrt(var(column));    
    Xnew(:, n) = (Xold(:, n) - mu)./sigma;
end
end

