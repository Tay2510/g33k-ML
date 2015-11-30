function d = EucliDist(xObject, xReference)
delta = xObject - xReference;
d = norm(delta);
end

