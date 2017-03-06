function [U]=updateU(X,U,V)
U= U.* (X*V) ./ (U *(V'*V)+1e-12);
end