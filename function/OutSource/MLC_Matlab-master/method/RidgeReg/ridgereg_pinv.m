function invX = ridgereg_pinv(X, lambda)
% ``pseudo-inverse'' subject to regulariztion in ridge regression
%   needs lambda > 0

invX = inv(X' * X + lambda * eye(size(X, 2))) * X';
% invX = (X' * X + lambda * eye(size(X, 2))) \ X';
 

