function H = ridgereg_hat(X, lambda)
% hat matrix subject to regulariztion in ridge regression
% needs lambda > 0
% See also ridgereg_pinv

  H = X * ridgereg_pinv(X, lambda);
