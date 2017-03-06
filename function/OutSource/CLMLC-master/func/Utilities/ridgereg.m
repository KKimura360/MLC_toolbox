function W = ridgereg(Y, X, lambda)
% ridge regression from (X padded with 1) to multiple outputs
% ww = ridgereg(Y, X, lambda) returns ww such that
% ([1, X] * ww) approximates Y
% needs lambda > 0

  [N, D] = size(X);
  X = [ones(N, 1) X]; 
  W = inv(X'*X + lambda*eye(D+1)) * X' * Y;
  
end