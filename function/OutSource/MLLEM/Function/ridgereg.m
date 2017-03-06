function ww = ridgereg(Y, X, lambda)
% ridge regression from (X padded with 1) to multiple outputs
% ww = ridgereg(Y, X, lambda) returns ww such that
% ([1, X] * ww) approximates Y
% needs lambda > 0
% See also ridgereg_pinv

  [N, K] = size(Y);
  XX = [ones(N, 1) X]; %pad with 1 in xx_1 so ww_1 corresponds to bias
  ww = ridgereg_pinv(XX, lambda) * Y;
