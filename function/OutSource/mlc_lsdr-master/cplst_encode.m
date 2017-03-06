function [Z, Vm, shift] = cplst_encode(Y, M, X, lambda)

  shift = mean(Y);

  [N, K] = size(Y);
  Yshift = Y - repmat(shift, N, 1);

  [~, ~, V] = svd(Yshift' * ridgereg_hat(X, lambda) * Yshift, 0);
  Vm = V(:, 1:M);
  Z = Yshift * Vm;

