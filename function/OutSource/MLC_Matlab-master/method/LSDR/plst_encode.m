function [Z, Vm, shift] = plst_encode(Y, M)

  shift = mean(Y);

  [N, K] = size(Y);
  Yshift = Y - repmat(shift, N, 1);

  [~, ~, V] = svd(Yshift, 0);
  Vm = V(:, 1:M);
  Z = Yshift * Vm;
