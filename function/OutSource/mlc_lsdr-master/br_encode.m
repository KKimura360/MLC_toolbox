function [Z, Vm] = br_encode(Y, M)
  [N, K] = size(Y);
  Vm = speye(K);
  idx = randperm(K);
  Vm = Vm(idx, 1:M);
  Z = Y * Vm;

