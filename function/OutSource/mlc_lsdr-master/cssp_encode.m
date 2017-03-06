function [Z, Vm] = cssp_encode(Y, M)
  [N, K] = size(Y);

  %%R stands for right singular vectors
  %%denoted V in the original paper
  [~, ~ , R] = svd(Y, 0);
  Rm = R(:, 1:M);
  p = diag(Rm * Rm') ./ M;
  max_p = max(p);
  
  %%Z stands for the codes to be learned
  %%denoted C in the original paper
  Z = zeros(N, M);
  used = zeros(1, K);
  for m = 1:M
    idx = 1;
    accept = false;
    while ~accept
      idx = floor(rand() * K) + 1;
      accept = (used(idx) == 0 && rand() * max_p <= p(idx));
    end
    used(idx) = 1;
    Z(:, m) = Y(:, idx);
  end    

  %%Vm stands for the decoding matrix
  %%named for consistency with PLST and CPLST  
  Vm = (pinv(Z) * Y)';
end
