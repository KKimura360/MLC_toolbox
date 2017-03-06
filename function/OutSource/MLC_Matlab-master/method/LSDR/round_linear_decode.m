function [Y_pred, Y_real] = round_linear_decode(Z_pred, Vm, shift)

  N = size(Z_pred, 1);
  K = size(Vm, 1);

  if nargin < 3
    shift = zeros(1, K);
  end

%   Y_real = Z_pred * Vm' + repmat(shift, N, 1);
  Y_real = bsxfun(@plus,Z_pred*Vm',shift);
  Y_pred = round(Y_real');
%   Y_pred = (sign(Y_real) - (Y_real == 0)); %%0 set to -1
end


