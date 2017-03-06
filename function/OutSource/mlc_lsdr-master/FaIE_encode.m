function [Z, Vm] = FaIE_encode(Y, M, X, lambda, alpha)
    if (~exist('alpha', 'var'))
        alpha = 0.1;
    end

    Delta = ridgereg_hat(X, lambda);
    Omega = Y * Y' + alpha * Delta;
    [Z, D] = eigs(Omega, M);
    Vm = (Z' * Y)';
