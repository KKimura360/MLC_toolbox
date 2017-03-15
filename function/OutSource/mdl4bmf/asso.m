% ASSO - Solves the Discrete Basis problem using the Asso algorithm.
% B = ASSO(D, k, t) returns k basis vectors (rows of B) of data matrix D
% obtained using threshold t. If t is a vector, all of its values are
% tried and the least-error B is returned.
% 
% B = ASSO(D, k, t, b) uses b to profit for covering 1s or penalize for
% covering 0s. b must be of form a/b with a and b integers; covering 1s
% gives a times more profit and covering 0s gives b times more
% penalty. If b is a vector, all of its values are tried and the
% least-error B is returned. Defaults to 1=1/1.
%
% [B, X] = ASSO(D, k, t) returns also matrix X such that D ~ X o B.
%
% [B, X, err] = ASSO(D, k, t) returns also error
% sum(sum(abs(D-min(1,X*B)))).
%
% [B, X, err, opt_t] = ASSO(D, k, t) returns the value of (vector) t that
% gave the smallest error and was used to construct B and X.
%
% [B, X, err, opt_t, opt_b1, opt_b0] = ASSO(D, k, t, b) returns the
% optimal profit (opt_b1) and penalty (opt_b0) from (vector) b
% (opt_b1/opt_b0 belongs to b) used to construct B and X.
%
% When both t and b are vectors, all combinations of their values are
% tried. This can take some time. 
%
% Note that the decomposition is of form X*B, not B*X.




