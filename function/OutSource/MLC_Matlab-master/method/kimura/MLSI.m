function W = MLSI(X,Y,opts)
% Multi-label informed Latent Semantic Indexing

%%Input 
% X: d x n matrix  data matrix
% Y: l x n matrix  label matrix\
% opts: we use the same structure used in MLDR_v1 provided vy Jieping Ye
%Options for specific for MLSI
% opts.beta: weighing parameter 

%%Output
% W d x k matrix 

if strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr') 
    opts.type='mlsi';
    [W W1 W2]=TwoStage_DR(X,Y,opts);
else
    C= (1-opts.beta) .* (X*X') + opts.beta .*(Y*Y');
    [L W]=eigs((X'*X),C,opts.k);
end

