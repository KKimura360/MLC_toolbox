function [W]= constructSimMat(X,Y,param)
%% Input
%X: Feature matrix
%Y: Label matrix
%param.sim.type: 'Insk-nn',
%param.sim.
%param.sim.k='';
%% Output
% W (instance x instance matrix)
%% Reference 
% None

%error_check
if ~isfield(param.sim,'type')
    warning('param.sim.type is not set, we use Ins-nn')
    param.sim.type='Ins-nn';
end


switch param.sim.type
    case 'Ins-nn'
        if ~isfield(param.sim,'k')
            warning('param.sim.k is not se we use k=5')
            param.sim.k=5;
        end
        W=adjacency(X,'nn',param.sim.k);
    case {'Lab-nn','k-sets'}
        % k-sets
        % Batzaya Norov-Erdene, Mineichi Kudo, Lu Sun and Keigo Kimura, "Locality in Multi-Label Classification Problems," in Proceedings of the 23rd International Conference on Pattern Recognition (ICPR 2016), Cancun, Mexico.
         if ~isfield(param.sim,'k')
            warning('param.sim.k is not se we use k=5')
            param.sim.k=5;
         end
        W=adjacency(Y,'nn',param.sim.k);
    case 'CLMLC'
        % returns Label Clustering
        W1= 1-pdist(Y','jaccard');
        size(Y'*X)
        label_mean = bsxfun(@rdivide,Y'*X,sum(Y)');
        W2 = exp(-pdist(label_mean));
        W = (W1 + W2) / 2;
        W(isnan(W)) = 0;
        W = sparse(squareform(W));  
    otherwise
        error('we cannot find %s type for construct a similarity matrix',param.sim.type);
end

