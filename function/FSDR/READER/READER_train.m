function [model] = READER_train( Xw,X,Y,method )
%READER Robust Semi-Supervised Multi-Label Dimension Reduction [1]
%
%    Syntax
%
%       Yt = READER( Xw,X,Y,method )
%
%    Description
%
%       Input:
%           Xw           An N x M data matrix, each row denotes a sample
%           X            An n x M data matrix, each row denotes a sample (n < N)
%           Y            An L x n label matrix, each column is a label set
%           opts         Parameters for READER
%             opts.k     The dimensionality of embedded label space
%             opts.p     The number of nearest neighbors
%             opts.alpha The factor on the projection matrix
%             opts.beta  The factor on manifold learning of Xw
%             opts.gamma The factor on manifold learning of Y
% 
%       Output
%           Yt           An L x Nt predicted label matrix, each column is a predicted label set
%
%READER: Robust Semi-Supervised Multi-Label Dimension Reduction.

%% Get the parameters
alpha = method.param{1}.alpha;
beta  = method.param{1}.beta;
gamma = method.param{1}.gamma;
k     = method.param{1}.k;
p     = method.param{1}.p;
dim   = method.param{1}.dim;
MaxIt = method.param{1}.MaxIt;

%% Compute the Laplacian matrix
% Build the weighted graph for X 
opt_x.NeighborMode = 'KNN';
opt_x.k = p;
opt_x.WeightMode = 'HeatKernel';
Sx = constructW(Xw,opt_x);
Lx = diag(sum(Sx)) - Sx;
% Build the weighted graph for X
if k < 1
    opt_y.NeighborMode = 'KNN';
    opt_y.k = p;
    opt_y.WeightMode = 'Cosine';
    Sy = constructW(Y',opt_y);
    Ly = diag(sum(Sy)) - Sy;
end

%% Absorb the bias b into X
Xw = [Xw, ones(size(Xw,1),1)];
X  = [X, ones(size(X,1),1)];

%% The alterating algorithm
L        = size(Y,1);
[n,M]    = size(X);
XLX      = beta*Xw'*Lx*Xw;
alpha_it = alpha.^2;
diag_Hn  = ones(n,1);
diag_Hm  = ones(M,1);
% XTX = X'*X;
if opts.k == 1
    W   = rand(M,L);
    for t = 1:MaxIt
        % Get Hn and Hm
        Hm  = diag(alpha_it*diag_Hm);
        HnX = bsxfun(@times,X,diag_Hn);
        
        % Update W
        W = (HnX'*X + Hm + XLX) \ (HnX' * Y');
        
        % Update H
        diag_Hn = 0.5 ./ sqrt(sum((X*W-Y').^2,2) + eps);
        diag_Hm = 0.5 ./ sqrt(sum((alpha*W).^2,2) + eps);
    end
else
    k       = round(k * L);
    W       = rand(M,k);  
    Ly_it   = gamma * Ly;
    for t = 1:MaxIt
        % Get Hn and Hm
        Hm  = diag(alpha_it*diag_Hm);
        Hn  = diag(diag_Hn);
        HnX = bsxfun(@times,X,diag_Hn);
        
        % Updata V
        V = sparse(Hn + Ly_it) \ (HnX * W);
        
        % Update W
        W = (HnX'*X + Hm + XLX) \ (HnX' * V);
        
        % Update H
        diag_Hn = 0.5 ./ sqrt(sum((X*W-V).^2,2) + eps);
        diag_Hm = 0.5 ./ sqrt(sum((alpha*W).^2,2) + eps);
    end
end

%% Remove the first row of W (corresponding to b)
X = X(:,1:(end-1));
W = W(1:(end-1),:);


%% Feature selection or Sparse linear projection
[~, fea_order] = sort(sum(W.^2,2),'descend');
id = fea_order(1:dim);
tmpX  = X(:,id);

%% Return the learned model
[model{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));
model{2} = id;
    
end