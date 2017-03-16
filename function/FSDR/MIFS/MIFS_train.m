function [model,time] = MIFS_train(X,Y,method)
%MIFS Multi-Label Informed Feature Selection (MIFS)
%% Input
%X
%Y
%method.param{x}.alpha: weighting parameter
%method.param{x}.dim  : dimension of reduced features
%method.param{x}.iter  : max number of iterations
%% OutPut
% model
%
%% Reference
% Jian, L., Li, J., Shu, K., & Liu, H. (2016). Multi-label informed feature selection. In 25th International Joint Conference on Artificial Intelligence (pp. 1627-1633).

%%% Method 

%% Initialization
[~,numF] = size(X);
[numN,numL] = size(Y);
alpha = method.param{1}.alpha;
beta  = method.param{1}.beta;
gamma = method.param{1}.gamma;
k     = method.param{1}.k;
dim   = method.param{1}.dim;
t_MAX = method.param{1}.tmax;
opt_w = method.param{1}.opt_w;
epsilon = method.param{1}.epsilon;
lambda = method.param{1}.lambda;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end

time=cell(2,1);
tmptime=cputime;

%% Compute the Laplacian matrix
Sx = constructW(X,opt_w);
Dx = diag(sum(Sx));
Lx = Dx - Sx;

%% Initialization
k = round(k * numL);
W = rand(numF,k);
V = rand(numN,k);
B = rand(k,numL);
diag_D = ones(numF,1);

%% The iterative algorithm
lambda_v = lambda;
lambda_w = lambda;
lambda_b = lambda;
for t = 1 : t_MAX
    % Cache some results
    DW = bsxfun(@times,W,gamma*diag_D);
    XW = X*W;    
    % Update V, B, W
    V = V - lambda_v * ( (V-XW) + alpha*(V*B-Y)*B' + beta*Lx*V );
    B = B - lambda_b * V' * (V*B-Y);
    W = W - lambda_w * ( X'*(XW-V) + DW );  
    % Update D
    diag_D = 1 ./ (2 * sqrt(sum(W.^2,2) + epsilon));
end

%% Select the top ranked features
[~, fea_order] = sort(sum(W.^2,2),'descend');
id = fea_order(1:dim);
X  = X(:,id);
time{end}=cputime-tmptime;

%% Return the learned model
model = cell(3,1);
[model{1},time{1}] = feval([method.name{2},'_train'],X,Y,Popmethod(method));
model{2} = id;
model{3} = W;