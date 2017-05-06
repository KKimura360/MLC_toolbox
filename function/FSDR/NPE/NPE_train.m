function[model,time]=NPE_train(X,Y,method)
%NPE: Neighborhood Preserving Embedding
%
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim   : lower-dim of features
%% Output
%model: A learned model of PCA
%
%% Reference (APA style from google scholar)
%He, X., Cai, D., Yan, S., & Zhang, H. J. (2005, October). Neighborhood preserving embedding. In Computer Vision, 2005. ICCV 2005. Tenth IEEE International Conference on (Vol. 2, pp. 1208-1213). IEEE.

%%% Method

%% initialization
dim   = method.param{1}.dim;
gamma = method.param{1}.gamma;
k     = method.param{1}.k;
[numN,numF] = size(X);
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
model   = cell(2,1);
time    = cell(2,1);
tmptime = cputime;

%% Construct an adjacency graph
dotX = dot(X,X,2);
EuD = bsxfun(@plus,dotX,dotX')-2*(X*X');
[~,kNN] = sort(EuD,2);
kNN = kNN(:,2:(k+1));

%% Compute the weights for the graph
W = zeros(numN,numN);
vecOne = ones(k,1);
for i = 1 : numN
    Z = bsxfun(@minus,X(kNN(i,:),:),X(i,:));
    G = Z * Z';
    if rcond(full(G)) < eps
        w = vecOne;
    else
        G = G + eps*trace(G)*speye(k);
        w = G \ vecOne;        
    end
    w = w / sum(w);
    W(i,kNN(i,:)) = w;
end
W = sparse(W);

%% Calcaulate M
M = speye(numN) - W;
M = M' * M;
M = max(M,M');

%% Data centering (It can be commented for large-scale datasets)
X = bsxfun(@minus,X,mean(X,1));

%% Solve the generalize eigenvalue problem
M = speye(numN) - M;
A = full(X'*M*X);
B = full(X'*X + gamma*speye(numF));
A = max(A,A');
B = max(B,B');
[U,~] = eig(A,B);
U = U(:,(numF-dim+1):end);
U = bsxfun(@rdivide,U,sqrt(sum(U.^2,1)));

%% CALL base classfier
X = X * U;
model{2}  = U;
time{end} = cputime-tmptime;
[model{1},time{1}] = feval([method.name{2},'_train'],X,Y,Popmethod(method));