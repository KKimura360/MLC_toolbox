function [X,Xt,R,C] = CLMLC_cluster( X,Y,Xt,d,K )
%CLMLC_cluster Data subspace clustering for CLMLC.
%
%    Syntax
%
%       [X,Xt,R,C] = CLMLC_cluster( X,Y,Xt,d,K )
%
%    Description
%
%       Input:
%           X       An N x D data matrix, each row denotes a sample
%           Y       An L x N label matrix, each column is a label set
%           Xt      An Nt x D test data matrix, each row is a test sample
%           d       The size of feature subspace
%           K       The number of data clusters
% 
%       Output
%           X       An N x d data matrix, each row denotes a sample
%           Xt      An Nt x d data matrix, each row denotes a sample
%           R       An N x K indicator matrix, each row denotes a sample
%           C       An K x d centroid matrix, each row is a cluster centroid

%% Center the data and target matrix
Xmean = mean(X,1);
X  = bsxfun(@minus,X,Xmean);
Xt = bsxfun(@minus,Xt,Xmean);
Y  = bsxfun(@minus,Y,mean(Y));

%% The two-stage approach for OPLS
% Stage 1: solve the regularized least squares problem
[W1,S1,V1] = svd(X','econ');
r1 = rank(S1);
W1 = W1(:,1:r1); S1 = S1(1:r1,1:r1); V1 = V1(:,1:r1);
s1 = diag(S1);
U1 = W1 * diag(s1./(s1.^2+1)) * V1' * Y;

% Stage 2: solve the resulting optimization problem
H  = Y' * X * U1;
[UH,SH,~] = svd(H,'econ');
rH = min(size(Y,2),d);
UH = UH(:,1:rH); SH = SH(1:rH,1:rH);
sH = diag(SH);
U2 = UH * diag(1./sqrt(sH));
U  = U1 * U2;

%% Encode the data matrix
X  = X * U;
Xt = Xt * U;

%% Apply k-means on the low-dimensional data
[R,C] = litekmeans(X,K,'MaxIter',20);

end