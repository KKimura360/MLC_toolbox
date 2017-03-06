function [assign]=Spectral_Clustering(W,numCls,param)
%% Input
% W: similarity matrix (N x N)
% numCls: number of clusters
%param.SCtype
%% Output
% assign vector  (Nx1 vector)
%% Option
%% Reference 
%Von Luxburg, U. (2007). A tutorial on spectral clustering. Statistics and computing, 17(4), 395-416.

%numbe of instances
numN=size(W,1);
%calculate Laplacian matrix
sumW=sum(W,2);
%D=sparse(diag(sumW));
D=spdiags(sumW,0,numN,numN);
L= D - W;

if ~isfield(param,'SCtype')
    param.SCtype=1;
end

switch param.SCtype
    case 2
        sumW(sumW==0)=1e-08;
        D =spdiags(1./sumW,0,numN,numN);
        L= D * L;
    case 3
        sumW(sumW==0)=1e-08;
        D=spdiags(1./(sumW.^0.5), 0, size(D, 1), size(D, 2));
        L = D * L * D;
end
L = L + eye(numN)* 1e-08;
[U,~]=eigs(L,numCls);

assign=litekmeans(U,numCls);