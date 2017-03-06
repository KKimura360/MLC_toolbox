function [assign, centroid, metric]=MPCKmeans(X,K,M,C)
%%Input
% X: N x F matrix
% K: number of clusters
% M: must-link constraint
% C: cannot-link constraint
%%Output
% assign: Nxk indicator matrix
% centroid: kxF matrix
% metric  : k dimensional cell, each cell contains FxF matrix

%%extract info
[N,F]=size(X);
maxiter=30;

%%Initialization
%Random initialization for centroids
centroidCandidates=randperm(N,K);
centroid=X(centroidCandidates,:);

%For metrics, identity matrix
metric=cell(K,1);
detcell=metric;
for k=1:K
    metric{k}=eye(F);
    detcell{k}=1;
end



%%Repeat until convergence

for iter=1:maxiter
    %assign instances
    %delete past assign
    assign=zeors(N,K);
    for n= 1:N
        best=0;
        bestind=1;
        for k=1:K
            distance=PairwiseDistance(X(n,:),centroid(k,:),metric{k});
            if k==1
             best=ditsance+detcell{k};
             bestind=k;
            end
            if best> distance+detcell{k}
              best=ditsance+detcell{k};
              bestind=k;
            end
        end
        assign(n,bestind)=1;
    end
    %updating centroid
    sumIns=ones(1,N)*assign;
    centroid= (assign * diag(1./sumIns))' * X;
    
    %updateing metric
    %This implementation consideres only diagonal elements
    

end


