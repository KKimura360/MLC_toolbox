function [assign, centroid] = balancedkmeans(X, K, iter)
%% Input
% X: label matrix (NxL)
% K: number of clusters
%iter: number of iterations
%% Output
% assign: assign vector (vetor L)
% centeroid: centroid matrix (KxN)
%% Reference
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMDÅf08) (pp. 30-44).

%Initialization
X=X';
[numL, numN] = size(X);
centroid = X(randsample(numL,K),:);
assign=zeros(numL,1);
Dist=zeros(numL,K);
Clsind=cell(K,1);
Clsdist=cell(K,1);
bL=ceil(numL/K);

for i= 1:iter
    %Change distance here
    Dist=bsxfun(@plus,dot(X,X,2),dot(centroid,centroid,2)')-2*(X*centroid');
    for target=1:numL
       [rankDist,rankInd]=sortAll(Dist);
      [assign,Dist,Clsdist,Clsind,bL,rankDist,rankInd]=assignLabels(K,target,assign,Dist,Clsdist,Clsind,bL,rankDist,rankInd);
       [centroid]=calcCentroid(X,assign,K);
    end
end
                
            
       







