function [assign, centroid] = balancedkmeans(X, K, iter)
% Conduct the balanced kmeans on the rows of X
%% Input
% X: data matrix (NxF)
% K: number of clusters
%iter: number of iterations
%% Output
% assign: assign vector (N)
% centeroid: centroid matrix (KxN)
%% Reference
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD�f08) (pp. 30-44).

%Initialization
numN = size(X,1);
centroid = X(randsample(numN,K),:);
assign=zeros(numN,1);
Clsind=cell(K,1);
Clsdist=cell(K,1);
bL=ceil(numN/K);

for i= 1:iter
    %Change distance here
    Dist=bsxfun(@plus,dot(X,X,2),dot(centroid,centroid,2)')-2*(X*centroid');
    for target=1:numN
        [rankDist,rankInd]=sortAll(Dist);
        [assign,Dist,Clsdist,Clsind,bL,~,~]=assignLabels(K,target,assign,Dist,Clsdist,Clsind,bL,rankDist,rankInd);
        [centroid]=calcCentroid(X,assign,K);
    end
end

end