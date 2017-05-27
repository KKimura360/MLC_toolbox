function [assign, centroid] = randpartition(X, K)
% Randomly partition the rows of X into K disjoint parts
%% Input
% X: data matrix (NxF)
% K: number of clusters
%% Output
% assign: assign vector (N)
% centeroid: centroid matrix (KxN)
%% Reference
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD�f08) (pp. 30-44).

%Initialization
numN = size(X,1);
assign=zeros(numN,1);
bL=ceil(numN/K);

%randomly produce paritioning
labelList=randperm(numN);
for i=1:K
    if i==K
        assign(labelList(((i-1)*bL)+1:end))=K;
    else
        assign(labelList(((i-1)*bL)+1:(i*bL)))=i;
    end
end
[centroid]=calcCentroid(X,assign,K);

end