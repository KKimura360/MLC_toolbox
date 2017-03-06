function [assign, centroid] = randpartition(X, K)
%% Input
% X: label matrix (NxL)
% K: number of clusters
%% Output
% assign: assign vector (vetor L)
% centeroid: centroid matrix (KxN)
%% Reference
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMDÅf08) (pp. 30-44).

%Initialization
X=X';
[numL, numN] = size(X);
%centroid = X(randsample(numL,K),:);
assign=zeros(numL,1);
bL=ceil(numL/K);

%randomly produce paritioning
labelList=randperm(numL);
    for i=1:k
        if i==K 
            assign(labelList(((i-1)*bL)+1:end))=k;
        else
            assign(labelList(((i-1)*bL)+1:(i*bL)))=k;
        end
    end