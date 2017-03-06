function [centroid]=calcCentroid(X,assign,K)
centroid=zeros(K,size(X,2));
for k=1:K
    ind=(assign==k);
    centroid(k,:)=mean(X(ind,:));
end
