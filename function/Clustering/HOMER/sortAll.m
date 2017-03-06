function [rankDist,rankInd]=sortAll(Dist)

[numL, K]=size(Dist);
rankDist=zeros(numL,K);
rankInd=zeros(numL,K);
for l=1:numL
    [rankDist(l,:),rankInd(l,:)]=sort(Dist(l,:));
end



