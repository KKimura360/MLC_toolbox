function[assign,centroid,metrics]=MPCKmeans(X,M,C,K);
%% Input
% X: Feature Matrix: (NxF)
% M: Must-link Matrix (NxN)
% C: Cannnot-line Matrix (NxN)
% K: Number of Clusters
%% Output
% centroid : centroid matrix (Nxk)
% assign   : assign vector   (vector N)
% metrics  : metrics          (cell(k,1))
% each cell has a matrix     (FxF)
%% Reference

%Diagonalization method is only implemented.

%Initialization
[numN,numF]=size(X);
% for the metric matrices
metrics=cell(K,1);
% to keep distance from centers
dist=cell(K,1);
% to keep -logdet(A)
dets=zeros(K,1);
% to keep what?
maxdist=zeros(K,1);

% here consider only digonal elements
% metrics are initilaized as identity matrices
for k=1:K
    metrics{k}=eye(numF);
    dist{k}=zeros(numN);
end

%Init centroid and assign

%Indicator matrix 
indMat=zeros(numN,K);
for i=1:numN
    indMat(i,assign(i))=1;
end


%Assign step
for k=1:K
    dist{k}=Mahdist(X,X,metrics{k});
    distC{k}=Mahdist(X,centroid,metrics{k});
end


for i=1:numN
    %Calculate distances to centers 
    distToCenter=zeros(K,1);
    %Mahdist returns distance (point x, points y, A)
    for k=1:K
        distToCenter(k)=Mahdist(X(i,:),centroid(k,:),metrics{k});
    end
    
    %Calculation Must-link violations
    mustCost=zeros(K,1);
    mustInd=M(i,:);
    for k=1:K
        clsInd=(indMat(:,k)==0);
        costInd=mustInd .* clsInd;
        tmp=Mahdist(X(i,:),X((costInd~=0),:),metrics{k});
        costInd(costInd==0)=[];
        mustCost(k)= tmp* costInd';
    end
    cannotCost=zeros(K,1);
    cannotInd=C(i,:);
    for k=1:K
        clsInd=(indMat(:,k)==1);
        costInd=cannotInd .*clsInd;
        cannotCost(k)= sum(cannotInd)*maxdist(k) - Mahdist(X,i,costInd,dist{k});
    end
    allCost= distToCenter + mustCost + cannotCost;
    indMat(i,assign(i))=0;
    [~,assign(i)]=min(allCost);
    indMat(i,assign(i))=1;
end

%update centroids
centroid= indMat * X; 
centroid=  diag(1./sum(indMat)) * centroid;    


%update metrics
 constMat=indMat*indMat';
 constMat(constMat==0)=-1;
 constMat=constMat-diag(diag(constMat));
 for k=1:K
    numIns=sum(assign==k);
    tmpX=X(assign==k,:);
    distToCenter=tmpX-repmat(centroid(k,:),numIns,1);
    
    
    
    


