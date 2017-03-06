function[conf,method,time]=knn_test(X,Y,Xt,model,method)
%X: Feature matrix (NxF)
%Y: Label matrix (NxL)
%method: method.base.param.k is number of nerarest neighbor is needed
%model: not used 
%% Output
%conf: confidence value of test instances for the label (Nt x1 real-value vector)
%method: returns the same param
%time: computation time for the prediction

% number of k-nearest neighbor;
if ~isfield(method.base.param,'k')
    error('k-nn needs number of nearest neighbors');
end
k=method.base.param.k;

% obtain size 
[numN,~]=size(X);
[numNt,~]=size(Xt);
[~,numL]=size(Y);
conf=zeros(numNt,numL);

% if #nearest neighbors is smaller than total number of training instances
if k < numN
    time=cputime;
    % calc L2 distances between training and test instances 
    W=L2_distance(X',Xt'); % W is N x Nt matrix
    %sort with ascending order 
    [~,Ind]=sort(W); 
    for i= 1:numNt
        index=Ind(1:k,i);
        conf(i,:)=sum(Y(index,:));
    end
    % divided by the number of nearest negihbors
    conf = conf ./ k;
    time=cputime-time;
%If #nearest neighbors is larger than total number of tranining instances
else
    time=cputime;
    %use all instances 
    conf=repmat(sum(Y),numNt,1);
    conf = conf./ numNt;
    time=cputime-time;
end