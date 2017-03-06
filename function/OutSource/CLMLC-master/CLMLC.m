function Yt = CLMLC( X,Y,Xt,opts )
%CLMLC Clustering-based Local Multi-Label Classification [1].
%
%    Syntax
%
%       Yt = CLMLC( X,Y,Xt,opts )
%
%    Description
%
%       Input:
%           X           An N x D data matrix, each row denotes a sample
%           Y           An L x N label matrix, each column is a label set
%           Xt          An Nt x D test data matrix, each row is a test sample
%           opts        Parameters for CLMLC
%             opts.d    The size of feature subspace
%             opts.k    The number of data clusters
%             opts.n    The size of a meta-label
% 
%       Output
%           Yt          An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] A scalable clustering-based local multi-label classification method. A ECAI-16 Submission.
 
%% Set parameters
d  = opts.d;
K  = opts.k;
n  = opts.n;

%% Data subspace clustering
[X,Xt,R,C] = CLMLC_cluster(X,Y',Xt,d,K);

%% Local model learning
% Find the nearest cluster for each test instance
[~,Rt] = min(bsxfun(@plus,dot(Xt,Xt,2),dot(C,C,2)')-2*(Xt*C'),[],2); 
Yt = zeros(size(Y,1),size(Xt,1));
for k = 1:K
    local_Rt = Rt==k;
    if all(local_Rt==0)
        continue
    end
    local_R = R==k;
    remove = all(Y(:,local_R)==0,2);
    Yt(~remove,local_Rt) = CLMLC_local(X(local_R,:),...
        Y(~remove,local_R),Xt(local_Rt,:),n);
end

end