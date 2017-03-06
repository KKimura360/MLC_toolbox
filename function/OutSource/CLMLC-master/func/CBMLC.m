function Pre_Labels = CBMLC(train_data,train_target,test_data,opts)
%CBMLC Clustering-Based Multi-Label Classification [1]
%
%    Syntax
%
%       Pre_Labels = CBMLC(train_data,train_target,test_data,opts)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           opts             Parameters for metaCC
%             opts.k         The number of data cluster
% 
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] G. Nasierding et al. Clustering-based multi-label classification for image annotation and retrieval. ICSMC, 2009.

%% Set parameters
K = opts.k;

%% Apply kmeans on input data
[R,C] = litekmeans(train_data,K,'MaxIter',20);

%% Apply local MLC on each cluster 
% Find the nearest cluster for each test instance
[~,Rt] = min(bsxfun(@plus,dot(test_data,test_data,2),dot(C,C,2)')-2*(test_data*C'),[],2);  
Pre_Labels = zeros(size(train_target,1),size(test_data,1));
for k = 1:K
    local_Rt = Rt==k;
    if all(local_Rt==0)
        continue
    end
    local_R = R==k;
    remove = all(train_target(:,local_R)==0,2);
    Pre_Labels(~remove,local_Rt) = CC(train_data(local_R,:),...
        train_target(~remove,local_R),test_data(local_Rt,:));
end

end