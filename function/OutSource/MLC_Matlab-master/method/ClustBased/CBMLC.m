function [Pre_Labels,Outputs] = CBMLC(train_data,train_target,test_data,num_cluster,model)
%CBMLC Clustering Based Multi-Label Classification
%   此处显示详细说明

% % kmeans++
% [trainInd,centroid] = kmeanspro(train_data',num_cluster);
% trainInd = trainInd';
% centroid = centroid';

% Lite kmeans
[trainInd,centroid] = litekmeans(train_data,num_cluster,'MaxIter',50);


% % kmeans -- the MATLAB version
% [trainInd,centroid] = kmeans(train_data,num_cluster,...
%     'EmptyAction','singleton','OnlinePhase','off','Display','off');

% % Spectral clustering
% % A = gen_nn_distance(train_data,20,30);
% % trainInd = sc(A,0,num_cluster);
% trainInd = nystrom_no_orth(train_data, 100, 20, num_cluster);
% num_data = size(train_data,1);
% centroid = sparse(trainInd,1:num_data,1,num_cluster,num_data,num_data) * train_data;
% count = zeros(1,num_cluster);
% for i = 1:num_cluster
%     count(i) = size(trainInd(trainInd==i),1);
% end
% centroid = bsxfun(@rdivide,centroid,count');


% [trainInd,centroid] = kmeans([train_data,train_target'],num_cluster,...
%     'EmptyAction','singleton','OnlinePhase','off','Display','off');
% centroid = centroid(:,1:size(train_data,2));

% Find the nearest cluster for each test instance
v1 = dot(test_data,test_data,2); 
v2 = dot(centroid,centroid,2);
D = bsxfun(@plus,v1,v2') - 2*(test_data*centroid');
[~,testInd] = min(D,[],2);  % find the nearest cluster

% Apply local MLC on each cluster 
Pre_Labels = zeros(size(train_target,1),size(test_data,1));
Outputs = zeros(size(train_target,1),size(test_data,1));
if any(isequal(model,@EnMLC))
    percent = [0.8,0.8,1];
    for i = 1:num_cluster
        local_testInd = find(testInd == i);
        if isempty(local_testInd)
            continue
        end
        local_trainInd = find(trainInd == i);
        remove = all(train_target(:,local_trainInd')==0,2);        
        [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = model(train_data(local_trainInd,:),...
            train_target(~remove,local_trainInd'),test_data(local_testInd,:),percent,5,@CCridge);    
%         cluster_train = train_data(local_trainInd,:);
%         cluster_target = train_target(~remove,local_trainInd');
%         cluster_test = test_data(local_testInd,:);
%         alg = 'OPLS';
%         [cluster_train,cluster_test] = DRwrapper(cluster_train,cluster_test,cluster_target,alg);
%         [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = model(cluster_train,...
%             cluster_target,cluster_test,percent,20,@CCridge);
    end
elseif (any(isequal(model,@EMLC)))
    for i = 1:num_cluster
        local_testInd = find(testInd == i);
        if isempty(local_testInd)
            continue
        end
        local_trainInd = find(trainInd == i);        
        remove = all(train_target(:,local_trainInd')==0,2);        
        [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = model(train_data(local_trainInd,:),...
             train_target(~remove,local_trainInd'),test_data(local_testInd,:),3,@CCridge);
        
    end
elseif (any(isequal(model,@DR)))
    for i = 1:num_cluster
        local_testInd = find(testInd == i);
        if isempty(local_testInd)
            continue
        end
        local_trainInd = find(trainInd == i);
        remove = all(train_target(:,local_trainInd')==0,2);       
        cluster_train = train_data(local_trainInd,:);
        cluster_target = train_target(~remove,local_trainInd');
        cluster_test = test_data(local_testInd,:);       
        alg = 'OPLS';
        [cluster_train,cluster_test] = DRwrapper(cluster_train,cluster_test,cluster_target,alg);        
        [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = CCridge(cluster_train,...
            cluster_target,cluster_test);   
    end
elseif (any(isequal(model,@metaCC)))
    for i = 1:num_cluster
        local_testInd = find(testInd == i);
        if isempty(local_testInd)
            continue
        end
        local_trainInd = find(trainInd == i);
        remove = all(train_target(:,local_trainInd')==0,2);
        [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = metaCC(train_data(local_trainInd,:),...
            train_target(~remove,local_trainInd'),test_data(local_testInd,:));
    end
else
    for i = 1:num_cluster
        local_testInd = find(testInd == i);
        if isempty(local_testInd)
            continue
        end
        local_trainInd = find(trainInd == i);        
        remove = all(train_target(:,local_trainInd')==0,2);        
        [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = model(train_data(local_trainInd,:),...
            train_target(~remove,local_trainInd'),test_data(local_testInd,:));
        
%         cluster_train = train_data(local_trainInd,:);
%         cluster_test = test_data(local_testInd,:);
%         cluster_target = train_target(~remove,local_trainInd);
%         [Pre_Labels(~remove,local_testInd'),Outputs(~remove,local_testInd')] = ...
%             model(cluster_train,cluster_target,cluster_test);
%          if ~isempty(find(remove, 1))
%             disp(['Cluster ', num2str(i)]);
%             disp(['Size ', num2str(size(find(~remove),1))]);
%             disp(find(~remove));
%         end

    end
end 

end
