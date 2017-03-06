function [Pre_Labels,Outputs] = metaCC(train_data,train_target,test_data)
%METACC Train classifier chains over meta-labels
%   此处显示详细说明

% 0. Check if #label is enough large
num_label = size(train_target,1);
if num_label < 6 
    [Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data);
    return;
end

% % 1. Group labels into meta-labels
% % Label similarity
% A1 = 1 - pdist(train_target,'jaccard');
% 
% % % Instance similarity
% % mean_data = bsxfun(@rdivide,train_target*train_data,sum(train_target,2));
% % A2 = exp(-pdist(mean_data));
% % 
% % % Affinity matrix
% % A = squareform(A1.*round(A2));
% 
% A = squareform(A1);
% 
% % Apply spectral clustering 
% meta_size = 5;
% k = ceil(num_label/meta_size);
% [C, ~, ~] = SpectralClustering(A, k, 2);

meta_size = 5;
k = ceil(num_label/meta_size);
C = litekmeans(train_target,k);

% 2. Build classifier chains over meta-labels
% Perform CC over labels within each meta-label
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);
for i = 1:k 
    meta_target = train_target((C==i),:);
    [Pre_Labels((C==i),:),Outputs((C==i),:)] = CCridge(train_data,meta_target,test_data);
    train_data = [train_data,meta_target'];
    test_data = [test_data,Pre_Labels((C==i),:)'];
end

end

