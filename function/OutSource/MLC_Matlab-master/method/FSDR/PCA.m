function [train_data,test_data] = PCA(train_data,test_data,percent)
%PCA Principle Component Analysis
%   此处显示详细说明

% Perform unsupervised PCA on the feature space
[num_data,num_att] = size(train_data);
mean_data = mean(train_data,1);
train_data = bsxfun(@minus, train_data, mean_data);                        
[V,~] = eigs((train_data'*train_data)./(num_data-1),round(percent*num_att));   

% Encoding the training and test data
train_data = train_data * V;
test_data = bsxfun(@minus,test_data,mean_data) * V;  

end

