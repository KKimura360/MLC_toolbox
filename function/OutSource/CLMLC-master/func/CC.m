function Pre_Labels = CC(train_data,train_target,test_data)
%CC Classifier Chains [1] with ridge regression
%
%    Syntax
%
%       Pre_Labels = CC(train_data,train_target,test_data)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
% 
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] J. Read et al. Classifier chains for multi-label classifition. Machine Learning, 2011.

%% Set parameter for ridge regression
lambda = 0.1;

%% Randomly generate a chain order
num_label = size(train_target,1);
chain = randperm(num_label);

%% Form classifier chains
pa = [];
num_test = size(test_data,1);
Pre_Labels = zeros(num_label,num_test);
Ones = ones(num_test,1);
train_target = train_target';
for j = chain
    W = ridgereg(train_target(:,j),[train_data,train_target(:,pa)],lambda);
    temp = round([Ones,test_data,Pre_Labels(pa,:)']*W);
    temp(temp>1) = 1; temp(temp<0) = 0;
    Pre_Labels(j,:) = temp;
    pa = [pa,j];
end

end