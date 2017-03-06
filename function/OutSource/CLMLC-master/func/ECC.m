function Pre_Labels = ECC(train_data,train_target,test_data,opts)
%ECC An ensemble of Classifier Chains [1]
%
%    Syntax
%
%       Pre_Labels = ECC(train_data,train_target,test_data,opts)
%
%    Description
%
%       Input:
%           train_data       An N x D data matrix, each row denotes a sample
%           train_target     An L x N label matrix, each column is a label set
%           test_data        An Nt x D test data matrix, each row is a test sample
%           opts             Parameters for metaCC
%             opts.m         Size of the ensemble
%             opts.fea_per   Percentage of features for random selection
%             opts.ins_per   Percentage of instances for random selection
% 
%       Output:
%           Pre_Labels       An L x Nt predicted label matrix, each column is a predicted label set
%
%  [1] J. Read et al. Classifier chains for multi-label classifition. Machine Learning, 2011.

%% Set parameters
m = opts.m;
fea_per =   opts.fea_per;
ins_per =   opts.ins_per;

%% Get the size of sub-problem
[num_ins,num_fea] = size(train_data);
D = round(num_fea*ins_per);
N = round(num_ins*fea_per);

%% Build the ensemble of classifiers
Outputs = zeros(size(train_target,1),size(test_data,1));
for i = 1:m
    % generate the random number
    idx_fea = randperm(num_fea);
    idx_ins = randperm(num_ins);
    % Find the subsets of features and instances
    sub_train  = train_data(idx_ins(1:N),idx_fea(1:D));
    sub_target = train_target(:,idx_ins(1:N));
    sub_test   = test_data(:,idx_fea(1:D));
    % Build the classifiers in the subspace
    Temp_Labels = CC(sub_train,sub_target,sub_test);
    Outputs = Outputs + Temp_Labels;
end

Outputs = Outputs ./ m;
Pre_Labels = round(Outputs);

end