function [Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data)
%CCRIDGE 此处显示有关此函数的摘要
%   Pre_Labels:  predicted label matrix in L * N

% Ridge parameter
lambda = 0.1;

% Randomly generate a chain order
num_label = size(train_target,1);
chain = randperm(num_label);

% Ridge Regression
pa = [];
% ww = cell(1,num_label);
num_test = size(test_data,1);
Outputs = zeros(num_label,num_test);
Pre_Labels = zeros(num_label,num_test);
for i = chain
    if isempty(pa)
        ww = ridgereg(train_target(i,:)',train_data,lambda);
        Outputs(i,:) = [ones(num_test,1),test_data] * ww;
    else
        ww = ridgereg(train_target(i,:)',[train_data train_target(pa,:)'],lambda);
        Outputs(i,:) = [ones(num_test,1),test_data,Pre_Labels(pa,:)'] * ww;
    end
    Pre_Labels(i,:) = round(Outputs(i,:));
    pa = [pa i];
end

Pre_Labels(Pre_Labels>1) = 1;
Pre_Labels(Pre_Labels<1) = 0;

end

