function [Pre_Labels,Outputs] = BR_DR(train_data,train_target,test_data,opts)
%BR_HSL BR with supervised Hypergraph Spectral Learning
%   此处显示详细说明

% Applying supervised FSDR
[train_data,test_data] = DRwrapper(train_data,test_data,train_target,opts);

% Perform Ridge Regression on the encoded data
[Pre_Labels,Outputs] = BRridge(train_data,train_target,test_data);

end

