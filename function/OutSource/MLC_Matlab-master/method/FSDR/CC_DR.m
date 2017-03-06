function [Pre_Labels,Outputs] = CC_DR(train_data,train_target,test_data,opts)
%CC_DR CC with Feature Space Dimension Reduction
%   此处显示详细说明

% Applying supervised FSDR
[train_data,test_data] = DRwrapper(train_data,test_data,train_target,opts);

% Perform Ridge Regression on the encoded data
[Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data);

end


