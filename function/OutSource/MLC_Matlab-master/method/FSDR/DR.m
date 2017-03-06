function [Pre_Labels,Outputs] = DR(train_data,train_target,test_data,test_target,opts,classifier)
%CC_DR CC with Feature Space Dimension Reduction
%   ¥À¥¶œ‘ æœ?∏Àµ√?

% Applying supervised FSDR
[train_data,test_data] = DRwrapper(train_data,test_data,train_target,opts);

if strcmpi(classifier,'CC')
% Perform Ridge Regression on the encoded data
[Pre_Labels,Outputs] = CCridge(train_data,train_target,test_data);
elseif strcmpi(classifier,'BR')
[Pre_Labels,Outputs] = BRridge(train_data,train_target,test_data);
elseif strcmpi(classifier,'MLKNN')
    Num=10;
    Smooth=1;
   [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
   [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
else
    disp('wrong paramter for classfier')
    return;
end