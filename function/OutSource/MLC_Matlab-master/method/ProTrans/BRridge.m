function [Pre_Labels,Outputs] = BRridge(train_data,train_target,test_data)
%BRRIDGE BR with linear Ridge Regression
%   此处显示详细说明

% Ridge parameter
lambda = 0.1;

% Ridge Regression
ww = ridgereg(train_target',train_data,lambda);
Outputs = [ones(size(test_data,1),1) test_data] * ww;
Outputs = Outputs';
Pre_Labels = round(Outputs);

Pre_Labels(Pre_Labels>1) = 1;
Pre_Labels(Pre_Labels<1) = 0;
   
end

