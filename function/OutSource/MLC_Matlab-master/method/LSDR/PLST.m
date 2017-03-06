function Yt_pred = PLST(train_data,train_target,test_data,M)
%PLST Principal label space transformation
%   此处显示详细说明

% Parameter setting
lambda = 0.1;
M = round(size(train_target,1)*M);

% CPLST encoding
[Z,Vm,shift] = plst_encode(train_target',M);

% Linear Ridge Regression
ww = ridgereg(Z,train_data,lambda);
Zt_pred = [ones(size(test_data,1),1) test_data] * ww;

% Round-based Decoding
[Yt_pred, ~] = round_linear_decode(Zt_pred, Vm, shift);
  
end
