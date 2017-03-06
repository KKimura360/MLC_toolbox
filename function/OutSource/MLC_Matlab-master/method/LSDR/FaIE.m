function Yt_pred = FaIE(train_data,train_target,test_data,M)
% FaIE Feature-aware Implicit Label Space Encoding
%   此处显示详细说明

% Parameter setting
lambda = 0.1;
M = round(size(train_target,1)*M);

% FaIE encoding
[Z,Vm] = FaIE_encode(train_target',M,train_data,lambda);

% % Linear Ridge Regression
% ww = ridgereg(Z,train_data,lambda);
% Zt_pred = [ones(size(test_data,1),1) test_data] * ww;

% % BRridge as baseline classifier
% [~,Zt_pred] = BRridge(train_data,Z',test_data);
% Zt_pred = Zt_pred';

% CCridge as baseline classifier
[~,Zt_pred] = CCridge(train_data,Z',test_data);
Zt_pred = Zt_pred';

% Round-based Decoding
[Yt_pred, ~] = round_linear_decode(Zt_pred, Vm);
  
end

