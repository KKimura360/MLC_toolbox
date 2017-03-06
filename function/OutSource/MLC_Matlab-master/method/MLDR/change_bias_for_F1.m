function [bias_new] = change_bias_for_F1(X, Y, W)
% function [bias_new] = change_bias_for_F1(X, Y, W)
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
for i=1:size(Y,2);
    score =X*W(:,i);
    bias_new(i) = find_offset(score, Y(:,i));
    clear score;
end

function [bias] = find_offset(pred, y)
% This is a script to change the bias of SVM such that certain criteria is optimized. Currently, we only optimize with respect to F1 of each task.
TP = length(find(y==1));
FP = length(find(y==-1));
FN = 0;
TN = 0;

n = length(y);
%pred  = X*w;
[val ind] = sort(pred);

F1(1) = 2*TP/(2*TP+FP+FN); % predict all to be positive
for i = 1:n
    k =  ind(i);
    if y( k ) == 1
      TP = TP - 1;
      FN = FN + 1;
    elseif y( k ) == -1
      FP = FP - 1;
      TN = TN + 1;
    end
    
    if  2*TP + FP + FN ~= 0
      F1(i+1) = 2 * TP / ( 2*TP + FP + FN );
    else 
      F1(i+1) = 0;
    end
end

% find out the maximum F1, index corresponds to val(index-1);
[F1_val,index] = max(F1);

if (index==1) 
  b = val(1);
else
  b = (val(index)+val(index-1))/2;
end

% we change the sign such that the final prediction function is y=wx+b
bias = -b; 
