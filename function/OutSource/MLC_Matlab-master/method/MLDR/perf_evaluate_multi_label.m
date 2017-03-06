function perf = perf_evaluate_multi_label(X,Y,W,b)
% perf = perf_evaluate_multi_label(X,Y,W,b)
% evaluate the performance of multi_label classification
% The last bias parameter is optional.
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
  
  [n] = size(X,1);
  k = size(Y,2);

  if nargin < 4
    score = X*W;
  else   
    b_new(1,:) = b;
    score = X*W+repmat(b_new,n,1);
  end
  
  pred = sign(score);
  acc = length(find(sum((pred  .*  Y), 2)  == k)) / n;

  AUC_ind = zeros(k,1);
  acc_balanced = zeros(k,1);
  [ Precision  Recall  F1 micro_F1   ] = compute_stat( pred, Y ) ; 

for i = 1 : k
    acc_balanced(i) = sum(pred(:, i)==Y(:,i))/n;
    [uparea AUC_ind(i)] = roc(score(:,i), Y(:,i));
end
AUC =mean(AUC_ind);

perf.acc = acc;
perf.acc_balanced = mean(acc_balanced);
perf.AUC = AUC;
perf.F1 = mean(F1);
perf.micro_F1 = micro_F1;
perf.precision = mean(Precision);
perf.recall = mean(Recall);

perf.AUC_ind = AUC_ind;
perf.F1_ind = F1;
perf.precision_ind = Precision;
perf.recall_ind = Recall;
perf.acc_balanced_ind = acc_balanced;


% a function for calculate varios measures, this is different from
% multi-class
function [ Precision Recall F1 micro_F1 ] = compute_stat( pred , Y ) 

k = size(pred,2);

for i = 1:k
    ind_pos = ( find(Y(:,i)==1) );
    ind_neg = ( find(Y(:,i)== -1) );
    
    TP(i) = length( find( pred(ind_pos,i) == 1 ) );
    FN(i) = length(ind_pos) - TP(i);
    
    FP(i) = length( find( pred(ind_neg,i) == 1 ) );
    TN(i) = length( ind_neg ) - FP(i);
    
    if TP(i) + FP(i) ~= 0
        Precision(i) = TP(i) / ( TP(i) + FP(i) ) ;
    else
        %count = count - 1;
        Precision(i) = 0;
    end
    
    if TP(i) + FN(i) ~=0
        Recall(i) = TP(i) / (TP(i) + FN(i) );
    else 
        %count = count - 1;
        Recall(i) = 0;
    end
    
    if  2*TP(i) + FP(i) + FN(i) ~= 0
        F1(i) = 2 * TP(i) / ( 2*TP(i) + FP(i) + FN(i) );
    else 
        %count = count - 1;
        F1(i) = 0;
    end
    
end

% calculate micro-F1
micro_F1 = 2 * sum(TP) / (2 * sum(TP) + sum(FP) + sum(FN));

function [up_area down_area] = roc(pred,y)

% pred denotes the vector with prediction value
% y denotes the vector with data label

[val ind] = sort(pred);


TP = length(find(y==1));
FP = length(find(y==-1));
FN = 0;
TN = 0;


for i = 1:length(pred)
    k =  ind(i);
    if y( k ) == 1
        TP = TP - 1;
        FN = FN + 1;
        
    else if y( k ) == -1
            FP = FP - 1;
            TN = TN + 1;
         end
    end
    
    TPR( i ) = TP / (TP + FN);
    FPR( i ) = FP / (FP + TN);
    
end

TPR = [1 TPR];  
TPR = TPR(length(TPR):-1:1);

FPR = [1 FPR];
FPR = FPR(length(FPR):-1:1);

up_area = 0;
down_area = 0;
for i = 2 : length(pred)+1
    up_area = up_area + (TPR(i) - TPR(i-1))  * FPR(i);
    down_area = down_area + (FPR(i) - FPR(i-1))  * TPR(i);
end


