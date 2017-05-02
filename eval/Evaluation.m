function [evalRes,metList]=Evaluation(Yt,conf,pred,trainT,testT,evalType)
%Yt is a ground truth
%conf has confidence values
%pred has binary values

%eval has:
%eval.hamming  Hammingloss
%eval.exact    ExactMatch
%eval.one      One-error
%eval.cov      coverage
%eval.rank     Ranking loss
%eval.pre      Average precision
%eval.macroF1  macroF1
%eval.microF1  microF1

%eval.top1     Top-1 accuracy
%eval.top3     Top-3 accuracy
%eval.top5     Top-5 accuracy
%eval.dcg1     nDCG@1
%eval.dcg3     nDCG@3
%eval.dcg5     nDCG@5
%eval.auc      label averaged auc

if ~exist('evalType','var')
    evalType = 1;
end

%Multi-label ranking based evaluation criteria
tmp=evalPrecision(conf,Yt,5);
evalRes(1:3) = tmp([1,3,5]);

tmp=evalnDCG(conf,Yt,5);
evalRes(4:6) = tmp([1,3,5]);

evalRes(7) = AVGauc(Yt',conf');

%based on the other implementations, tranpose predictions and ground truth.
Yt=Yt';
pred=pred';
conf=conf';

%exact-match
evalRes(8) = Exact_match(pred,Yt);
%hamming-loss
evalRes(9) = Hamming_score(pred,Yt);
%Label macro-F1
[~,~,~,evalRes(10)]= LabelBasedMeasure(Yt,pred);
%micro-F1
evalRes(11) = MicroFMeasure(Yt,pred);
if evalType == 1
    evalRes(12) = cell_sum(trainT);
    evalRes(13) = cell_sum(testT);   
    metList = 'top1 top3 top5 dcg1 dcg3 dcg5 auc exact hamming macroF1 microF1 trainT testT';
elseif evalType == 2 %traing time
    %coverage
    evalRes(12) = coverage(conf,Yt);
    %average precision
    evalRes(13) = Average_precision(conf,Yt);
    %ranking-loss
    evalRes(14) = Ranking_score(conf,Yt);
    %one-error
    evalRes(15) = One_error(conf,Yt);
    %traing time
    evalRes(16) = cell_sum(trainT);
    %testing time
    evalRes(17) = cell_sum(testT);  
    metList = 'top1 top3 top5 dcg1 dcg3 dcg5 auc exact hamming macroF1 microF1 cov pre rank one trainT testT';    
end

end

% Sum the elements in a nested cell
function s = cell_sum(c)
s = 0;
if isnumeric(c)
    s = c;
else
    for j = 1 : length(c)
        if isnumeric(c{j})
            s = s + c{j};
        else
            s = s + cell_sum(c{j});
        end
    end
end
end