function [evalRes,metList]=Evaluation(Yt,conf,pred,trainT,testT,evalType)
%Yt       ground truth matrix
%conf     confidence matrix
%pred     predicted binary matrix 

%eval has:
%eval.top1     Top-1 accuracy
%eval.top3     Top-3 accuracy
%eval.top5     Top-5 accuracy
%eval.dcg1     nDCG@1
%eval.dcg3     nDCG@3
%eval.dcg5     nDCG@5
%eval.auc      label averaged AUC

%eval.exact    Exact-Match   (1 - Subset 0-1 loss)
%eval.hamming  Hamming-Score (1 - Hamming loss)
%eval.macroF1  macro-averaged F1-score
%eval.microF1  micro-averaged F1-score
%eval.fscore   F1-score
%eval.acc      Accuracy
%eval.rank     Ranking-Score (1 - ranking loss)
%eval.pre      Average precision
%eval.one      One-error
%eval.cov      coverage (Normalized by numL)

%Check the evaluation mode
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
%hamming-score
evalRes(9) = Hamming_score(pred,Yt);
%Label macro-F1
evalRes(10) = Macro_F1(Yt,pred);
%micro-F1
evalRes(11) = Micro_F1(Yt,pred);
%F1-Score
evalRes(12) = FScore(Yt,pred);
%Accuracy
evalRes(13) = Accuracy(Yt,pred);
if evalType == 1
    evalRes(14) = cell_sum(trainT);
    evalRes(15) = cell_sum(testT);   
    metList = 'top1 top3 top5 dcg1 dcg3 dcg5 auc exact hamming macroF1 microF1 fscore acc trainT testT';
elseif evalType == 2 %traing time
    %intermediate results
    [tmpConf,lcell,nlcell,lSize] = RankBased(conf,Yt);
    %average precision
    evalRes(14) = Average_precision(tmpConf,lcell,lSize);
    %ranking-score
    evalRes(15) = Ranking_score(tmpConf,lcell,nlcell,lSize);
    % The smaller the better for the following four metrics
    %one-error
    evalRes(16) = One_error(tmpConf,lcell);
    %coverage
    evalRes(17) = Coverage(conf,Yt);
    %traing time
    evalRes(18) = cell_sum(trainT);
    %testing time
    evalRes(19) = cell_sum(testT);  
    metList = 'top1 top3 top5 dcg1 dcg3 dcg5 auc exact hamming macroF1 microF1 fscore acc pre rank one cov trainT testT';    
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