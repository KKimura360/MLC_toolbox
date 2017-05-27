function [evalRes,metList]=Evaluation(Yt,conf,pred,trainT,testT,evalType)
%Yt       ground truth matrix
%conf     confidence matrix
%pred     predicted binary matrix 
%evalType {1,2,3}. 1: Binary output; 2: Probabilistic output; 3: More results

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
if nnz(conf-pred)==0 
    evalType = 1;   % Only binary output
elseif ~(exist('evalType','var') && ismember(evalType,[1,2,3]))
    evalType = 2;
end

%Multi-label ranking based evaluation criteria
if evalType ~= 1
    % Precision @ k
    tmp=evalPrecision(conf,Yt,5);
    evalRes(1:3) = tmp([1,3,5]);
    % nDCG @ k
    tmp=evalnDCG(conf,Yt,5);
    evalRes(4:6) = tmp([1,3,5]);
    % Area Under the Curve
    evalRes(7) = AVGauc(Yt',conf');
end

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

switch evalType
    case 1
        evalRes(14) = cell_sum(trainT);
        evalRes(15) = cell_sum(testT);
        evalRes(1:7) = [];
        metList = {'exact','hamming','macroF1','microF1','fscore','acc','trainT','testT'};
    case 2
        evalRes(14) = cell_sum(trainT);
        evalRes(15) = cell_sum(testT);
        metList = {'top1','top3','top5','dcg1','dcg3','dcg5','auc','exact','hamming','macroF1','microF1','fscore','acc','trainT','testT'};
    case 3  % increase the time cose
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
        metList = {'top1','top3','top5','dcg1','dcg3','dcg5','auc','exact','hamming','macroF1','microF1','fscore','acc','pre','rank','one','cov','trainT','testT'};
end

end

function s = cell_sum(c)
% Sum the elements in a nested cell

s = 0;
if isnumeric(c)
    s = c;
else
    for j = 1 : length(c)
        if isempty(c{j})
            continue
        elseif isnumeric(c{j})
            s = s + c{j};
        elseif iscell(c{j})
            s = s + cell_sum(c{j});
        end
    end
end
end