function [eval]=Evaluation(Yt,conf,pred)
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

%Multi-label ranking based evaluation criteria
tmp=evalPrecision(conf,Yt,5);
eval.top1=tmp(1);
eval.top3=tmp(3);
eval.top5=tmp(5);

tmp=evalnDCG(conf,Yt,5);
eval.dcg1=tmp(1);
eval.dcg3=tmp(3);
eval.dcg5=tmp(5);
[~,eval.auc]=AVGauc(Yt',conf');

%based on the other implementations, tranpose predictions and ground truth.
Yt=Yt';
pred=pred';
conf=conf';

%exact-match
eval.exact = Exact_match(pred,Yt);
%hamming-loss
eval.hamming = Hamming_score(pred,Yt);
%Label macro-F1
[~,~,~,eval.macroF1]= LabelBasedMeasure(Yt,pred);
%micro-F1
eval.microF1 = MicroFMeasure(Yt,pred);
%coverage
eval.cov= coverage(pred,Yt);
%average precision
eval.pre=Average_precision(pred,Yt);
%ranking-loss
eval.rank=Ranking_score(pred,Yt);
%one-error
eval.one=One_error(pred,Yt);


