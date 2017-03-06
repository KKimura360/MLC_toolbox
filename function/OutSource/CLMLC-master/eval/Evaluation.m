function [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,test_target)
%EVALUATION Multi-label Evaluation
%
%      Syntax:
% 
%          [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,test_target)
%
%      Input:
%
%          Pre_Labels          L x Nt predicted label matrix           
%          test_target         L x Nt groundtruth label matrix
%
%      Output:
%
%          ExactM              Exact-Match
%          HamS                Hamming-Score
%          MacroF1             Macro-averaged F1 measure
%          MicroF1             Micro-averaged F1 measure

    ExactM = Exact_match(Pre_Labels,test_target);
    HamS = Hamming_score(Pre_Labels,test_target);
    MacroF1 = Macro_F1(test_target,Pre_Labels);  
    MicroF1 = Micro_F1(test_target,Pre_Labels);

end