function [mat] = Eval_F1(Pre_Labels,test_target)
%EVALUATION with Macro- and Micro-F1
    t=1:9;
    mat=zeros(2,length(t));
    for i=t
        tmp=Pre_Labels;  
        tmp(tmp<i*0.1)=0;
        tmp(tmp>=i*0.1)=1;
        [~,~,~,mat(1,i)]= LabelBasedMeasure(test_target,tmp); 
        mat(2,i) = MicroFMeasure(test_target,tmp);
    end