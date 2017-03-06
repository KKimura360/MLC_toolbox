function MacroF1 = Macro_F1(test_targets,predict_targets)
%MACROF1 Macro-averaged F1 Measure
%
%      Syntax:
% 
%          MacroF1 = Macro_F1(test_targets,predict_targets)
%
%      Input:
%          
%          test_target         L x Nt groundtruth label matrix
%          predict_targets     L x Nt predicted label matrix 
%
%      Output:
%
%          MacroF1             Macro-averaged F1 Measure

num_label = size(test_targets,1);
test_targets=double(test_targets==1);
predict_targets=double(predict_targets==1);
LabelBasedFmeasure = 0;
for i = 1:num_label
    intersection = test_targets(i,:)*predict_targets(i,:)';
    if sum(predict_targets(i,:))~=0
        precision_i = intersection/sum(predict_targets(i,:));
    else
        precision_i=0;
    end
    if sum(test_targets(i,:))~=0
        recall_i = intersection/sum(test_targets(i,:));
    else
        recall_i=0;
    end
    if recall_i~=0 || precision_i~=0
        LabelBasedFmeasure = LabelBasedFmeasure + 2*recall_i*precision_i/(recall_i+precision_i);
    end
end
MacroF1 = LabelBasedFmeasure/num_label;
    
end