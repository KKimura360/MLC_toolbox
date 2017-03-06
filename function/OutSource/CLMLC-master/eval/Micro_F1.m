function MicroF1 = Micro_F1(test_targets,predict_targets)
%MICROF1 Micro-averaged F1 Measure
%
%      Syntax:
% 
%          MicroF1 = Micro_F1(test_targets,predict_targets)
%
%      Input:
%          
%          test_target         L x Nt groundtruth label matrix
%          predict_targets     L x Nt predicted label matrix 
%
%      Output:
%
%          MicroF1             Micro-averaged F1 Measure

    test_targets = double(test_targets==1);
    predict_targets = double(predict_targets==1);
    [L,num_test] = size(test_targets);
    groundtruth = reshape(test_targets,1,L*num_test);
    predict=reshape(predict_targets,1,L*num_test);
    intersection = groundtruth*predict';
    precision = intersection/sum(predict);
    recall = intersection/sum(groundtruth);
    MicroF1 = 2*precision*recall/(precision+recall);
    
end