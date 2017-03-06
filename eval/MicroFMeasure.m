
function MicroF1Measure=MicroFMeasure(test_targets,predict_targets)
% syntax
%   MicroF1Measure=MicroFMeasure(test_targets,predict_targets)
%
% Input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels
%
% Output
%   MicroF1Measure

    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    [L,num_test]=size(test_targets);
    groundtruth=reshape(test_targets,1,L*num_test);
    predict=reshape(predict_targets,1,L*num_test);
    intersection=groundtruth*predict';
    precision = intersection/sum(predict);
    recall = intersection/sum(groundtruth);
    MicroF1Measure=2*precision*recall/(precision+recall);
    
end