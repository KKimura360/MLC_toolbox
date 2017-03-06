
function [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]=ExampleBasedMeasure(test_targets,predict_targets)
% syntax
%   [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]=ExampleBasedMeasure(test_targets,predict_targets)
%
% input
%   test_targets        - L x num_test data matrix of groundtruth labels
%   predict_targets     - L x num_test data matrix of predicted labels
%
% output
%   [ExampleBasedAccuracy,ExampleBasedPrecision,ExampleBasedRecall,ExampleBasedFmeasure]



    [~,num_test]=size(test_targets);
    test_targets=double(test_targets==1);
    predict_targets=double(predict_targets==1);
    
    ExampleBasedAccuracy=0;
    ExampleBasedPrecision=0;
    ExampleBasedRecall=0;
    ExampleBasedFmeasure=0;
    
    for i=1:num_test
        intersection=test_targets(:,i)'*predict_targets(:,i);
        union=sum(or(test_targets(:,i),predict_targets(:,i)));
        
        if union~=0
            ExampleBasedAccuracy=ExampleBasedAccuracy + intersection/union;
        end
        
        if sum(predict_targets(:,i))~=0
            precision_i = intersection/sum(predict_targets(:,i));
        else
            precision_i=0;
        end
        if sum(test_targets(:,i))~=0
            recall_i = intersection/sum(test_targets(:,i));
        else
            recall_i=0;
        end
        ExampleBasedPrecision=ExampleBasedPrecision + precision_i;
        ExampleBasedRecall=ExampleBasedRecall + recall_i;
        if recall_i~=0 || precision_i~=0
            ExampleBasedFmeasure=ExampleBasedFmeasure + 2*recall_i*precision_i/(recall_i+precision_i);
        end
    end
    
    ExampleBasedAccuracy=ExampleBasedAccuracy/num_test;
    ExampleBasedPrecision=ExampleBasedPrecision/num_test;
    ExampleBasedRecall=ExampleBasedRecall/num_test;
    ExampleBasedFmeasure=ExampleBasedFmeasure/num_test;

end