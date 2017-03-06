function [Outputs]=BPMLL_test_raw(train_data,train_target,test_data,test_target,net)
%BPMLL_test tests a multi-label neural network.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Threshold,Pre_Labels]=BPMLL_test(train_data,train_target,test_data,test_target,net)
%
%    Description
%
%       BPMLL_test takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           net              - The trained multi-label neural network
%      and returns,
%           HammingLoss      - The hamming loss on testing data as described in [1]
%           RankingLoss      - The ranking loss on testing data as described in [1]
%           OneError         - The one-error on testing data as described in [1]
%           Coverage         - The coverage on testing data as described in [1]
%           Average_Precision- The average precision on testing data as described in [1]
%           Outputs          - The output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Threshold        - The threshold of the ith testing instance for assessing class membership is stored in Threshold(1,i)
%           Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%
%    [1] Schapire R. E., Singer Y. BoosTexter: a boosting based system for text categorization. Machine Learning, 39(2/3): 135-168, 2000.

    [num_class,num_testing]=size(test_target);
    
   % Threshold=get_threshold(train_data,train_target,test_data,net);
    Outputs=sim(net,test_data');
   % Pre_Labels=zeros(num_class,num_testing);
   % for i=1:num_testing
   %    for k=1:num_class
   %         if(Outputs(k,i)>=Threshold(1,i))
   %
   % Pre_Labels(k,i)=1;
   %         else
   %             Pre_Labels(k,i)=-1;
   %         end
   %     end
   % end