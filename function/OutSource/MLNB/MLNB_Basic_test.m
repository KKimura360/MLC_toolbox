function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB_Basic_test(test_data,test_target,Prior,PriorN,mu,muN,sigma,sigmaN)
%MLNB_Basic_test tests a multi-label naive bayesian classifier.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB_Basic_test(test_data,test_target,Prior,PriorN,mu,muN,sigma,sigmaN)
%
%    Description
%
%       MLNB_Basic_test takes,
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           mu               - A QxN array, where the mean of the Gaussian distribution for the distribution P(xi|Ci) is stored in mu(1,i)
%           muN              - A QxN array, where the mean of the Gaussian distribution for the distribution P(xi|~Ci) is stored in muN(1,i)
%           sigma            - A QxN array, where the standard deviation of the Gaussian distribution for the the distribution P(xi|Ci) is stored in sigma(1,i)
%           sigmaN           - A QxN array, where the standard deviation of the Gaussian distribution for the the distribution P(xi|~Ci) is stored in sigmaN(1,i)
%      and returns,
%           HammingLoss      - The hamming loss on testing data
%           RankingLoss      - The ranking loss on testing data
%           OneError         - The one-error on testing data as
%           Coverage         - The coverage on testing data as
%           Average_Precision- The average precision on testing data
%           Outputs          - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1

    [num_class,num_testing]=size(test_target);
    [num_testing,Dim]=size(test_data);
    
    Outputs=zeros(num_class,num_testing);    
    
    for i=1:num_testing
        temp_in=log(normpdf(concur(test_data(i,:)',num_class)',mu,sigma)+1e-20*ones(num_class,Dim));
        temp_out=log(normpdf(concur(test_data(i,:)',num_class)',muN,sigmaN)+1e-20*ones(num_class,Dim));
        for j=1:num_class
            Prob_in=log(Prior(j))+sum(temp_in(j,:));
            Prob_out=log(PriorN(j))+sum(temp_out(j,:));
            Outputs(j,i)=1/(1+exp(Prob_out-Prob_in));
        end
    end
    
%Evaluation
    Pre_Labels=zeros(num_class,num_testing);
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>=0.5)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    
    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);