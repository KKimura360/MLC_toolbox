function [Prior,PriorN,mu,muN,sigma,sigmaN]=MLNB_Basic_train(train_data,train_target,Smooth)
%MLNB_Basic_train trains a multi-label naive bayes classifier
%
%    Syntax
%
%       [Prior,PriorN,mu,muN,sigma,sigmaN]=MLNB_Basic_train(train_data,train_target,Smooth)
%
%    Description
%
%       MLNB_Basic_train takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           Smooth           - Smooth parameter used in estimating Prior and PriorN
%      and returns,
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           mu               - A QxN array, where the mean of the Gaussian distribution for the distribution P(xj|Ci) is stored in mu(i,j)
%           muN              - A QxN array, where the mean of the Gaussian distribution for the distribution P(xj|~Ci) is stored in muN(i,j)
%           sigma            - A QxN array, where the standard deviation of the Gaussian distribution for the distribution P(xj|Ci) is stored in sigma(i,j)
%           sigmaN           - A QxN array, where the standard deviation of the Gaussian distribution for the distribution P(xj|~Ci) is stored in sigmaN(i,j)


    [num_class,num_training]=size(train_target);
    [num_training,Dim]=size(train_data);
    
    for i=1:num_class
        temp_Ci=sum(train_target(i,:)==ones(1,num_training));
        Prior(i,1)=(Smooth+temp_Ci)/(Smooth*2+num_training);
        PriorN(i,1)=1-Prior(i,1);
        
        [temp,index]=sort(train_target(i,:),'descend');
        index1=index(1:temp_Ci);
        index2=index(temp_Ci+1:num_training);
        
        train_data_p=train_data(index1,:);
        train_data_n=train_data(index2,:);
        for j=1:Dim
            [mu(i,j),sigma(i,j)]=normfit(train_data_p(:,j));
            [muN(i,j),sigmaN(i,j)]=normfit(train_data_n(:,j));
        end
    end