function [Outputs,Pre_Labels]=MLNB(train_data,train_target,test_data,test_target,pca_remained,Smooth)
%MLNB implements the algorithm proposed in [1]
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB(train_data,train_target,test_data,test_target,pca_remained,Smooth)
%
%    Description
%
%       MLNB takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           pca_remained     - The number of remained features after performing principal component analysis
%           Smooth           - The smoothing parameter, default=1
%      and returns,
%           Prior            - A Qx1 array, for the ith class Ci, the prior probability of P(Ci) is stored in Prior(i,1)
%           PriorN           - A Qx1 array, for the ith class Ci, the prior probability of P(~Ci) is stored in PriorN(i,1)
%           mu               - A QxN array, where the mean of the Gaussian distribution for the distribution P(xj|Ci) is stored in mu(i,j)
%           muN              - A QxN array, where the mean of the Gaussian distribution for the distribution P(xj|~Ci) is stored in muN(i,j)
%           sigma            - A QxN array, where the standard deviation of the Gaussian distribution for the distribution P(xj|Ci) is stored in sigma(i,j)
%           sigmaN           - A QxN array, where the standard deviation of the Gaussian distribution for the distribution P(xj|~Ci) is stored in sigmaN(i,j)
%
%[1] M.-L. Zhang, J. M. Pena, V. Robles. Feature selection for multi-label naive bayes classification. Information Sciences, 2009, 179(19): 3218-3229.

    if(nargin<5)
        error('Not enough input parameters, please check again.');
    end
    
    if(nargin<6)
        Smooth=1;
    end
    
    %Performing PCA
    [num_class,num_train]=size(train_target);
    num_test=size(test_data,1);

    all=[train_data;test_data];
    ave=mean(all);
    all=(all'-concur(ave',num_train+num_test))';

    covar=cov(all);
    
    covar=full(covar);

    [u,s,v]=svd(covar);

    t_matrix=u(:,1:pca_remained)';
    all=(t_matrix*all')';

    train_data=all(1:num_train,:);
    test_data=all((num_train+1):(num_train+num_test),:);
    
    %Perfoming GA    
    [num_train,Dim]=size(train_data);
    
    eval_fold=10;
    popsize=20;
    gensize=100;
    
    eval_fold_size=floor(num_train/eval_fold);

    cut_points=[0,eval_fold_size:eval_fold_size:(eval_fold-1)*eval_fold_size,num_train];

    eval_train_data=cell(eval_fold,1);
    eval_train_target=cell(eval_fold,1);
    eval_test_data=cell(eval_fold,1);
    eval_test_target=cell(eval_fold,1);

    eval_Prior=cell(eval_fold,1);
    eval_PriorN=cell(eval_fold,1);
    eval_mu=cell(eval_fold,1);
    eval_muN=cell(eval_fold,1);
    eval_sigma=cell(eval_fold,1);
    eval_sigmaN=cell(eval_fold,1);

    for i=1:eval_fold
        for j=1:cut_points(i)
            eval_train_data{i,1}=[eval_train_data{i,1};train_data(j,:)];
            eval_train_target{i,1}=[eval_train_target{i,1},train_target(:,j)];
        end
        for j=(cut_points(i)+1):cut_points(i+1)
            eval_test_data{i,1}=[eval_test_data{i,1};train_data(j,:)];
            eval_test_target{i,1}=[eval_test_target{i,1},train_target(:,j)];
        end
        for j=(cut_points(i+1)+1):num_train
            eval_train_data{i,1}=[eval_train_data{i,1};train_data(j,:)];
            eval_train_target{i,1}=[eval_train_target{i,1},train_target(:,j)];
        end
        [eval_Prior{i,1},eval_PriorN{i,1},eval_mu{i,1},eval_muN{i,1},eval_sigma{i,1},eval_sigmaN{i,1}]=MLNB_Basic_train(eval_train_data{i,1},eval_train_target{i,1},Smooth);
    end
    
    objfun=@(x) MLNB_fitfunc(x,eval_test_data,eval_test_target,eval_Prior,eval_PriorN,eval_mu,eval_muN,eval_sigma,eval_sigmaN);

    options=gaoptimset('PopulationType','bitstring','PopulationSize',popsize,'Display','iter','StallTimeLimit',Inf,'Generations',gensize);

    [x,fval,reason,output,population,scores]=ga(objfun,Dim,options);
    
    num_retained=sum(x);
    [tempvalue,index]=sort(x,'descend');
    index=index(1:num_retained);
    
    if(isempty(index))
        index=1:Dim;
    end

    [Prior,PriorN,mu,muN,sigma,sigmaN]=MLNB_Basic_train(train_data(:,index),train_target,Smooth);
    [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB_Basic_test(test_data(:,index),test_target,Prior,PriorN,mu,muN,sigma,sigmaN);