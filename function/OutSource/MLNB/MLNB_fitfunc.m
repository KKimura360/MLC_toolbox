function fitness=MLNB_fitfunc(x,test_data,test_target,Prior,PriorN,mu,muN,sigma,sigmaN)
%MLNB_fitfunc returns a fitness value for a specific attribute selector x.
%
%    Syntax
%
%       fitness=MLNB_fitfunc(x,test_data,test_target,Prior,PriorN,mu,muN,sigma,sigmaN)
%
%    Description
%
%       MLNB_fitness takes,
%           x                - A 1xN binary vector, where if the ith component equals 1, then the ith attribute is retained, othwise discarded
%           test_data        - An Mx1 cell, the ith test set is stored in test_data{i,1}
%           test_target      - An Mx1 cell, the test target of the ith test set is stored in test_target{i,1}
%           Prior            - An Mx1 cell, the Prior parameter as used in MLNB_Basic_test of the ith test set is stored in Prior{i,1}
%           PriorN           - An Mx1 cell, the PriorN parameter as used in MLNB_Basic_test of the ith test set is stored in PriorN{i,1}
%           mu               - An Mx1 cell, the mu parameter as used in MLNB_Basic_test of the ith test set is stored in mu{i,1}
%           muN              - An Mx1 cell, the muN parameter as used in MLNB_Basic_test of the ith test set is stored in muN{i,1}
%           sigma            - An Mx1 cell, the sigma parameter as used in MLNB_Basic_test of the ith test set is stored in sigma{i,1}
%           sigmaN           - An Mx1 cell, the sigmaN parameter as used in MLNB_Basic_test of the ith test set is stored in sigmaN{i,1}
%      and returns,
%           fitness          - The returned fitness value

    num_retained=sum(x);
    [tempvalue,index]=sort(x,'descend');
    index=index(1:num_retained);
    
    num_dataset=size(test_data,1);
    
    fitness=0;
    
    for i=1:num_dataset
        [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB_Basic_test(test_data{i,1}(:,index),test_target{i,1},Prior{i,1},PriorN{i,1},mu{i,1}(:,index),muN{i,1}(:,index),sigma{i,1}(:,index),sigmaN{i,1}(:,index));
        fitness=fitness+(HammingLoss+RankingLoss)/2;
    end
    
    fitness=fitness/num_dataset;