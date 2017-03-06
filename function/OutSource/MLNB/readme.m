%This is an examplar file on how the MLNB program could be used (The main function is "MLNB.m")
%
%Type 'help MLNB' under Matlab prompt for more detailed information


% Loading the file containing the necessary inputs for calling the MLNB function
load('sample data.mat'); 

%Set the number of remained features after PCA
dim=size(train_data,2);
ratio=0.3;
pca_remained=ceil(ratio*dim); % Set the number of reamined features after PCA to 30% of the original dimensionality, as suggested in the literature

% Calling the main function MLNB
[HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=MLNB(train_data,train_target,test_data,test_target,pca_remained);