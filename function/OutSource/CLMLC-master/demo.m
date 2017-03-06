% This is an example program for the paper: 
% 
% L. Sun et al. A scalable clustering-based local multi-label classification method. An ECAI-16 submission. 
%
% The program shows how the CLMLC program (The main function is "CLMLC.m") can be used.
%
% Please type 'help CLMLC' under MATLAB prompt for more information.
%
% The program was developed based on the following packages:
%
% [1] Ridge Regression
% URL: https://github.com/hsuantien/mlc_lsdr
%
% [2] Lite k-means
% URL: http://www.cad.zju.edu.cn/home/dengcai/Data/Clustering.html

%% Make experiments repeatedly
rng(1);

%% Add pathes containing supporting functions
addpath('data','eval');
addpath(genpath('func'));

%% Load a multi-label method and dataset
load('corel5k.mat');

%% Set parameters
opts.d  =  30;
opts.k  =  100;
opts.n  =  5;

%% Perform n-fold cross validation
num_fold = 5; 
indices = crossvalind('Kfold',size(data,1),num_fold);
Results = zeros(5,num_fold);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test;  
    tic; Pre_Labels = CLMLC(data(train,:),target(:,train),data(test,:),opts);
    Results(1,i) = toc;
    [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test));
    Results(2:end,i) = [ExactM,HamS,MacroF1,MicroF1];
end
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

%% Show the experimental results
printmat([meanResults,stdResults],'corel5k','Time ExactM HammingS MacroF1 MicroF1','Mean Std.');