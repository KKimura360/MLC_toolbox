% Add sub-folders containing functions
addpath('data','evaluation');
addpath(genpath('method'));

% methods used in the experiments
methods={'OPLS','HSL','MDDM','DMDDM'};
classfier='BR';
% Load a multi-label dataset
dataset = 'yeast';
load([dataset,'.mat']);

% Set parameters
%num_cluster1 = 5;    % CBMLC
%model1 = @CCridge;   % CBMLC
%ratio = 0.4;         % CPLST
%num_cluster2 = 5;    % CLMLC
%model2 = @metaCC;    % CLMLC
% Make experimental resutls repeatly
rng('default'); 
% Perform n-fold cross validation and obtain evaluation results
num_fold = 5; num_metric = 4; num_method = length(methods);
indices = crossvalind('Kfold',size(data,1),num_fold);
Results = zeros(num_metric+1,num_fold,num_method);
for i = 1:num_fold
    disp(['Fold ',num2str(i)]);
    test = (indices == i); train = ~test; 
      for j=1: length(methods)
          fprintf('Start: %s \n',methods{j});
        tic; [Pre_Labels,~] = DR(data(train,:),target(:,train'),data(test,:),target(:,test'),methods(j),classfier);
        Results(1,i,j) = toc;
        [ExactM,HamS,MacroF1,MicroF1] = Evaluation(Pre_Labels,target(:,test'));
        Results(2:end,i,j) = [ExactM,HamS,MacroF1,MicroF1];
    end
  end
ignore = [];  Results(:,:,ignore) = [];
meanResults = squeeze(mean(Results,2));
stdResults = squeeze(std(Results,0,2) / sqrt(size(Results,2)));

% Save the evaluation results
filename=strcat('results/',dataset,'.mat');
save(filename,'meanResults','stdResults','-mat');

% Show the experimental results
disp(dataset);
disp(meanResults(2:end,:));
figure('Position', [300 300 800 500]);
bar(meanResults(2:end,:));
str1 = {'Exact match';'Hamming Score';'Macro F1';'Micro F1'};
set(gca,'XTickLabel',str1);

xlabel('Metric','FontSize', 14); ylabel('Performance','FontSize', 14);
str2 = methods; 
str2(ignore) = [];
legend(str2,'Location','NorthWest');
hold on;
title([ classfier,':',dataset],'FontSize', 18);
hold off;









