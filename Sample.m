%%% Sample code

%% Initial Setting 
% load this library
clear
addAllpath
% Seeding
rng('default')

%% Select Dataset with a rate of training/test instances
%Dataset name
dataname='enron';
%number of CV corresponding to the rate of instances 
numCV=5;   %: 80%traning/10%test: 3,5,10 are available

%% Load a dataset with cv index 
[data,target,indices]=read_Data(dataname,numCV);
% if you want to use your seed for the cross validation,
%seed=10
%[data,target,indices]=readData(dataname,numCV,seed)

%% Select methods to use 
%Select methods with an order
%if you want to conduct feature selection, clustering and MLC with this
%order, method.func will be set as follows:
% method.name={'PCA','CBMLC','BR'};  % FSDR => Clustering => MLC
method.name={'BR'};
method.param=cell(length(method.name),1);
%parameter set
for m= 1:length(method.name)
    %parameter is controlled on SetmethodnameParameter.m See those file
    [method.param{m}]=feval(['Set',method.name{m},'Parameter'],[]);
end

%% BaseClassifier
% 1. Support Vector Machines
method.base.name='linear_svm';
method.base.param.svmparam='-s 2 -B 1 -q';
% 2. Ridge Regression
% method.base.name='ridge';
% method.base.param.lambda=1;
% 3. k Nearest Neighbors
% method.base.name='knn';
% method.base.param.k=10;

%% Thresholding 
% 'Scut','Pcut','Rcut' are availabel, but 'RCut','Pcut' are not implemented yet (2017/02/08)
method.th.type='SCut';
method.th.param=0.5;

%% show your selection of methods, datasets and methodeters.
DispSelection;

%% Experiments part
% data=sparse(data);
for trial=1:1
    index=indices(:,trial);
    for fold=1:numCV
        %Separate dataset into training and test
        test = (index == fold);
        train = ~test;
        X=data(train,:);
        Xt=data(test,:);
        Y=target(:,train)';
        Yt=target(:,test)';
        %training
        [model,train_time]=MLC_train(X,Y,method);
        %testing
        [conf,test_time]=MLC_test(X,Y,Xt,model,method);
        %Thresholding
        [pred]=Thresholding(conf,method.th);
        %Evalution
        [tmpRes(:,fold),metList]=Evaluation(Yt,conf,pred,train_time,test_time);
    end
    meanRes = squeeze(mean(tmpRes,2));
    stdRes  = squeeze(std(tmpRes,0,2)/sqrt(numCV));
end
%Visualization
printmat([meanRes,stdRes],'Eval_Res',metList,'Mean Std.');

% filenames={'LASample1.png','LASample2.png','LASample3.png','LASample4.png'};
% getLabelAnalysis(Y,Yt,pred,conf,filenames);