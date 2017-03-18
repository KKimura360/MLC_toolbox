%%% Sample code

%% Initial Setting 
% load this library
clear all;
addAllpath;
% Seeding
rng('default')

%% Select Dataset with a rate of training/test instances
%Dataset name
dataname='enron';
%number of CV corresponding to the rate of instances 
numCV=10;   %: 80%traning/10%test: 3,5,10 are available

%% Loading Data with cv index 
[data,target,indices]=readData(dataname,numCV);
% if you want to use your seed for the cross validation,
%seed=10
%[data,target,indices]=readData(dataname,numCV,seed)

%% Select methods to use 
%Select method with an order
%if you want to conduct feature selection, clustering and MLC with this
%order, method.func will be set as follows:
%method.func={'FSDR','Clustering','MLC'}
method.name={'CBMLC','Topk'};
method.param=cell(length(method.name),1);
%parameter set
for m= 1:length(method.name)
    %parameter is controlled on SetmethodnameParameter.m See those file
    [method.param{m}]=feval(['Set',method.name{m},'Parameter'],[]);
end

%% BaseClassifier
% base classifier name
%method.base.name='knn';
%method.base.name='ridge';
method.base.name='linear_svm';
% base classifier methodeters 
method.base.param.svmparam='-s 2 -q';
%method.base.param.k=10;
%method.base.param.lambda=10;

%% Thresholding 
% 'Scut','Pcut','Rcut' are availabel, but 'RCut','Pcut' are not implemented
% yet (2017/02/08)
method.th.type='SCut';
method.th.param=0.5;


%% show your selection of methods, datasets and methodeters.
DispSelection;


%Experiments part
for trial=1:1
    index=indices(:,trial);
for fold=1:numCV
    %for test
    if fold==2
       break;
    end
   %Separate dataset into training and test
   test = (index == fold); 
   train = ~test; 
   data=sparse(data);
   X=data(train,:);
   Xt=data(test,:);
   Y=target(:,train')';
   Yt=target(:,test')';
    
   %training (will write a wrapper later)   
   [model,train_time]=MLC_train(X,Y,method);
   %testing 
   [conf,test_time]=MLC_test(X,Y,Xt,model,method);
   %Thresholding
   [pred]=Thresholding(conf,method.th);
   %Evalution
   [res]=Evaluation(Yt,conf,pred);   
end

end
%Visualization
res

filenames={'LASample1.png','LASample2.png','LASample3.png','LASample4.png'};
%getLabelAnalysis(Y,Yt,pred,conf,filenames);

