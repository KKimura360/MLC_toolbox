%%% Parameter Analysis with two parameter

%% Initial Setting 
addAllpath; rng('default');

%% Datasets to use 
dataNames={'enron'};
numCV=5; numTrial=1; % less than 10


%% Select methods to use
%CAUTION: functionNames is for the plot
functionName={'CPLST+BR'};

method.name={'CPLST','BR'};
method=SetALLParams(method);
method.base.name='ridge';
%method.base.param.svmparam='-s 2 -q';
method.base.param.lambda=10;
method.th.type='SCut';
method.th.param=0.5;

% changeMethod
change.name='CPLST';
change.param ='alpha';
change.value=[0 0.1 1 10];

change2.name='CPLST';
change2.param ='dim';
change2.value=[5 10 20 30];
Result=cell(length(dataNames),1);

for countData=1:length(dataNames)
    dataname=dataNames{countData};
    Result{countData}=cell(length(change.value),1);
    for countParam1=1:length(change.value)
        [method]=paramChange(method,change,countParam1); 
        Result{countParam1}=cell(length(change2.value),1);
        for countParam2=1:length(change2.value);
            [method]=paramChange(method,change,countParam2);
            [res]=conductExpriments(method,numTrial,numCV,dataname);
            Result{countData}{countParam1}{countParam2}=res;
        end 
    end
end


criteria={'top1','auc','exact','hamming','macroF1','microF1'};
% visualize result
for i=1:length(criteria)
    criterion=criteria{i};
    getHeatmap(Result{1},dataname,change,criterion);
end