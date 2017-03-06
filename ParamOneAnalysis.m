%%% Parameter Analysis with one parameter

%% Initial Setting 
addAllpath; rng('default');

%% Datasets to use 
dataNames={'enron'};
numCV=5; numTrial=1; % less than 10


%% Select methods to use
%CAUTION: functionNames is for the plot
functionName={'CBMLC+BR'};

method.name={'CPLST','BR'};
method=SetALLParams(method);
method.base.name='ridge';
%method.base.param.svmparam='-s 2 -q';
method.base.param.lambda=10;
method.th.type='SCut';
method.th.param=0.5;

% changeMethod
change.name='CPLST';
change.param ='dim';
change.value=[5 10 15 20 25];
Result=cell(length(dataNames),1);

for countData=1:length(dataNames)
    dataname=dataNames{countData};
    Result{countData}=cell(length(change.value),1);
    for countParam=1:length(change.value)
        [method]=paramChange(method,change,countParam);
        [res]=conductExpriments(method,numTrial,numCV,dataname);
        Result{countData}{countParam}=res;
    end
end


criteria={'top1','auc','exact','hamming','macroF1','microF1'};
% Visualization of results
    getFigure(Result{1},dataname,change,criteria);

