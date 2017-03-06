function[model,time]=MLCEnsemble_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.names is a cell contains MLC methods
%method.param{x}.params is a cell contains parameters of MLC methods
%method.param{x}.bases  is a cell contains paramters of base classifier of
%MLC methods
%method.param{x}.th    is a cell contains paramters of thresholding
%% Output
%model: A learned model (cell(method.param{x}.numM,1))
%model{1:numM}: MLC classifiers (depends on called method)
%% Reference (APA style from google scholar)
%


%% Initialization 
[numN numF]=size(X);
[numNL,numL]=size(Y);
numM=length(method.param{1}.names);
model=cell(numM,1);
time=cell(numM+1,1);
time{end}=0;


fprintf('CALL: %s \n',method.name{2});
for i=1:numM
    tmpmethod.name=method.param{1}.names{i};
    tmpmethod.param=method.param{1}.params{i};
    tmpmethod.base=method.param{1}.bases{i};
    tmpmethod.th=method.param{1}.ths{i};
    [model{i}]=feval([tmpmethod.name{1},'_train'],X,Y,tmpmethod);
end
