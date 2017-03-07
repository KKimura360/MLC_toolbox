function[conf,time]=EPS_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)

%%% Method
%% EPS utlize TREMLC frame work
%% TREMLC -> PS -> some classifier
%% Thus, we change method structure 

fprintf('EPS calls TREMLC and PS');
numLayer = length(method.param)-1;

newmethod.name=cell(numLayer+2,1);
newmethod.param=cell(numLayer+2,1);

%% TREMLC
newmethod.name{1}='TREMLC';
newmethod.param{1}.numF=method.param{1}.numF;
newmethod.param{1}.numN=method.param{1}.numF;
newmethod.param{1}.numL=method.param{1}.numL;
newmethod.param{1}.numM=method.param{1}.numM;

%% PS
newmethod.name{2}='PS';
newmethod.param{2}.type=method.param{1}.type;
newmethod.param{2}.numClass=method.param{1}.numClass;

count=0;
for i=1:numLayer
    if strcmpi(method.name{i+1},'PS')
        continue;
    end
        count=count+1;
        newmethod.name{count+2}=method.name{1+count};
        newmethod.param{count+2}=method.param{1+count};
end
%% Base and Threshold 
newmethod.base=method.base;
newmethod.th=method.th;

[conf,time]=feval([newmethod.name{1},'_test'],X,Y,Xt,model,newmethod);
   


