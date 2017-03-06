function[model,time]=ECC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.numM % a number of ensembles
%% Output
%model: A learned model (cell(method.param{x}.numM,1))
%model{1:numM}: classifiers (depends on called method)
%% Option
%method.param.numN  numN% instances are randomly sampled
%method.param.numF  numF% fatures are randomly sampled
%% Reference (APA style from google scholar)
%Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%%% Method
if ~isfield(method.param{1},'numM')
    error('numM, a number of  samples is not set \n');
end
if ~isfield(method.param{1},'numN')
    warning('param.numN is not set, we use all instances');
    method.param{1}.numN=1;
end
if ~isfield(method.param{1},'numF')
    warning('param.numF is not set, we use all features');
    method.param{1}.numF=1;
end

if length(method.name) <2
     warning('rCC must be selected, we use rCC\n')
     method.name{2}='rCC';
     method.param{2}='none';
else
    if ~strcmpi(method.name{2},'rCC')
        % destructive and re-consider this implementation
        warning('rCC must be selected, we use rCC\n')
        method.name{2}='rCC';
    end
end
%% Initialization 
[numN numF]=size(X);
[numNL,numL]=size(Y);
numM=method.param{1}.numM;
model=cell(numM*3,1);
time=cell(numM+1,1);
time{end}=0;


%Learning
if method.param{1}.numN > 1 || method.param{1}.numN <0
    warning('numN is wrong we use all instances');
    method.param{1}.numN=1;
end
if method.param{1}.numF > 1 || method.param{1}.numF <0
    warning('numN is wrong we use all features');
    method.param{1}.numF=1;
end

fprintf('CALL: %s \n',method.name{2});
for i=1:numM
    %Call method
    %sample instances
    numIns= ceil(numN*method.param{1}.numN);
    model{i+numM}=randperm(numN,numIns);
    %sample features
    numFea= ceil(numF*method.param{1}.numF);
    model{i+numM*2}=randperm(numF,numFea);
    % problem transformation
    tmpX=X(model{i+numM},model{i+numM*2});
    tmpY=Y(model{i+numM},:);
    %Call next model
    [model{i}]=feval([method.name{2},'_train'],tmpX,tmpY,Popmethod(method));
end
