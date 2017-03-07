function[model,time]=TREMLC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.numM % a number of ensembles
%method.param{x}.numN  numN% instances are randomly sampled.
%method.param{x}.numF  numF% fatures are randomly sampled.
%method.param{x}.numL  numL% labels are randomly sampled.
%% Output
%model: A learned model (cell(method.param{x}.numM,1))
%model{1:numM}: classifiers (depends on called method)

%% Reference (APA style from google scholar)
%Nasierding, G., Kouzani, A. Z., & Tsoumakas, G. (2010, December). A triple-random ensemble classification method for mining multi-label data. In Data Mining Workshops (ICDMW), 2010 IEEE International Conference on (pp. 49-56). IEEE.

%%% Method
%% Initialization 
[numN numF]=size(X);
[numNL,numL]=size(Y);

% error check
if ~isfield(method.param{1},'numM')
    error('numM, a number of  samples is not set \n');
end
if ~isfield(method.param{1},'numN')
    warning('param.numN is not set, we use all instances');
    method.param{1}.numN=numN;
end
if ~isfield(method.param{1},'numF')
    warning('param.numF is not set, we use all features');
    method.param{1}.numF=numF;
end

if ~isfield(method.param{1},'numL')
    warning('param.numL is not set, we use all features');
    method.param{1}.numL=numL;
end



numM=method.param{1}.numM;
numIns=method.param{1}.numN;
numFea=method.param{1}.numF;
numLab=method.param{1}.numL;

model=cell(numM*2,1);
time=cell(numM+1,1);
time{end}=0;

if ischar(numIns)
    eval(['numIns=',numIns,';']);
    numIns=ceil(numIns);
end
if ischar(numFea)
    eval(['numFea=',numFea,';']);
    numFea=ceil(numFea);
end

if ischar(numLab)
    eval(['numLab=',numLab,';']);
    numLab=ceil(numLab);
end
%Learning

if numIns > numN || numIns <0
    warning('numN is wrong we use all instances');
    numIns=numN;
end

if numFea > numF || numFea <0
    warning('numF is wrong we use all features');
    numFea=numF;
end

if numLab > numL || numLab <0
    warning('numL is wrong we use all features');
    numLab=numL;
end

fprintf('CALL: %s \n',method.name{2});
for i=1:numM
    %sample instances, features and labels in this order
    %sample instances
    indIns=randperm(numN,numIns);
    %transform problem (instance)
    tmpX=X(indIns,:);
    %obtain candidates of features
    candFea=(sum(tmpX)>0);
    candFeaNum=sum(candFea);
    %obtain index representation
    candFea=find(candFea>0);
    %sample features
    %if # candidates of features, 
    if numFea < candFeaNum
        indFea=randperm(candFeaNum,numFea);
        indFea=candFea(indFea);
    else
        indFea=candFea;
    end
    %problem transformation
    tmpX=tmpX(:,indFea);
    
    tmpY=Y(indIns,:);
    %obtain candidates of labels 
    candLab=(sum(tmpY)>0);
    candLabNum=sum(candLab);
    %obtain index representation
    candLab=find(candLab>0);
    %sample labels
    if numLab < candLabNum
        indLab=randperm(candLabNum,numLab);
        indLab=candLab(indLab);
    else
        indLab=candLab;
    end
    % problem transformation  
    tmpY=Y(indIns,indLab);
    indice.Ins=indIns;
    indice.Fea=indFea;
    indice.Lab=indLab;
    model{i+numM}=indice;
    %Call next model
    [model{i}]=feval([method.name{2},'_train'],tmpX,tmpY,Popmethod(method));
end
