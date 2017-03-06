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

if ~isfield(method.param{1},'numL')
    warning('param.numL is not set, we use all features');
    method.param{1}.numF=1;
end


%% Initialization 
[numN numF]=size(X);
[numNL,numL]=size(Y);
numM=method.param{1}.numM;
model=cell(numM*2,1);
time=cell(numM+1,1);
time{end}=0;


%Learning
if method.param{1}.numN > 1 || method.param{1}.numN <0
    warning('numN is wrong we use all instances');
    method.param{1}.numN=1;
end
if method.param{1}.numF > 1 || method.param{1}.numF <0
    warning('numF is wrong we use all features');
    method.param{1}.numF=1;
end

if method.param{1}.numL > 1 || method.param{1}.numF <0
    warning('numL is wrong we use all features');
    method.param{1}.numL=1;
end

fprintf('CALL: %s \n',method.name{2});
for i=1:numM
    %sample instances, features and labels in this order
    %sample instances
    numIns= ceil(numN*method.param{1}.numN);
    indIns=randperm(numN,numIns);
    %transform problem (instance)
    tmpX=X(indIns,:);
    %obtain candidates of features
    candFea=(sum(tmpX)>0);
    candFeaNum=sum(candFea);
    %obtain index representation
    candFea=find(candFea>0);
    %sample features
    numFea= ceil(numF*method.param{1}.numF);
    %if # candidates of features, 
    if numFea < candFeaNum
        indFea=randperm(candFeaNum,numFea);
        indFea=candFea(indFea);
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
    numLab=ceil(numL*method.param{1}.numL);
    if numLab < candLabNum
        indLab=randperm(candLabNum,numLab);
        indLab=candLab(indLab);
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
