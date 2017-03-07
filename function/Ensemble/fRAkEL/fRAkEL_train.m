function[model,time]=fRAkEL_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.numK: # labels to be sampled 
%method.param{x}.numM: # number of samples
%method.param{x}.MLC: MLC method for the 1st layer
    %method.param{x}.MLC.name{'method','method'}
    %method.param{x}.MLC.param{x}
    %method.param{x}.MLC.base
    %method.param{x}.MLC.th
    %DONOT use fRAkEL for this an ifinite loops appears ><
%% Output
%model: A learned model (cell(method.param{x}.2*numM+1,1))
%model{1:2*numM}: classifiers (depends on called method)
%model{2*numM+1}: cell(2*numM,1), information of sampled labels
%% Option
%method.param{x}.type: 'disjoint', it calls RAkELd
%% Reference (APA style from google scholar)
%Keigo Kimura, Mineichi Kudo, Lu Sun and Sadamori Koujaku, "Fast Random k-labelsets for Large-Scale Multi-Label Classification," in Proceedings of the 23rd International Conference on Pattern Recognition (ICPR 2016), Cancun, Mexico. 

%%% Method

[numN numF]=size(X);
[numNL,numL]=size(Y);

%error check 
if ~isfield(method.param{1},'numK')
    warning('numK, a number of labels to sample is not set \n we use numK=3\n');
   method.param{1}.numK=3;
end
if ~isfield(method.param{1},'numM')
    warning('numM, a number of  samples is not set \n we use numM=2L\n');
    method.param{1}.numM=2*numL;
end

if ~isfield(method.param{1},'type')
    method.param{1}.type='normal';
end

if ~isfield(method.param{1},'MLC')
    error('methods for 1st layer is not set, see the detail fo fRAkEL_train')
end


%% Initilization
numM=method.param{1}.numM;
if ischar(numM)
    eval(['numM=',method.param{1}.numM]);
    numM=ceil(numM);
end
numK=method.param{1}.numK;
model=cell(numM+2,1);
time=cell(numM+2);
tmptime=cputime;
%% Label Sampling
if strcmpi(method.param{1}.type,'disjoint') %RAkELd 
    indList=randperm(numL);
    numM=ceil(numL/numK);
    model=cell((2*numM+1),1); % update size
    labelSet=cell(numM,1);
    U=zeros(numL,numM);
    for i=1:numM
        if i* umK > numL
            labelSet{i}=indList(((i-1)*numK)+1:end);
        else
            labelSet{i}=indList(((i-1)*numK)+1:(i*numK));
        end
        U(labelSet{i},i)=1;
    end
else %normal RakEL
    labelSet=cell(numM,1);
    U=zeros(numL,numM);
    for i=1:numM
        labelSet{i}=randsample(numL,numK);
        U(labelSet{i},i)=1;
    end
end
model{end}=labelSet;
time{end}=cputime-tmptime;
%Learning model for the sampled labelsets 
fprintf('CALL: %s \n',method.name{2});
tmptime=cputime;
%New target matrix
Z=Y*U;
Z(Z>0)=1;
% Call method method.param{1}.MLC
%this MLC has the same structure of method (MLC.name, MLC.param)
[model{numM+1}]=feval([method.param{1}.MLC.name{1},'_train'],X,Z,method.param{1}.MLC);
time{end-1}=cputime-tmptime;
for i=1:numM
    %Problem transformation
    Z= Y * U(:,i);
    Z(Z>0)=1;
    Z=logical(Z);
    tmpX=X(Z,:);
    tmpY=Y(Z,labelSet{i});
    %Call method
    [model{i},time{i}]=feval([method.name{2},'_train'],tmpX,tmpY,Popmethod(method));
end
