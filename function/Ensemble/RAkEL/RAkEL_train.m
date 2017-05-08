function[model,time]=RAkEL_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.numK: # labels to be sampled 
%method.param{x}.numM: # number of samples
%% Output
%model: A learned model (cell(method.param{x}.numM+1,1))
%model{1:numM}: classifiers (depends on called method)
%model{numM+1}: cell(numM,1), information of sampled labels
%time: computaiton time
%% Option
%method.param{x}.type: 'disjoint', it calls RAkELd
%% Reference (APA style from google scholar)
%Tsoumakas, G., & Vlahavas, I. (2007, September). Random k-labelsets: An ensemble method for multilabel classification. In European Conference on Machine Learning (pp. 406-417). Springer Berlin Heidelberg.

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
%size check
sizeCheck;

%% Initilaization
numM=method.param{1}.numM;
numK=method.param{1}.numK;
time=cell(numM+1,1);
model=cell(numM+1,1);
tmptime=cputime;
%
if numL <=numK
     [model]=feval([method.name{2},'_train'],X,Y,Popmethod(method));
     return;
end

%% Label Sampling
if strcmpi(method.param{1}.type,'disjoint') %RAkELd 
    indList=randperm(numL);
    numM=ceil(numL/numK);
    model=cell(numM+1,1); %% update model
    Labelset=cell(numM,1);
    for i=1:numM
        if i* method.numK > numL
            Labelset{i}=indList(((i-1)*numK)+1:end);
        else
            Labelset{i}=indList(((i-1)*numK)+1:(i*numK));
        end
    end
else %normal RakEL
    Labelset=cell(numM,1);
    for i=1:numM
        Labelset{i}=randsample(numL,numK);
    end
end
model{end}=Labelset;

time{end}=cputime-tmptime;
%Learning
% fprintf('CALL: %s \n',method.name{2});
for i=1:numM
    %Problem transformation
    tmpY=Y(:,Labelset{i});
    %Call method
    [model{i},time{i}]=feval([method.name{2},'_train'],X,tmpY,Popmethod(method));
end
