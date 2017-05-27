function[conf,time]=RAkEL_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Option
%method.param{x}.vote % not implemented yet for RAkeL++ or something
%% Reference (APA style from google scholar)
%Tsoumakas, G., & Vlahavas, I. (2007, September). Random k-labelsets: An ensemble method for multilabel classification. In European Conference on Machine Learning (pp. 406-417). Springer Berlin Heidelberg.

%% Method

%error check 

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
conf=zeros(numNt,numL); 
sumLabel=zeros(numNt,numL);
numM=length(model)-1;
time=cell(numM+1,1);
time{end}=0;
if  numL <= method.param{1}.numK
    [conf,time]=feval([method.name{2},'_test'],X,Y,Xt,model,Popmethod(method));
    return;
end


for i=1:numM
    % set traind model for ith sample
    tmpmodel=model{i};
    %problem transform for k-nn classifier
    tmpY=Y(:,model{numM+1}{i});
    %Call next model
    [tmpconf,time{i}]=feval([method.name{2},'_test'],X,tmpY,Xt,tmpmodel,Popmethod(method));
    %substitute the result for sampled label
    conf(:,model{numM+1}{i})=conf(:,model{numM+1}{i})+tmpconf;
    % count label appearances
    sumLabel(:,model{numM+1}{i})=sumLabel(:,model{numM+1}{i})+1;
end
conf=conf./ sumLabel;
% if some labels are not sampled, 
conf(isnan(conf))=0;