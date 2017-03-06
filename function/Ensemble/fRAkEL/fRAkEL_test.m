function[conf,time]=fRAkEL_test(X,Y,Xt,model,method)
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
%Keigo Kimura, Mineichi Kudo, Lu Sun and Sadamori Koujaku, "Fast Random k-labelsets for Large-Scale Multi-Label Classification," in Proceedings of the 23rd International Conference on Pattern Recognition (ICPR 2016), Cancun, Mexico. 

%%% Method

if ~isfield(method.param{1},'MLC')
    error('methods for 1st layer is not set, see the detail fo fRAkEL_train')
end

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);

conf=zeros(numNt,numL); 
%appearance of labels
sumLabel=zeros(numNt,numL);
numM=length(model)-2;
labelSet=model{end};
time=cell((numM*2)+1,1);
time{end}=0;

% Re-construct new target
U=zeros(numL,numM);
    for i=1:numM
        U(labelSet{i},i)=1;
    end
Z=Y*U; Z(Z>1)=1;
[tmpconf,time{i+numM}]=feval([method.param{1}.MLC.name{1},'_test'],X,Z,Xt,model{end-1},method.param{1}.MLC);
[tmppred]=Thresholding(tmpconf,method.param{1}.MLC.th);
tmppred=logical(tmppred);
for i=1:numM
    %problem transform for k-nn classifier
    tmpXt=Xt(tmppred(:,i),:);
    tmpY=Y(:,labelSet{i});
    %Call next model
    [tmpconf,time{i}]=feval([method.name{2},'_test'],X,tmpY,tmpXt,model{i},Popmethod(method));
    %substitute the result for sampled label
    conf(tmppred(:,i),labelSet{i})=conf(tmppred(:,i),labelSet{i})+tmpconf;
    % count label appearances
    sumLabel(:,labelSet{i})=sumLabel(:,labelSet{i})+1;
end
conf=conf./ sumLabel;
% if some labels are not sampled, 
conf(isnan(conf))=0;


