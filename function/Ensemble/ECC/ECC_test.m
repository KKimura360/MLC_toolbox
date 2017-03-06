function[conf, time]=ECC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Option
%method.param.numN  numN% instances are randomly sampled
%method.param.numF  numF% fatures are randomly sampled
%% Reference (APA style from google scholar)
%Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%%% Method
if length(method.name) <2
     warning('on next method, CC must be selected, we use rCC\n')
     method.name{2}='rCC';
     method.param{2}='none';
else
    if strcmpi(method.name{2},'rCC')
        % destructive and re-consider this implementation
        warning('rCC must be selected, we use rCC\n')
        method.name{2}='rCC';
    end
end


%% initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
numM=method.param{1}.numM;
time=cell(numM+1,1);
time{end}=0;
conf=zeros(numNt,numL); 


for i=1:numM
    % recall sampled instances and features
    tmpX=X(model{i+numM},model{i+numM*2});
    tmpY=Y(model{i+numM},:);
    % use sampled fetaures
    tmpXt=Xt(:,model{i+numM*2});
    %Call Next model, rCC 
    [tmpconf,time{i}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{i},Popmethod(method));
    % summation
    tmpconf
    conf=conf+tmpconf;
end
% divide by the total ensemble to obtain ratio
conf=conf./ numM;



