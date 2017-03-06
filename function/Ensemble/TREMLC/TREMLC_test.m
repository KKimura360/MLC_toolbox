function[conf, time]=ECC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Nasierding, G., Kouzani, A. Z., & Tsoumakas, G. (2010, December). A triple-random ensemble classification method for mining multi-label data. In Data Mining Workshops (ICDMW), 2010 IEEE International Conference on (pp. 49-56). IEEE.

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
sumLabels=zeros(numNt,numL);


for i=1:numM
    %recall Samplings
    indice=model{i+numM};
    tmpX=X(indice.Ins,indice.Fea);
    tmpY=Y(indice.Ins,indice.Lab);
    % use sampled fetaures
    tmpXt=Xt(:,indice.Fea);
    %Call Next model, rCC 
    [tmpconf,time{i}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{i},Popmethod(method));
    % summation
    conf(:,indice.Lab)=conf(:,indice.Lab)+tmpconf;
    sumLabels(:,indice.Lab)=sumLabels(:,indice.Lab)+1;
end
% divide by the total ensemble to obtain ratio
conf=conf./ sumLabels;
conf(isnan(conf))=0;




