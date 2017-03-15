function[conf,time]=OSAMBR_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by BR_train
%% Output
%conf: confidence values (Nt x L);
%time: computation time 

%% Initialization
[numNt,~]=size(Xt);
numL=length(model)/2;
conf=zeros(numNt,numL);
time=cell(numL+1,1);
time{end}=0;
% classify for each label
for label=1:numL
    negInd=find(Y(:,label)==0);
    posInd=model{label+numL};
    tmpX=X([posInd; negInd],:);
    tmpY=Y([posInd; negInd],label);
    [conf(:,label),time{label}]=feval([method.base.name,'_test'],tmpX,tmpY,Xt,model{label},method);
end
