function[conf,time]=BR_test(X,Y,Xt,model,method)
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
numL=length(model);
conf=zeros(numNt,numL);
time=cell(numL+1,1);
time{end}=0;

% classify for each label
for label=1:numL
    [conf(:,label),time{label}]=feval([method.base.name,'_test'],X,Y(:,label),Xt,model{label},method);
end
