function[conf,time]=triClass_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by BR_train
%% Output
%conf: confidence values (Nt x L);
%time: computation time 
%% Reference
%Zhang, M. L., Li, Y. K., & Liu, X. Y. (2015, July). Towards Class-Imbalance Aware Multi-Label Learning. In IJCAI (pp. 4041-4047).

%% Initialization
[numNt,~]=size(Xt);
numL=length(model)/2;
conf=zeros(numNt,numL);
time=cell(numL+1,1);
time{end}=0;
% classify for each label
for label=1:numL
    labelSet=model{label+numL};
    tmpY=Y(:,labelSet);
    %construct tri-class problem
    indIns=(Y(:,label)>0);
    tmpY(indIns,2:end)=0;    
    [tmpconf,time{label}]=feval([method.name{2},'_test'],X,tmpY,Xt,model{label},Popmethod(method));
    conf(:,label)=tmpconf(:,1); 
end
