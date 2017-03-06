function[model,time]=triClass_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{1}.numK
%% Output
%model: weight matrix
% time: computation time
%% Reference
%Zhang, M. L., Li, Y. K., & Liu, X. Y. (2015, July). Towards Class-Imbalance Aware Multi-Label Learning. In IJCAI (pp. 4041-4047).

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
numK=method.param{1}.numK;
model=cell(numL*2,1);
time=cell(numL+1,1);
time{end}=0;
%Learning model
fprintf('CALL: %s\n',method.base.name);

for label=1:numL
    %candidates for sampling labels
    candLab=setdiff(1:numL,numK);
    %sampling label (index)
    sampledLab=candLab(randperm(numL-1,numK));
    %problem transformation, delete the other labels
    tmpY=Y(:,[label sampledLab]);
    %construct tri-class problem
    indIns=(Y(:,label)>0);
    tmpY(indIns,2:end)=0;    
    %call next method
    [model{label},time{label}]=feval([method.name{2},'_train'],X,tmpY,Popmethod(method));
    model{label+numL}=[label sampledLab];
end
size(model)
