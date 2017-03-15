function[model,time]=SLEEC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)

%% Reference (APA style from google scholar)
%Bhatia, K., Jain, H., Kar, P., Varma, M., & Jain, P. (2015). Sparse local embeddings for extreme multi-label classification. In Advances in Neural Information Processing Systems (pp. 730-738).
%%% Method
%% SLEEC is a combination method as follows
%% CBMLC (with hierarchical k-means) -> SVP -> knn 
%% Thus, we change method structure 

fprintf('SLEEC calls CBMLC, SVP and KNN');
numLayer = length(method.param)-1;
newmethod.name=cell(numLayer+3,1);
newmethod.param=cell(numLayer+3,1);

%% CBMLC
newmethod.name{1}='CBMLC';
newmethod.param{1}.dim=method.param{1}.dim;

%% CBMLC
newmethod.name{2}='CBMLC';
newmethod.param{2}.ClsMethod=method.param{1}.ClsMethod;
newmethod.param{2}.numCls=method.param{1}.numCls;
switch newmethod.param{2}.ClsMethod
    case 'SC'
        newmethod.param{2}.sim=method.param{1}.sim;
        newmethod.param{2}.SCtype=method.param{1}.SCtype;
end

%% MLCC (Meta-Label Classifier Chain)
newmethod.name{3}='MLCC';
newmethod.param{3}.ClsMethod='SC';
newmethod.param{3}.numCls=method.param{1}.numMeta;
newmethod.param{3}.SCtype=1;
newmethod.param{3}.sim.type='CLMLC';

for i=1:numLayer   
    newmethod.name{i+3}=method.name{1+i};
    newmethod.param{i+3}=method.param{1+i};
end
newmethod.base=method.base;
newmethod.th=method.th;


%Call next model
fprintf('CALL: %s\n',newmethod.name{1});
[model,time]=feval([newmethod.name{1},'_train'],X,Y,newmethod);


