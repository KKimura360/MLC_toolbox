function[conf,time]=SLEEC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Bhatia, K., Jain, H., Kar, P., Varma, M., & Jain, P. (2015). Sparse local embeddings for extreme multi-label classification. In Advances in Neural Information Processing Systems (pp. 730-738).
%%% Method
%% SLEEC is a combination method as follows
%% CBMLC (with hierarchical k-means) -> SVP -> knn 
%% Thus, we change method structure 
numLayer = length(method.param)-1;

newmethod.name{1}='SVP';
newmethod.param{1}.dim=method.param{1}.dim;
newmethod.param{1}.numk=method.param{1}.numk1;


%% KNN 
newmethod.name{2}='MLKNN';
newmethod.param{2}.type=2;
newmethod.param{2}.numk=method.param{1}.numk2;

for i=1:numLayer   
    newmethod.name{i+2}=method.name{1+i};
    newmethod.param{i+2}=method.param{1+i};
end
newmethod.base=method.base;
newmethod.th=method.th;

for i=1:numLayer   
    newmethod.name{i+2}=method.name{1+i};
    newmethod.param{i+2}=method.param{1+i};
end
%% Base and Threshold 
newmethod.base=method.base;
newmethod.th=method.th;

[conf,time]=feval([newmethod.name{1},'_test'],X,Y,Xt,model,newmethod);
   


