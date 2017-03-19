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
%newmethod.name{1}='CBMLC';
%newmethod.param{1}.dim=method.param{1}.dim;

%% SVP
newmethod.name{1}='SVP';
newmethod.param{1}.dim=method.param{1}.dim;
newmethod.param{1}.numk=method.param{1}.numk1;
newmethod.param{1}.w_thresh=method.param{1}.w_thresh;
newmethod.param{1}.sp_thresh=method.param{1}.sp_thresh;
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


%Call next model
fprintf('CALL: %s\n',newmethod.name{1});
[model,time]=feval([newmethod.name{1},'_train'],X,Y,newmethod);


