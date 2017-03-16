function[conf,time]=MIFS_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by PCA_train
%% Output
%conf: confidence values (Nt x L);
%
%% Reference (APA style from google scholar)
% Jian, L., Li, J., Shu, K., & Liu, H. (2016). Multi-label informed feature selection. In 25th International Joint Conference on Artificial Intelligence (pp. 1627-1633).

%% Method 
% Get learned model
id = model{2};

%% Feature selection 
tmpX  = X(:,id);
tmpXt = Xt(:,id);
time=cell(2,1);
time{end}=0;

% Testing
[conf,time{1}] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));