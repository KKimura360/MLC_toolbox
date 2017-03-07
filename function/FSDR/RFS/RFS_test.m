function[conf,time]=RFS_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by PCA_train
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Nie, F., Huang, H., Cai, X., & Ding, C. H. (2010). Efficient and robust feature selection via joint ?2, 1-norms minimization. In Advances in neural information processing systems (pp. 1813-1821).
%READER: Robust Semi-Supervised Multi-Label Dimension Reduction.

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