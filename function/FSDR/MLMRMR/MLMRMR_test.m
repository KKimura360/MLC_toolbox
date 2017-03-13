function[conf,time]=MLMRMR_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by MLMRMR_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.

%% Reference (APA style from google scholar)
% Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern analysis and machine intelligence, 27(8), 1226-1238.

%% Method 
% Get learned model
id = model{2};

%% Feature selection 
tmpX  = X(:,id);
tmpXt = Xt(:,id);

% Testing
[conf,time] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));