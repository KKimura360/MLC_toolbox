function[conf,time]=MLJMI_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by MLJMI_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.

%% Reference (APA style from google scholar)
%Sechidis, K., Nikolaou, N., & Brown, G. (2014, August). Information theoretic feature selection in multi-label data through composite likelihood. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 143-152). Springer Berlin Heidelberg.

%%% Method 

%% Get learned model
id = model{2};

%% Feature selection 
tmpX  = X(:,id);
tmpXt = Xt(:,id);

% Testing
[conf,time] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));