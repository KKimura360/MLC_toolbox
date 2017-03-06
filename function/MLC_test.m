function[conf,time]=MLC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by LP_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it. 

if ~isfield(method,'count')
    method.count=0;
end

[conf,time]=feval([method.name{1},'_test'],X,Y,Xt,model,method);

