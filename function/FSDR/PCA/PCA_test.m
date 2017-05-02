function[conf,time]=PCA_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by PCA_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.

%% Reference (APA style from google scholar)
%Peason, K. (1901). On lines and planes of closest fit to systems of point in space. Philosophical Magazine, 2(11), 559-572.

%%% Method 

%% Get learned model
U = model{2};
time=cell(2,1);
tmptime=cputime;

%% Feature projection
meanX = mean(X,1);
tmpX  = bsxfun(@minus,X,meanX);
tmpXt = bsxfun(@minus,Xt,meanX);
tmpX  = tmpX * U;
tmpXt = tmpXt * U;
time{end}=cputime-tmptime;

%% Testing
[conf,time{1}] = feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));

end