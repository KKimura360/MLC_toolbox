function[conf,time]=PLST_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by FaIE_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Reference (APA style from google scholar)
% Tai, F., & Lin, H. T. (2012). Multilabel classification with principal label space transformation. Neural Computation, 24(9), 2508-2542.

%%% Method 

%error check 
[numN,numF]=size(X);
[numNL,numL]=size(Y);

%% Initialization
[numNt,~]=size(Xt);
dim=method.param{1}.dim;
%confidence value for latent labels
tmpconf=zeros(numNt,dim);
time=cell(dim+1,1);

% obtain regression result
for label=1:dim
    %obtain confidence value from base classifier model 
    [tmpconf(:,label),method,time{label}]=feval([method.base.name,'_test'],X,Y,Xt,model{label},method);
end
% confidence value for true labels
conf=zeros(numNt,numL);
%decoding 
tmptime=cputime;
[~, conf] = round_linear_decode(tmpconf, model{dim+2},model{dim+3});
time{end}=cputime-tmptime;
