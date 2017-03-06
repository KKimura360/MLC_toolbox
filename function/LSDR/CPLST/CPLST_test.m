function[conf,time]=CPLST_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by CPLST_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Reference (APA style from google scholar)
% Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%% Method 

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
dim=method.param{1}.dim;
time=cell(dim+1,1);
time{end}=0;
%confidence value for latent labels
tmpconf=zeros(numNt,dim);
% obtain regression result
for label=1:dim
    %obtain confidence value from base classifier model 
    [tmpconf(:,label),method,time{label}]=feval([method.base.name,'_test'],X,Y,Xt,model{label},method);
end
% confidence value for true labels
conf=zeros(numNt,numL);
%decoding 
tmptime=cputime;
Vm=model{end-1};
shift=model{end};
conf = bsxfun(@plus,tmpconf*Vm',shift);
time{end}=cputime-tmptime;