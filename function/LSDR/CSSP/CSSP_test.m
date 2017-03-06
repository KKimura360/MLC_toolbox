function[conf,time]=CSSP_test(X,Y,Xt,model,method)
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
% Hsu, D. J., Kakade, S., Langford, J., & Zhang, T. (2009, December). Multi-label prediction via compressed sensing. In NIPS (Vol. 22, pp. 772-780).

%% Method 
%error check 
%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
% reduced dim of labels
dim=method.param{1}.dim;
%confidence value for latent labels
tmpconf=zeros(numNt,dim);
time=cell(dim+1,1);
time{end}=0;

% obtain regression result
for label=1:dim
    %obtain confidence value from base classifier model 
    [tmpconf(:,label),method]=feval([method.base.name,'_test'],X,Y,Xt,model{label},method);
end
% confidence value for true labels
conf=zeros(numNt,numL);
%decoding 
tmptime=cputime;
[~, conf] = round_linear_decode(tmpconf, model{dim+2});
time{end}=cputime-tmptime;

