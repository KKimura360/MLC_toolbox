function[conf,time]=FaIE_test(X,Y,Xt,model,method)
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
% Lin, Z., Ding, G., Hu, M., & Wang, J. (2014). Multi-label Classification via Feature-aware Implicit Label Space Encoding. In ICML (pp. 325-333).

%%% Method  
[numN,numF]=size(X);
[numNL,numL]=size(Y);

%% Initialization
[numNt,~]=size(Xt);
dim=length(model)-2;
time=cell(dim+1,1);
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
[~, conf] = round_linear_decode(tmpconf, model{dim+2});
time{end}=cputime-tmptime;
