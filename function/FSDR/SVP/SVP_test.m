function[conf,time]=SVP_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by SVP_train
%% Output
%conf: confidence values (Nt x L);
%time: computaiton time
%% Reference (APA style from google scholar)
%Yu, K., Yu, S., & Tresp, V. (2005, August). Multi-label informed latent semantic indexing. In Proceedings of the 28th annual international ACM SIGIR conference on Research and development in information retrieval (pp. 258-265). ACM.

%%% Method
%% Initialization
time=cell(2,1);
tmptime=cputime;
%NOTE X must be processed correctly on previous layers
% to shift in the same manner
W=model{1};
tmpX=X*W;
tmpXt=Xt*W;

time{end}=cputime-tmptime;

[conf,time{1}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));


