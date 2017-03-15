function[conf,time]=BMaD_test(X,Y,Xt,model,method)
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
time=cell(2,1);
%Call next model
[tmpconf,time{1}]=feval([method.name{2},'_test'],X,model{2},Xt,model{1},Popmethod(method));
tmptime=cputime;
[tmppred]=Thresholding(tmpconf,method.th);
conf=tmppred * model{3};
conf(conf>0)=1;
conf(conf<0)=0;
