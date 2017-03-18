function[conf,time]=FastXML_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param: see setFastXML
%% Output
%model: A learned model of FastXML
%% Reference (APA style from google scholar)
% Prabhu, Y., & Varma, M. (2014, August). Fastxml: A fast, accurate and stable tree-classifier for extreme multi-label learning. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 263-272). ACM.
%%% Method
%% Initialization
param=method.param{1};
X_name=model{1};
Y_name=model{2};
Xt_name=model{3};
Yt_name=model{4};
model_name=model{5};
dicname=model{6};
time=cputime;
Xt=sparse(Xt');
conf=fastXML_test_raw(Xt, param,Xt_name,Yt_name,model_name);
conf=conf';
rmdir(dicname,'s');
time=cputime-time;
