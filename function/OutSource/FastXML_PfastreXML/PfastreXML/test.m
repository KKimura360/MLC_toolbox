load ../../../../dataset/matfile/bibtex.mat;
load ../../../../dataset/index/5-fold/bibtex.mat;

addpath('../Tools/matlab');
addpath('../Tools/metrics');


trainind=(indices(:,1)~=1);
testind=(indices(:,1)==1);

X=sparse(data(trainind,:));
Y=sparse(target(:,trainind));
Xt=sparse(data(testind,:));
Yt=sparse(target(:,testind));


param.num_thread=1;
param.start_tree=0;
param.num_tree=50;
param.bias=1.0;
param.log_loss_coeff=1.0;
param.lbl_per_leaf=100;
param.max_leaf=10;
param.gamma=30;
param.alpha=0.8;

[inv_prop]=inv_propensity(Y,param.alpha,param.gamma);

PfastreXML_train(X',Y,inv_prop,param,'tmp/model/');
score_mat=PfastreXML_test(Xt', param,'tmp/model/');