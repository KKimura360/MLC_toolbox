load ../../../../dataset/matfile/bibtex.mat;
load ../../../../dataset/index/5-fold/bibtex.mat;

addpath('../Tools/matlab');


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


fastXML_train(X',Y,param);
score_mat=fastXML_test(Xt', param);