function[conf,time]=BPMLL_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by rankSVM_train
%% Output
%conf: confidence values
%% Reference
%Zhang, M. L., & Zhou, Z. H. (2006). Multilabel neural networks with applications to functional genomics and text categorization. IEEE transactions on Knowledge and Data Engineering, 18(10), 1338-1351.
%http://cse.seu.edu.cn/people/zhangml/Resources.htm

%errorcheck
[numN,numF]=size(X);
[numNL,numL]=size(Y);

%initialization
net=model{1}{end};
time=cputime;
[conf]=BPMLL_test_raw(X,Y',Xt,Yt',net);
time=cputime-time;
conf=conf';