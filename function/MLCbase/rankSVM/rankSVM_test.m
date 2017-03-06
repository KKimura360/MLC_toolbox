function[conf,time]=rankSVM_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%model learned by rankSVM_train
%% Output
%conf: confidence values
%% Reference
%Elisseeff, A., & Weston, J. (2001, December). A kernel method for multi-labelled classification. In NIPS (Vol. 14, pp. 681-687).
%http://cse.seu.edu.cn/people/zhangml/Resources.htm


%errorcheck
[numN,numF]=size(X);
[numNL,numL]=size(Y);


%initialization
svm=model{6};
Weights=model{1};
Bias=model{2};
SVs=model{3};
Weights_sizepre=model{4};
Bias_sizepre=model{5};

time=cputime;
[conf,~,~]=RankSVM_test_raw(Xt,Yt',svm,Weights,Bias,SVs,Weights_sizepre,Bias_sizepre);
conf=conf';
time=cputime-time;
