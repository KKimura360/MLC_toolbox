function[conf,time]=PS_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Option
%method.param{x}.vote % not implemented yet for RAkeL++ or something
%% Reference (APA style from google scholar)
%Read, J., Pfahringer, B., & Holmes, G. (2008, December). Multi-label classification using ensembles of pruned sets. In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 995-1000). IEEE.
%% Method

%error check 
newX=model{2};
newY=model{3};
%initialization
time=cell(2,1);
time{end}=0;
[conf,time{1}] = feval([method.name{2},'_test'],newX,newY,Xt,model{1},Popmethod(method));

