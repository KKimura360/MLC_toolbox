function[conf,time]=NMF_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by NMF_train
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Lee, D. D., & Seung, H. S. (2001). Algorithms for non-negative matrix factorization. In Advances in neural information processing systems (pp. 556-562).

%%% Method 
%% Get learned model
time=cell(2,1);
tmpX=model{2};
V = model{3};

U=sprand(size(Xt,1),size(V,2),1);
tmptime=cputime;
for i=1:30
    U=updateNMF(Xt,U,V);
end
time{end}=cputime-tmptime;
% Testing
[conf,time{1}] = feval([method.name{2},'_test'],tmpX,Y,U,model{1},Popmethod(method));