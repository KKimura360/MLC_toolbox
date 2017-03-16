function[conf,time]=MLDA_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by FaIE_train
%% Output
%conf: confidence values (Nt x L);
%time: computaiton time
%
%% Reference (APA style from google scholar)
%Wang, H., Ding, C., & Huang, H. (2010, September). Multi-label linear discriminant analysis. In European Conference on Computer Vision (pp. 126-139). Springer Berlin Heidelberg.

%%% Method

%% Initialization
U       = model{2};
m       = model{3};
time    = cell(2,1);
tmptime = cputime;

%% Prediction
tmpX  = X - ones(size(X,1),1)*m;
tmpXt = Xt - ones(size(Xt,1),1)*m;
tmpX  = sparse(tmpX * U);
tmpXt = sparse(tmpXt * U);
time{end}=cputime-tmptime;
[conf,time{1}]=feval([method.name{2},'_test'],tmpX,Y,tmpXt,model{1},Popmethod(method));


