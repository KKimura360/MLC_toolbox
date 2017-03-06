function[model,time]=LP_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.base.name= base classifier
%method.base.param= parameters of the base classifier
%% Output
%model: A learned model (cell(2,1))
%model{1}: Larned classifier (depends on return of base classifier)
%model{2}: Labelset (#distinct label x #Label) to obtain multi-label
%classification result

%% Method 
%error check 

[numN numF]=size(X);
[numNL,numL]=size(Y);

%size check
sizeCheck;

%initialization
model=cell(2,1);
time=cell(2,1);
tmptime=cputime;
%Problem transformation
[Labelset, ~, newY]=unique(Y,'rows');
%Learning model
fprintf('CALL: %s\n',method.base.name);
time{end}=cputime-tmptime;
[tmpmodel,method,time{1}]=feval([method.base.name,'_train'],X,newY,method);
%Learned model
model{1}=tmpmodel;
model{2}=Labelset;
