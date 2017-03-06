function [model,time] = MLJMI_train(X,Y,method)
%% Input
%X: Feature matrix
%Y: Label matrix
%method.param{x}.dim: dimension to use
%method.param{x}.num_state
%method.param{x}.factor
%% Output
%model: leanred model
%time: computaiton time
%% Reference
%MLJMI The multi-label joint mutual information method 

%%% Method
dim = method.param{1}.dim;
num_state = method.param{1}.num_state;
factor = method.param{1}.factor;
time=cell(2,1);
tmptime=cputime;
model = cell(2,1);

%% Discretize the training data 
disc_X = myDisc(X,num_state,factor);

%% Perform global multi-label feature selection
ind_F = gJMI(disc_X,Y,dim);
time{end}=cputime-tmptime;

%% Return the learned model
[model{1},time] = feval([method.name{2},'_train'],X(:,ind_F),Y,Popmethod(method));
model{2} = ind_F;


end