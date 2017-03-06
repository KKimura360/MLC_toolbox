function [model] = MLMRMR_train(X,Y,method)
%MLJMI The multi-label joint mutual information method (global)

dim = method.param{1}.dim;
num_state = method.param{1}.num_state;
factor = method.param{1}.factor;

%% Discretize the training data 
disc_X = myDisc(X,num_state,factor);

%% Perform global multi-label feature selection
num_feature = size(X,2);
if K < num_feature
    ind_F = gMRMR(disc_X,Y,dim);
else
    ind_F = 1:num_feature;
end

%% Return the learned model
model = cell(2,1);
[model{1}] = feval([method.name{2},'_train'],X(:,ind_F),Y,Popmethod(method));
model{2} = ind_F;


end

