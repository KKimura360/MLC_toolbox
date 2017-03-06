function [model] = MLMIM_train(X,Y,method)
%MLMIM The multi-label mutual information maximization method (local)

%% Get the parameters

%% Discretize the training data 
dim = method.param{1}.dim;
num_state = method.param{1}.num_state;
factor = method.param{1}.factor;

%% Discretize the numeric features
num_feature = size(X,2);
disc_X = myDisc(X,num_state,factor);

%% Perform global multi-label feature selection
if dim < num_feature
    ind_F = gMIM(disc_X,Y,dim);
else
    ind_F = 1:num_feature;
end

%% Return the learned model
model = cell(2,1);
[model{1}] = feval([method.name{2},'_train'],X(:,ind_F),Y,Popmethod(method));
model{2} = ind_F;

end