function [model,time] = MLMIM_train(X,Y,method)
%MLMIM The multi-label mutual information maximization method
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim      lower-dim of labels
%method.param{x}.numStat  statuses of discretized features (3 or 5)
%method.param{x}.factor   factor of discretization
%% Output
%model: A learned model 

%% Reference (APA style from google scholar)
%Sechidis, K., Nikolaou, N., & Brown, G. (2014, August). Information theoretic feature selection in multi-label data through composite likelihood. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 143-152). Springer Berlin Heidelberg.

%%% Method

%% Get the input paramters
dim = method.param{1}.dim;
numStat = method.param{1}.numStat;
factor = method.param{1}.factor;

%% Initialization
numF = size(X,2);
numL = size(Y,2);
time=cell(2,1);
tmptime=cputime;

%% Discretize the numeric features
disX = myDisc(X,numStat,factor);

%% Perform global multi-label feature selection
if dim < numF
    disX = full(disX);
    rel_mat = zeros(numF,numL);
    for i = 1:numF
        for j = 1:numL
            rel_mat(i,j) = mi(disX(:,i), Y(:,j));
        end
    end
    [~, idxs] = sort(sum(rel_mat,2),'descend');
    id_F = idxs(1:dim);
else
    id_F = 1:numF;
end

%% Return the learned model
time{end}=cputime-tmptime;
model = cell(2,1);
[model{1},time{1}] = feval([method.name{2},'_train'],X(:,id_F),Y,Popmethod(method));
model{2} = id_F;

end