function [model,time] = MLMRMR_train(X,Y,method)
%MLJMI The multi-label max-relevance and min-redanduncy method
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
% Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern analysis and machine intelligence, 27(8), 1226-1238.

%%% Method
[numN,numF]=size(X);
[~,numL]=size(Y);

%% Get the input parameters
dim     = method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
numStat = method.param{1}.numStat;
factor  = method.param{1}.factor;

%% Initialization
time=cell(2,1);
tmptime=cputime;

%% Discretize the training data
disX = myDisc(X,numStat,factor);

%% Perform global multi-label feature selection
if dim < numF
    disX = full(disX);
    % Cache the relevancy term
    rel = zeros(1,numF);
    for i = 1:numF
        for j = 1:numL
            rel(i) = rel(i) + mi(disX(:,i), Y(:,j));
        end
    end   
    % Limit the search space by KMAX
    MAX = 500;
    id_F = zeros(dim,1);
    [~, idxs] = sort(rel,'descend');
    id_F(1) = idxs(1);
    KMAX = min(MAX,numF);
    idxleft = idxs(2:KMAX);
    num_select = 1;
    num_unselect = KMAX-1;    
    % Greedy forward selection
    red_mat = zeros(num_unselect,dim);
    for k = 2:dim
        rel_mi = zeros(num_unselect,1);   % Relevance
        red_mi = zeros(num_unselect,1);   % Redundancy
        for i = 1:num_unselect,
            rel_mi(i) = rel(idxleft(i));
            red_mat(idxleft(i),num_select) = mi(disX(:,id_F(num_select)),...
                disX(:,idxleft(i)));
            red_mi(i) = red_mi(i) + mean(red_mat(idxleft(i),1:num_select));
        end
        % Multi-label feature selection criterion
        [~, id_F(k)] = max( rel_mi + red_mi);
        tmpidx = id_F(k);
        id_F(k) = idxleft(tmpidx);
        idxleft(tmpidx) = [];
        num_select = num_select + 1;
        num_unselect = num_unselect - 1;
    end
else
    id_F = 1:numF;
end

%% Return the learned model
time{end}=cputime-tmptime;
model = cell(2,1);
[model{1},time{1}] = feval([method.name{2},'_train'],X(:,id_F),Y,Popmethod(method));
model{2} = id_F;

end