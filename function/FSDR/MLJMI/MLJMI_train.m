function [model,time] = MLJMI_train(X,Y,method)
%MLJMI The multi-label joint mutual information method
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.dim      lower-dim of labels
%method.param{x}.numStat  statuses of discretized features (3 or 5)
%method.param{x}.factor   factor of discretization
%method.param{x}.maxF     factor on limiting the search space
%% Output
%model: A learned model
%
%% Reference (APA style from google scholar)
%Sechidis, K., Nikolaou, N., & Brown, G. (2014, August). Information theoretic feature selection in multi-label data through composite likelihood. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 143-152). Springer Berlin Heidelberg.

%%% Method

%% Get input parameters
dim     = method.param{1}.dim;
numStat = method.param{1}.numStat;
factor  = method.param{1}.factor;
maxF    = method.param{1}.maxF;

%% Initialization
numF  = size(X,2);
numL  = size(Y,2);
time  = cell(2,1);
tmptime = cputime;

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
    % Limit the search space by maxF
    id_F = zeros(dim,1);
    [~, idxs] = sort(rel,'descend');    
    if maxF <= dim
        id_F = idxs(1:dim);
    else
        id_F(1) = idxs(1);
        maxF = min(maxF,numF);
        idxleft = idxs(2:maxF);
        num_select = 1;
        num_unselect = maxF-1;      
        % Greedy forward selection
        dif_mat = zeros(num_unselect,dim);
        for k = 2:dim
            rel_mi = zeros(num_unselect,1);   % Relevance
            dif_mi = zeros(num_unselect,1);   % Difference of Redundancy
            for i = 1:num_unselect,
                rel_mi(i) = rel(idxleft(i));
                dif_mat(idxleft(i),num_select) = cmi(Y(:,j), ...
                    disX(:,id_F(num_select)), disX(:,idxleft(i)));
                dif_mi(i) = dif_mi(i) + mean(dif_mat(idxleft(i),1:num_select));
            end
            % Multi-label feature selection criterion
            [~, id_F(k)] = max( rel_mi + dif_mi);
            tmpidx = id_F(k);
            id_F(k) = idxleft(tmpidx);
            idxleft(tmpidx) = [];
            num_select = num_select + 1;
            num_unselect = num_unselect - 1;
        end
    end
else
    id_F = 1:numF;
end

%% Return the learned model
time{end}=cputime-tmptime;
model = cell(2,1);
[model{1},time] = feval([method.name{2},'_train'],X(:,id_F),Y,Popmethod(method));
model{2} = id_F;