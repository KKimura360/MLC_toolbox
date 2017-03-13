function [model,time] = MLJMI_train(X,Y,method)
%MLJMI The multi-label joint mutual information method
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

%% Get input parameters
dim     = method.param{1}.dim;
numStat = method.param{1}.numStat;
factor  = method.param{1}.factor;

if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end

%% Initialization
numF = size(X,2);
numL = size(Y,2);
time=cell(2,1);
tmptime=cputime;
model = cell(2,1);

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
    MAX_F = 500;
    id_F = zeros(dim,1);
    [~, idxs] = sort(rel,'descend');
    id_F(1) = idxs(1);
    KMAX = min(MAX_F,numF);
    idxleft = idxs(2:KMAX);
    num_select = 1;
    num_unselect = KMAX-1;
    
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
else
    id_F = 1:numF;
end
time{end}=cputime-tmptime;

%% Return the learned model
[model{1},time] = feval([method.name{2},'_train'],X(:,id_F),Y,Popmethod(method));
model{2} = id_F;


end