function [fea] = gJMI(data, target, M)
% function [fea] = gJMI(data, target, pa, K)
%
% Global multi-label feature selection
%
% The parameters:
%  data   - a N*M matrix, indicating N samples, each having M dimensions. Must be integers.
%  target - a N*1 matrix (vector), indicating the class/category of the N samples. Must be categorical.
%  parent - a N*|pa| matrix, parent label matrix of target
%  K      - the number of features need to be selected
% 
% Output
%  fea    - a K*1 array, save indices of selected features

%% Control the search space
MAX = 200;

%% Get the statistics of data
num_fea = size(data,2);
num_label = size(target,2);

%% Cache the relevancy term
rel = zeros(1,num_fea);
for i = 1:num_fea
    for j = 1:num_label
        rel(i) = rel(i) + mi(data(:,i), target(:,j));
    end
end

%% Limit the search space by KMAX
fea = zeros(M,1);
[~, idxs] = sort(rel,'descend');
fea(1) = idxs(1);
KMAX = min(MAX,num_fea);
idxleft = idxs(2:KMAX);
num_select = 1;
num_unselect = KMAX-1;

%% Greedy forward selection
diff_mat = zeros(num_unselect,M);
for k = 2:M
    rel_mi = zeros(num_unselect,1);   % Relevance
    diff_mi = zeros(num_unselect,1);  % Difference of Redundancy
    for i = 1:num_unselect,
        rel_mi(i) = rel(idxleft(i));
        diff_mat(idxleft(i),num_select) = cmi(target(:,j), ...
            data(:,fea(num_select)), data(:,idxleft(i)));
        diff_mi(i) = diff_mi(i) + mean(diff_mat(idxleft(i),1:num_select));
    end
    
    % Multi-label feature selection criterion
    [~, fea(k)] = max( rel_mi + diff_mi);
    
    tmpidx = fea(k);
    fea(k) = idxleft(tmpidx);
    idxleft(tmpidx) = [];
    num_select = num_select + 1;
    num_unselect = num_unselect - 1;
end

end