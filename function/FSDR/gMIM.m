function [fea] = gMIM(data, target, M)
% function [fea] = gMIM(data, target, pa, K)
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

[~, idxs] = sort(rel,'descend');
fea = idxs(1:M);


end