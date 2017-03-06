function A = gen_nn_distance(data, num_neighbors, block_size)
%GEN_NN_DISTANCE Generate (t-nearest-neighbor) sparse distance matrix.
%
%   Input  : data         : N-by-D data matrix, where N is the number of data,
%                           D is the number of dimensions
%            num_neighbors: number of nearest neighbors
%            block_size   : block size for partitioning data matrix
%            save_type    : 0 for .mat file, 1 for .txt file, 2, for both
%
%   Output : Either .mat or .txt or both files

%
% Divide data into blocks, process each block sequentially to alleviate memory use
%

n = size(data, 1);
num_iter = ceil(n/block_size);
A = sparse(n, n);
dataT = data';

% For Euclidean distance, computing data.*data outside the loop to save time
tmp = full(sum(data.*data, 2));
aa = tmp(:, ones(block_size, 1));
clear tmp;

for i = 1:num_iter
  start_index = 1 + (i-1)*block_size;
  end_index = min(i*block_size, n);
  num_data = end_index - start_index + 1;

  % Select a block of data, fetching in column order is faster
  block = dataT(:, start_index:end_index);

  % Compute Euclidean distance between block and data
  if (num_data < block_size)
    aa = aa(:, 1:num_data);
  end
  tmp = full(sum(block.*block, 1));
  bb = tmp(ones(n, 1), :);
  clear tmp;
  ab = full(data*block);
  dist = aa + bb - 2*ab;
  clear bb ab block;

  % Find nearest neighbors
  [value, index] = sort(dist, 1);
  tempindex = index(2:num_neighbors+1, :);
  rowindex = reshape(tempindex, size(tempindex, 1)*size(tempindex, 2), 1);
  tempindex = repmat(1:num_data, num_neighbors, 1);
  columnindex = reshape(tempindex, size(tempindex, 1)*size(tempindex, 2), 1);
  tempvalue = value(2:num_neighbors+1, :);
  value = reshape(tempvalue, size(tempvalue, 1)*size(tempvalue, 2), 1);
  value = sqrt(max(value, 1.0e-12));
  A(:, start_index:end_index) = sparse(rowindex, columnindex, double(value), n, num_data);
end
%outfile = [num2str(num_neighbors), '_NN_nonsym_distance.mat'];
%save(outfile, 'A');
clear data dataT tempindex rowindex columnindex tempvalue value;


%
% Make the sparse distance matrix symmetric
%
A1 = triu(A);
A1 = A1 + A1';
A2 = tril(A);
A2 = A2 + A2';
clear A;
max_num = 100000;
if (n < max_num)
  A = max(A1, A2);
else % Do 'max' function sequentially for very large data
  num_iter = ceil(n/max_num);
  B = sparse([]);
  for i = 1:num_iter
    disp(i);
    start_index = 1 + (i-1)*max_num;
    end_index = min(i*max_num, n);
    B = max(A1(:, start_index:end_index), A2(:, start_index:end_index));
    % temp matrix for saving memory use
    tmpfile = ['tmp_', num2str(i), '.mat'];
    save(tmpfile, 'B');
    clear B;
  end
end
clear A1 A2;

%
% Concatenate all temp matrices
%
if (n > max_num)
  A = sparse([]);
  for i = 1:num_iter
    tmpfile = ['tmp_', num2str(i), '.mat'];
    load(tmpfile); % temp matrix 'B'
    A = [A B];
    clear B;
  end
end
delete tmp*;

%
% Force symmetric matrix's diagonal to be 0
%
n = size(A, 1);
B = spdiags(diag(A), 0, n, n);
A = A - B;

