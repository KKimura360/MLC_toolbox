function Yt = CLMLC_local( X,Y,Xt,n )
%CLMLC_local Local model learning for CLMLC
%
%    Syntax
%
%       Yt = CLMLC_local( X,Y,Xt,opts )
%
%    Description
%
%       Input:
%           X       An N x D data matrix, each row denotes a sample
%           Y       An L x N label matrix, each column is a label set
%           Xt      An Nt x D test data matrix, each row is a test sample
%           n       The size of a meta-label
% 
%       Output:
%           Yt      An L x Nt predicted label matrix, each column is a predicted label set

%% 0. Check if #label is enough large
num_label = size(Y,1);
if num_label < 6 
    Yt = CC(X,Y,Xt);
    return;
end

%% 1. Group labels into meta-labels
n = ceil(num_label/n);
cluster_alg = 'kmeans';
if strcmp(cluster_alg,'sc')
    % Construct affinity matrix
    A1 = 1 - pdist(Y,'jaccard');
    label_mean = bsxfun(@rdivide,Y*X,sum(Y,2));
    A2 = exp(-pdist(label_mean));
    A = (A1 + A2) / 2;
    A(isnan(A)) = 0;
    A = sparse(squareform(A));  
    % Apply spectral clustering
    D = sum(A,2); D(D==0) = eps;
    D = spdiags(1./sqrt(D),0,num_label,num_label);
    [U, ~] = eigs(D*A*D,n,'LM');
    U = bsxfun(@rdivide, U, sqrt(sum(U.^2, 2)));
    C = litekmeans(U(:,2:end),n,'MaxIter',20);
elseif strcmp(cluster_alg,'kmeans')
    C = litekmeans(Y,n,'MaxIter',20);
else
    error('Unavailable clustering method.');
end

%% 2. Build classifier chains over meta-labels
num_test = size(Xt,1);
Yt = zeros(num_label,num_test);
Ones = ones(num_test,1);
for k = 1:n 
    meta_target = Y((C==k),:);
    temp_size = size(meta_target,1);  
    if temp_size == 1
        temp_Labels = round(([Ones,Xt]*ridgereg(meta_target',X,0.1))');
        temp_Labels(temp_Labels>1) = 1; temp_Labels(temp_Labels<0) = 0;
        Yt((C==k),:) = temp_Labels;
    elseif temp_size == 2
        Yt((C==k),:) = CC(X,meta_target,Xt);
    else
        Yt((C==k),:) = round((CC(X,meta_target,Xt)+CC(X,meta_target,Xt))./2);
    end
    X  = [X,meta_target'];
    Xt = [Xt,Yt((C==k),:)'];
end

end