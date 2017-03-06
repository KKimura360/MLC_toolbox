function [model] = RFS_train(X,Y,method)
%RFS Efficient and Robust Feature Selection via Joint L2,1-Norms Minimization

%Nie, F., Huang, H., Cai, X., & Ding, C. H. (2010). Efficient and robust feature selection via joint ?2, 1-norms minimization. In Advances in neural information processing systems (pp. 1813-1821).

%% Set parameters
alpha = method.param{1}.alpha;
dim   = method.param{1}.dim;
t_MAX = method.param{1}.max;
[N,M] = size(X); 

%% Absord b into the objective function
X = [ones(N,1),X];
M = M + 1;

%% The iterative algorithm
A = [X, alpha*eye(N)];
diag_D = ones(N+M,1);
% Q_val = zeros(t_MAX,1);
for t = 1:t_MAX
    % Cache the result
    DAT = bsxfun(@times,A',diag_D);
    
    % Update U
    U = DAT / (A*DAT) * Y';
    
    % Update D_inv
    diag_D = 2 * sqrt(sum(U.^2,2) + eps);
    
    % Calculate the vale of objective function
%     W = U(1:M,:);
%     Q_val(t) = sum(sqrt(sum((X*W-Y').^2,2))) + ...
%         alpha * sum(sqrt(sum(W.^2,2)));
end


%% Select the top ranked features
W = U(1:M,:);
% Remove the first row of W (corresponding to b)
X = X(:,2:end);
W = W(2:end,:);

%% Perform feature selection
[~, fea_order] = sort(sum(W.^2,2),'descend');
id = fea_order(1:dim);
X  = X(:,id);

%% Return the learned model
model = cell(3,1);
[model{1}] = feval([method.name{2},'_train'],X,Y,Popmethod(method));
model{2} = id;
model{3} = W;

end