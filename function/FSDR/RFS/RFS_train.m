function [model,time] = RFS_train(X,Y,method)
%% Input
%X
%Y
%method.param{x}.alpha: weighting parameter
%method.param{x}.dim  : dimension of reduced features
%method.param{x}.iter  : max number of iterations
%% OutPut
% model
%% Reference
%RFS Efficient and Robust Feature Selection via Joint L2,1-Norms Minimization
%Nie, F., Huang, H., Cai, X., & Ding, C. H. (2010). Efficient and robust feature selection via joint ?2, 1-norms minimization. In Advances in neural information processing systems (pp. 1813-1821).

%%% Method

%% Initialization
alpha = method.param{1}.alpha;
dim   = method.param{1}.dim;
maxIt = method.param{1}.maxIt;
epsIt = method.param{1}.epsIt;
[numN, numF] = size(X); 
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
time=cell(2,1);
tmptime=cputime;

%% Absord b into the objective function
X = [ones(numN,1),X];
numF = numF + 1;

%% The iterative algorithm
Qval = zeros(maxIt,1);
t = 1;
if numF > numN    % Solve by the original algorithm 
    A = [X, alpha.*eye(numN)];
    diagD = ones(numN+numF,1);
    while t < maxIt
        % Update U
        DAT = bsxfun(@times,A',diagD);
        U = DAT / (A*DAT) * Y;
        
        % Update D
        tmpU = sum(U.^2,2);
        diagD = 2 .* sqrt(tmpU+eps);
        
        % Check if convergence condition is matched
        Qval(t) = sum(sqrt(tmpU));
        if t > 1
            if ( abs(Qval(t)-Qval(t-1))<epsIt || abs(Qval(t)-Qval(t-1))/Qval(t-1)<epsIt )
                break;
            end
        end
        t = t + 1;
    end
    W = U(1:numF,:);
else    % Solve by the algorithm of READER (beta=gamma=0 )
    alphaIt = alpha.^2;
    diagHn  = ones(numN,1);
    diagHm  = ones(numF,1);
    while t < maxIt
        % Get Hn and Hm
        Hm  = diag(alphaIt*diagHm);
        HnX = bsxfun(@times,X,diagHn);
        
        % Update W
        W = (HnX'*X+Hm) \ (HnX'*Y);
        
        % Update H
        tmpXWY = sum((X*W-Y).^2,2);
        tmpW   = sum((alpha*W).^2,2);
        diagHn = 0.5 ./ sqrt(tmpXWY + eps);
        diagHm = 0.5 ./ sqrt(tmpW + eps);
        
        %% Convergence condition
        Qval(t) = sum(sqrt(tmpXWY)) + sum(sqrt(tmpW));
        if t > 1
            if ( abs(Qval(t)-Qval(t-1))<epsIt || abs(Qval(t)-Qval(t-1))/Qval(t-1)<epsIt )
                break;
            end
        end
        t = t + 1;
    end
end
disp(['RFS converged at the ',num2str(t),'th iteration with ',num2str(Qval(t-1)),'.']);

%% Remove the bias dimension
X = X(:,2:end);
W = W(2:end,:);

%% Perform feature selection
[~,idf] = sort(sum(W.^2,2),'descend');
id = idf(1:dim);
X  = X(:,id);
time{end}=cputime-tmptime;

%% Return the learned model
model = cell(3,1);
[model{1},time{1}] = feval([method.name{2},'_train'],X,Y,Popmethod(method));
model{2} = id;
model{3} = W;