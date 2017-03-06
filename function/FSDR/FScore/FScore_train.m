function [model] = FScore_train(X,Y,method)
%Fisher Score, use the N var formulation
%   X, the data, each raw is an instance
%   Y, the label in 1 2 3 ... format

%% Reference (APA style from google scholar)
% Hart, P. E., Stork, D. G., & Duda, R. O. (2001). Pattern classification. John Willey & Sons.

%% Set parameters
numL = size(Y,1);
[~, numF] = size(X);
out.W = zeros(1,numF);
dim = method.param{1}.dim;

% statistic for classes
cIDX = cell(numL,1);
n_i = zeros(numL,1);
for j = 1:numL
    cIDX{j} = find(Y(j,:)==1);
    n_i(j) = length(cIDX{j});
end

% calculate score for each features
for i = 1:numF
    temp1 = 0;
    temp2 = 0;
    f_i = X(:,i);
    u_i = mean(f_i);
    
    for j = 1:numL
        u_cj = mean(f_i(cIDX{j}));
        var_cj = var(f_i(cIDX{j}),1);
        temp1 = temp1 + n_i(j) * (u_cj-u_i)^2;
        temp2 = temp2 + n_i(j) * var_cj;
    end
    
    if temp1 == 0
        out.W(i) = 0;
    else
        if temp2 == 0
            out.W(i) = 100;
        else
            out.W(i) = temp1/temp2;
        end
    end
end

[~, out.fList] = sort(out.W, 'descend');
out.prf = 1;
id = out.fList(1:dim);

%% Return the learned model
model = cell(3,1);
[model{1}] = feval([method.name{2},'_train'],X(:,id),Y,Popmethod(method));
model{2} = id;
model{3} = out.W;

end