function [model,time] = FScore_train(X,Y,method)
%% Input 
%Fisher Score, use the N var formulation
%X: Feature matrix  (NxF) 
%Y: Label matrix    (NxL)
%% Output 
%model
%% Reference (APA style from google scholar)
% Hart, P. E., Stork, D. G., & Duda, R. O. (2001). Pattern classification. John Willey & Sons.

%%% Method

%% Initialization
[numN,numL] = size(Y);
[~, numF] = size(X);
W = zeros(1,numF);
dim = method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
time=cell(2,1);
tmptime=cputime;
% statistic for classes
cIDX = cell(numL,1);
n_i = zeros(numL,1);
for j = 1:numL
    cIDX{j} = find(Y(:,j)==1);
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
        W(i) = 0;
    else
        if temp2 == 0
            W(i) = 100;
        else
            W(i) = temp1/temp2;
        end
    end
end

[~, fList] = sort(W, 'descend');
id = fList(1:dim);

time{end}=cputime-tmptime;
%% Return the learned model
model = cell(3,1);
[model{1},time{1}] = feval([method.name{2},'_train'],X(:,id),Y,Popmethod(method));
model{2} = id;
model{3} = W;

end