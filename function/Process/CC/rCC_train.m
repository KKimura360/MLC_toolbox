function[model,time]=rCC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.base.name= base classifier
%method.base.param= parameters of the base classifier
%% Output
%model: A learned model (cell(numL,1))
%model{1:numL}: Larned classifier, corresponding to a label
%% Reference (APA style from google scholar)
% Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%%% Method

[numN, numF]=size(X);
[numNL,numL]=size(Y);

%% Initialization
model=cell(numL+1,1);
time=cell(numL+1,1);
tmptime=cputime;
%randomly obtain classifier chain (1-order)
chainorder=randperm(numL);
%keep this at last cell of model
model{numL+1}=chainorder;
time{end}=cputime-tmptime;
%Learning model
fprintf('CALL: %s\n',method.base.name);
for label=1:numL
    if label>1
        %problem transformation, adding label info. to the feature
        tmpX= [X, Y(:,chainorder(label-1))];
    else
        % the first label does not have any parents. 
        tmpX=X;
    end
    % obtain model with transformed problem
    [model{chainorder(label)},~,time{chainorder(label)}]=feval([method.base.name,'_train'],tmpX,Y(:,chainorder(label)),method);
end



