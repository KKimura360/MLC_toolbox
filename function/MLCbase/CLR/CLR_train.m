function[model,time]=CLR_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.base.name= base classifier
%method.base.param= parameters of the base classifier
%% Output
%model(number of pair of labels including the calibrated label,1)
%time: computaiton time
%% Reference (APA style from google scholar)
%F?rnkranz, J., H?llermeier, E., Menc?a, E. L., & Brinker, K. (2008). Multilabel classification via calibrated label ranking. Machine learning, 73(2), 133-153.
%error check 

%%% Method
%% Initialization
[numN, numF]=size(X);
[numNL,numL]=size(Y);
% all pairs (#pairs,2 matrix)
allComb=nchoosek(1:(numL+1),2);
% number of pairs
numComb=size(allComb,1);

model=cell(numComb+1,1);
model{end}=allComb;
time=cell(numComb+1,1);

%% Learning model
fprintf('CALL: %s\n',method.base.name);

for i=1:numComb
    % left label allComb(x,1) of a pair as positive label
    % right label allComb(x,2) of a pair as negative label 
    posLabel=allComb(i,1);
    negLabel=allComb(i,2);
    %Instance extraction 
    %if positive label is the calibrated label
    if posLabel==(numL+1);
        % instances in positive class are instances which do not have
        % negative label
        posInd=(Y(:,negLabel)==0);
        % instances in negative class are instances which have the negative
        % label
        negInd=(Y(:,negLabel)==1);
    %if negative label is the calibrated label (vice versa)
    elseif negLabel==(numL+1)
        negInd=(Y(:,posLabel)==0);
        posInd=(Y(:,posLabel)==1);
    %both positive and negative lavels are not the calibrated label
    %we may use the same instances twice  
    else 
        %positive instances have the positive label
        posInd=(Y(:,posLabel)==1);
        %negative instances have the negative label
        negInd=(Y(:,negLabel)==1);
    end
    %problem transformation
    %construct Feature matrix  
    tmpX=[X(posInd,:); X(negInd,:)];
    %construct Label matrix
    tmpY=[ones(sum(posInd),1); zeros(sum(negInd),1)];
    %Learn binary classification 
    [model{i},time{i}]=feval([method.base.name,'_train'],tmpX,tmpY,method);
end


