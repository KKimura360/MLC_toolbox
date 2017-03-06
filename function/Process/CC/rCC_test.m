function[conf,time]=rCC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%model learned by rCC_train
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it.
%% Parameter
%method.param{x}.th => thresholding method to obtain classification result
%(option) if theere is not this, pass confidence values.
%in the classification chain
    %method.param{x}.th.type= Thresholding strategy
    %method.param{x}.th.param= parameter for threshold.
%% Reference (APA style from google scholar)
% Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. Machine learning, 85(3), 333.

%%% Method 
[numN,numF]=size(X);
[numNL,numL]=size(Y);

%% Initialization
[numNt,~]=size(Xt);
% to keep confidence value for the next generaltion
conf=zeros(numNt,numL);
time=cell(numL+1,1);
time{end}=0;
pred=conf;

% use chainorder obtained by rCC_train
chainorder=model{numL+1};
for label=1:numL
    %Don't make me say again, see rCC_train.m
    if label >1
        tmpXt=[Xt pred(:,chainorder(label-1))];
    else
        tmpXt=Xt;
    end
    %obtain confidence value from base classifier model 
    [conf(:,chainorder(label)),time{chainorder(label)}]=feval([method.base.name,'_test'],X,Y,tmpXt,model{chainorder(label)},method);
    %to keep next generation of inference
    %not to pass confidence value
    if isfield(method.param{1},'th')
        %Call Thresholding 
        [pred(:,chainorder(label))]=Thresholding(conf(:,chainorder(label)),method.param{1}.th);
    else
        pred(:,chainorder(label))=conf(:,chainorder(label));
    end
end


