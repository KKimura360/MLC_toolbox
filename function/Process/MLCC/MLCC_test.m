function[conf,time]=MLCC_test(X,Y,Xt,model,method)
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

%% Method 

%error check 
[numN,numF]=size(X);
[numNL,numL]=size(Y);
numCls=length(model)-2;
time=cell(numCls+1,1);
time{end}=0;

%initialization
[numNt,~]=size(Xt);
% to keep confidence value for the next generaltion
conf=zeros(numNt,numL);
pred=conf;
% use chainorder obtained by rCC_train
chainorder=model{numCls+1};
assign=model{numCls+2};
for Clscount=1:numCls
    %Don't make me say again, see rCC_train.m
    if Clscount>1
        %the parent meta-label 
        ind= (assign==chainorder(Clscount-1));
        %problem transformation, adding label info. to the feature
        tmpX= [X Y(:,ind)];
        tmpXt= [Xt pred(:,ind)];
    else
        % the first label does not have any parents. 
        tmpX=X;
        tmpXt=Xt;
    end
        ind=(assign==chainorder(Clscount));
        tmpY=Y(:,ind);
        %obtain confidence value from base classifier model 
        [tmpconf,time{Clscount}]=feval([method.name{2},'_test'],tmpX,tmpY,tmpXt,model{Clscount},Popmethod(method));
        conf(:,ind)=tmpconf;
        %not to pass confidence value
    if isfield(method.param{1},'th')
        %Call Thresholding 
        [pred(:,ind)]=Thresholding(conf(:,ind),method.param{1}.th);
    else
        pred(:,ind)=conf(:,ind);
    end
end


