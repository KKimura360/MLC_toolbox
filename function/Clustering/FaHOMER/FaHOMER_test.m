function[conf]=FaHOMER_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%linear_svm does not return confidence value since LIBLINEAR does not
%support it. 
%% Reference (APA style from google scholar)
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMDÅf08) (pp. 30-44).
%% Method

%error check 
%initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
conf=zeros(numNt,numL);
numCls=method.param{1}.numCls;

if numL <numCls
    numCls=numL;
end
%test instance assign with neighborhood
%assign vector (need for k-nn classifier)
assign=model{numCls+1};
%centroid of clusters
tmpmodel=model{numCls+2};

tmpY=zeros(numN,numCls);
%Learning model on this layer
for i=1:numCls
    Lind=(assign==i);
    Nind=sum(Y(:,Lind),2)>0;
    tmpY(Nind,i)=1;
end
%classification on this layer
[numNt,numF]=size(Xt);
% just multiplying
XXt=[ones(numNt,1),Xt];
tmpconf=XXt*tmpmodel;
[tmppred]=Thresholding(tmpconf,method.th);

for i=1: numCls
    %problem transform
    Lind=(assign==i);
    Nind=sum(Y(:,Lind),2)>0;
    Ntind=(tmppred(:,i)>0); %must be binary
    % if no test instances assigned, skip the cluster
    if sum(sum(Ntind))==0
        continue;
    end
    % problem transformation
    tmpXt=Xt(Ntind,:);
    tmpX=X(Nind,:);
    tmpY=Y(Nind,Lind);
    % Set the model learned by FaHOMER_train with cluster(Clscount)
    tmpmodel=model{i};
    if iscell(tmpmodel)
        [tmpconf]=feval([method.name{1},'_test'],tmpX,tmpY,tmpXt,tmpmodel,method);
        conf(Ntind,Lind)=tmpconf;
    else
     conf(Ntind,i)=tmpconf(Ntind,i);
    end 
end

