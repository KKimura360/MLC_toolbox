function[conf,time]=HOMER_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%time: computation time 
%% Reference (APA style from google scholar)
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD�f08) (pp. 30-44).

%%% Method

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
conf=zeros(numNt,numL);
numCls=method.param{1}.numCls;
if numL <numCls
    numCls=numL;
end
time=cell(numCls+2,1);
tmptime=cputime;
%test instance assign with neighborhood
%assign vector (need for k-nn classifier)
assign=model{numCls+1};
%centroid of clusters
tmpmodel=model{numCls+2};

tmpY=zeros(numN,numCls);
%Learning model on this layer
for Clscount=1:numCls
    Lind=(assign==Clscount);
    Nind=sum(Y(:,Lind),2)>0;
    tmpY(Nind,Clscount)=1;
end
%classification on this layer
[tmpconf,time{end-1}]=feval([method.name{2},'_test'],X,tmpY,Xt,tmpmodel,Popmethod(method));
[tmppred]=Thresholding(tmpconf,method.th,Y);
time{end}=cputime - tmptime;
for Clscount=1: numCls
    %problem transform
    Lind=(assign==Clscount);
    Nind=sum(Y(:,Lind),2)>0;
    Ntind=(tmppred(:,Clscount)>0); %must be binary
    % if no test instances assigned, skip the cluster
    if sum(sum(Ntind))==0
        continue;
    end
    % problem transformation
    tmpXt=Xt(Ntind,:);
    tmpX=X(Nind,:);
    tmpY=Y(Nind,Lind);
    % Set the model learned by CBMLC_train with cluster(Clscount)
    tmpmodel=model{Clscount};
    if iscell(tmpmodel)
        [tmpconf,time{Clscount}]=feval([method.name{1},'_test'],tmpX,tmpY,tmpXt,tmpmodel,method);
        conf(Ntind,Lind)=tmpconf;
    else
     conf(Ntind,Clscount)=tmpconf(Ntind,Clscount);
    end 
end

