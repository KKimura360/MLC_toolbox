function[conf,time]=CBMLC_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt: Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Nasierding, G., Tsoumakas, G., & Kouzani, A. Z. (2009, October). Clustering based multi-label classification for image annotation and retrieval. In Systems, Man and Cybernetics, 2009. SMC 2009. IEEE International Conference on (pp. 4514-4519). IEEE.
%Batzaya Norov-Erdene, Mineichi Kudo, Lu Sun and Keigo Kimura, "Locality in Multi-Label Classification Problems," in Proceedings of the 23rd International Conference on Pattern Recognition (ICPR 2016), Cancun, Mexico.
%Also Bhatia, K., Jain, H., Kar, P., Varma, M., & Jain, P. (2015). Sparse local embeddings for extreme multi-label classification. In Advances in Neural Information Processing Systems (pp. 730-738).
%% NOTE: CBMLC assigns test instances by nearest neighbor

%%% Method

%% Initialization
[numN,numF]=size(X);
[numNL,numL]=size(Y);
[numNt,~]=size(Xt);
%
conf=zeros(numNt,numL);
numCls=length(model)-2;
time=cell(numCls+1,1);
tmptime=cputime;

%assign vector (need for k-nn classifier)
assign=model{end-1};
%centroid of clusters
centroid=model{end};
%obtain assign vector for test instances with Euclidean distance
[~,assignt] = min(bsxfun(@plus,dot(Xt,Xt,2),dot(centroid,centroid,2)')-2*(Xt*centroid'),[],2);  

% calculcaiton time ends here
% each clasdsification time is saved by 
time{end}=tmptime-cputime;

for Clscount =1:numCls
    % instance separation
    %training instance
    test_instanceindex=(assignt==Clscount);
    %test instance
    train_instanceindex=(assign==Clscount);
    % if no test instances assigned, skip the cluster
    if sum(sum(test_instanceindex))==0
        continue;
    end
    % problem transformation
    tmpXt=Xt(test_instanceindex,:);
    tmpX=X(train_instanceindex,:);
    tmpY=Y(train_instanceindex,:);
    nzeroLabelind=(sum(tmpY)>0);
    tmpY=tmpY(:,nzeroLabelind);
    % Set the model learned by CBMLC_train with cluster(Clscount)
    tmpmodel=model{Clscount};
    fprintf('Cluster %d has %d instances and %d labels \r\n',Clscount,sum(train_instanceindex),size(tmpY,2));   
    [tmpconf,time{Clscount}]=feval([method.name{2},'_test'],tmpX,tmpY,tmpXt,tmpmodel,Popmethod(method));
    %subsutitute the result for assigned test instance
    conf(test_instanceindex,nzeroLabelind)=tmpconf;
end

