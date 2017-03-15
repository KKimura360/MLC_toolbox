function[model,time]=CBMLC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.ClsMethod: Clustering method
%method.param{x}.numCls:    a number of clusters
%% Output
%model: A learned model (cell(numCls+2,1))
%model{1:numCls}: classifier depends on called method
%model{numCls+1}: indicator vector obtained by clustering (N vector)
%model{numCls+2}: centroids of clusters (matrix form: numCls x F) 
%% Option
%Here, we implemented k-means and spectral clustering
%% for k-means,
%method.param.ClsMethod='litekmeans'
%method.param.numCls 
%% for Spectral Clustering 
%method.param.ClsMethod='SC' or 'Spectral_Clustering'
%method.numCls
%method.sim.type: 'Ins-nn', 'Lab-nn' are avialble
%Ins-nn adjacency matrix based on feature space k-nearest neighbors
%Lab-nn adjacency matrix based on label space k-nearest neighbors 
%method.sim.k  #nearest neighbors for above two types
%methos.SCtype: to calculate Laplacian matrix, 
%SCtype=1 is unnormalized Laplacian
%SCtype=2 normalized Laplacian
%SCtype=3 normalized Laplacian   
% See util/Spectral_Clsutering/Spectral_Clustering.m
%% for hierarchical k-means clustering on SLEEC
%method.numCls: number of clusters for each layer
%method.mxPts : number of maximum instances in a cluster ( if #members
%larger than mxPts, next layer will be considered. 
%% Reference (APA style from google scholar)
%Nasierding, G., Tsoumakas, G., & Kouzani, A. Z. (2009, October). Clustering based multi-label classification for image annotation and retrieval. In Systems, Man and Cybernetics, 2009. SMC 2009. IEEE International Conference on (pp. 4514-4519). IEEE.
%Batzaya Norov-Erdene, Mineichi Kudo, Lu Sun and Keigo Kimura, "Locality in Multi-Label Classification Problems," in Proceedings of the 23rd International Conference on Pattern Recognition (ICPR 2016), Cancun, Mexico.
%Also Bhatia, K., Jain, H., Kar, P., Varma, M., & Jain, P. (2015). Sparse local embeddings for extreme multi-label classification. In Advances in Neural Information Processing Systems (pp. 730-738).
%
%%% Method 

%% error check 
if ~isfield(method.param{1},'numCls');
    error('CBMLC needs a number of clusters\n' );  
end

if ~isfield(method.param{1},'ClsMethod')
    warning('Clustering method is not set we use kmeans')
    method.param{1}.ClsMethod='litekmeans';
end 


%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
%number of clusters
numCls=method.param{1}.numCls;
%clustering method 
ClsMethod=method.param{1}.ClsMethod;
%for output
model=cell(numCls+2,1);
time=cell(numCls+1,1);

%% Clustering

fprintf('CALL: %s\n',ClsMethod);
tmptime =cputime;
switch ClsMethod
    %k-measn
    case {'litekmeans','kmeans'}
        [assign, centroid]=litekmeans(X,numCls,'MaxIter',20);
    %Spectral Clustering
    case {'SC','Spectral_Clustering'} %abbreviation of Spectral_Clsutering
        %Construct NxN adjacency matrix
        W=constructSimMat(X,Y,method.param{1});
        %spectral clustering
        [assign]=Spectral_Clustering(W,numCls,method.param{1});
        %calculate centroids of clusters
        centroid=zeros(numCls,numF);
        for i=1:numCls
            index= (assign==i);
            centroid(i,:)=mean(X(index,:));
        end
    case {'hkmeans','hierarchical_kmeans'}
        %#number of maximum instances in a cluster
        mxPts=method.param{1}.mxPts;
        % number of iterations
        iter=10;
        % number of total clusters (for inner loop)
        totalClusters=0;
        % if frac =0, not hierarchical 
        frac=1;
        % #threads to use
        numThreads=1;
        % temporary files 
        fid=fopen('tmp.txt','w');
        [assign1, totalClusters, ~] =hierKmeansFt(X',iter, numCls, mxPts,totalClusters, fid, frac, numThreads);
        assign = zeros(size(X, 1), 1);
        totalClusters
        %flatten clusters 
        clusterCount = 0;
        for i = 0:max(assign1)
            clsidx = find(assign1 == i);
            if(isempty(clsidx))
                continue
            end
            assign(clsidx) = clusterCount;
            clusterCount = clusterCount+1;
        end
        fclose(fid);
        numCls=max(assign);
        centroid=zeros(max(assign1),numF);
        for i=1:numCls
            index= (assign==i);
            centroid(i,:)=mean(X(index,:));
        end
    % Add new methods here
    otherwise
        error('%s is not implemented yet',ClsMethod);
end


%output
model{numCls+1}=assign;
model{numCls+2}=centroid;

time{end}=cputime-tmptime;
%% Call next model

fprintf('CALL: %s\n',method.name{2});
for Clscount =1:numCls
    % instance separation
    instanceindex=(assign==Clscount);
    tmpX=X(instanceindex,:);
    tmpY=Y(instanceindex,:);
    %Learning model
    [model{Clscount},time{Clscount}]=feval([method.name{2},'_train'],tmpX,tmpY,Popmethod(method));
end

