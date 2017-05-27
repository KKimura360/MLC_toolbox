function[model,time]=HOMER_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.ClsMethod: Clustering method
% randpatrition : HOMER-R
% balancedkmeans: HOMER-B
% kmeans or litekmeans: HOMER-K
%method.param{x}.numCls:    a number of clusters
%% Output
%model: A learned model ( cell(numCls+2,1))
%model{1:numCls}: learned model on each cluster
%model{1:numCls+1}:assign vector of labels
%model{1:numCls+2}:model of this layer
%% Option
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
%% Reference (APA style from google scholar)
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD�f08) (pp. 30-44).

%%% Method
%error check 
if ~isfield(method.param{1},'numCls');
    error('HOMER needs a number of clusters\n' );  
end
if ~isfield(method.param{1},'ClsMethod')
    warning('Clustering method is not set\n we use balancedkmeans')
    method.param{1}.ClsMethod='balancedkmeans';
end 
    
if isfield(method.param{1},'sim')
   method.param{1}.sim.type='Lab-nn'; %only allowed 
end

            

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);

%numCls
numCls=method.param{1}.numCls;
if ischar(numCls)
    eval(['numCls=',method.param{1}.numCls,';']);
    numCls=ceil(numCls);
end

ClsMethod=method.param{1}.ClsMethod;
time=cell(numCls+2,1);
tmptime=cputime;

%% Clustering
%if number of labels are larger than number of clusters
%conduct clustering
if numL > numCls
    switch ClsMethod
        case {'balancedkmeans','bkmeans'}
            [assign,~]=balancedkmeans(Y',numCls,10);
        case {'litekmeasn','kmeans'}
            [assign,~]=litekmeans(Y',numCls,'MaxIter',20);
        case {'random','randpartition'}
            [assign,~]=randpartition(Y',numCls);
        case {'SC','Spectral_Clustering'}
            %NOTE: positions of X and Y are exchanged
            W=constructSimMat(X,Y',method.param{1});
            %spectral clustering
            [assign]=Spectral_Clustering(W,numCls,method.param{1});
        otherwise
            error('%s is not surpported',ClsMethod);
    end
%if number of labels are smaller, we just assign each label to a clutser
%and reset number of clusters as number of labels
else
    assign=1:numL;
    numCls=numL;
end

% keep models
model=cell(numCls+2,1);
model{numCls+1}=assign;

%% Learning model on this layer
%obtain new lable vector (belong to clutser or not)
tmpY=zeros(numN,numCls);
for i=1:numCls    
    Lind=(assign==i);
    Nind=sum(Y(:,Lind),2)>0;
    tmpY(Nind,i)=1;
end
% HOME call next model for this new MLC problem
% fprintf('CALL: %s as base MLC clssifier \n',method.name{2});
[model{numCls+2},time{end-1}]=feval([method.name{2},'_train'],X,tmpY,Popmethod(method));

time{end}=cputime-tmptime;
%% Construct Hierarchcal structure
for i=1:numCls
    % obtain instance clusters
    Lind=(assign==i);
    Nind=sum(Y(:,Lind),2)>0;
    % obtain feature matrix and label matrix for next layer
    tmpX=X(Nind,:);
    tmpY=Y(Nind,Lind);
    %if number of labels and number of clusers are same  
    if numL <= numCls
        % This layer is leaf thus end here
        model{i}='';
        time{i}=0;
    else
       % Call HOMER again with the shrinked problem
        [model{i},time{i}]=feval([method.name{1},'_train'],tmpX,tmpY,method);
    end
end