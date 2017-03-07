function[model,time]=EPS_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.ClsMethod: Clustering method
%method.param{x}.numCls:  number of clusters
%method.param{x}.dim   : number of dimension reduction of OPLS
%method.param{x}.numMeta: number of meta-labels
%% Reference (APA style from google scholar)
%Sun, L., Kudo, M., & Kimura, K. (2016, August). A Scalable Clustering-Based Local Multi-Label Classification Method. In ECAI 2016: 22nd European Conference on Artificial Intelligence, 29 August-2 September 2016, The Hague, The Netherlands-Including Prestigious Applications of Artificial Intelligence (PAIS 2016) (Vol. 285, p. 261). IOS Press.

%%% Method
%% EPS utlize TREMLC frame work
%% TREMLC -> PS -> some classifier
%% Thus, we change method structure 

fprintf('EPS calls TREMLC and PS');
numLayer = length(method.param)-1;

newmethod.name=cell(numLayer+2,1);
newmethod.param=cell(numLayer+2,1);

%% TREMLC
newmethod.name{1}='TREMLC';
newmethod.param{1}.numF=method.param{1}.numF;
newmethod.param{1}.numN=method.param{1}.numF;
newmethod.param{1}.numL=method.param{1}.numL;
newmethod.param{1}.numM=method.param{1}.numM;

%% PS
newmethod.name{2}='PS';
newmethod.param{2}.type=method.param{1}.type;
newmethod.param{2}.numClass=method.param{1}.numClass;

count=0;
for i=1:numLayer
    if strcmpi(method.name{i+1},'PS')
        continue;
    end
        count=count+1;
        newmethod.name{count+2}=method.name{1+count};
        newmethod.param{count+2}=method.param{1+count};
end
newmethod.base=method.base;
newmethod.th=method.th;

newmethod.name
%Call next model
fprintf('CALL: %s\n',newmethod.name{1});
[model,time]=feval([newmethod.name{1},'_train'],X,Y,newmethod);


