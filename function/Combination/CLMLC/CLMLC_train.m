function[model,time]=CLMLC_train(X,Y,method)
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
%% CLMLC is a combination method as follows
%% OPLS -> CBMLC -> MLCC -> some classifier
%% Thus, we change method structure 

fprintf('CLMLC calls OPLS,CBMLC and MLCC');
numLayer = length(method.param)-1;

newmethod.name=cell(numLayer+3,1);
newmethod.param=cell(numLayer+3,1);

%% OPLS
newmethod.name{1}='OPLS';
newmethod.param{1}.dim=method.param{1}.dim;

%% CBMLC
newmethod.name{2}='CBMLC';
newmethod.param{2}.ClsMethod=method.param{1}.ClsMethod;
newmethod.param{2}.numCls=method.param{1}.numCls;
switch newmethod.param{2}.ClsMethod
    case 'SC'
        newmethod.param{2}.sim=method.param{1}.sim;
        newmethod.param{2}.SCtype=method.param{1}.SCtype;
end

%% MLCC (Meta-Label Classifier Chain)
newmethod.name{3}='MLCC';
newmethod.param{3}.ClsMethod='SC';
newmethod.param{3}.numCls=method.param{1}.numMeta;
newmethod.param{3}.SCtype=1;
newmethod.param{3}.sim.type='CLMLC';

for i=1:numLayer   
    newmethod.name{i+3}=method.name{1+i};
    newmethod.param{i+3}=method.param{1+i};
end
newmethod.base=method.base;
newmethod.th=method.th;


%Call next model
fprintf('CALL: %s\n',newmethod.name{1});
[model,time]=feval([newmethod.name{1},'_train'],X,Y,newmethod);


