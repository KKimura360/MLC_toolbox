function[conf,time]=CLMLC_test(X,Y,Xt,model,method)
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
        newmethod.param{2}.sim.type=method.param{1}.sim;
        newmethod.param{2}.SCtype=method.param{1}.SCtype;
end

%% MLCC
newmethod.name{3}='MLCC';
newmethod.param{3}.ClsMethod='SC';
newmethod.param{3}.numCls=method.param{1}.numMeta;
newmethod.param{3}.SCtype=1;
newmethod.param{3}.sim.type='CLMLC';

for i=1:numLayer   
    newmethod.name{i+3}=method.name{1+i};
    newmethod.param{i+3}=method.param{1+i};
end
%% Base and Threshold 
newmethod.base=method.base;
newmethod.th=method.th;

[conf,time]=feval([newmethod.name{1},'_test'],X,Y,Xt,model,newmethod);
   


