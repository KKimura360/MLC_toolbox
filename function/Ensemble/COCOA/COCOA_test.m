function[conf, time]=COCOA_test(X,Y,Xt,model,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%Xt:Feature Matrix (NtxF) for test data
%% Output
%conf: confidence values (Nt x L);
%% Reference (APA style from google scholar)
%Zhang, M. L., Li, Y. K., & Liu, X. Y. (2015, July). Towards Class-Imbalance Aware Multi-Label Learning. In IJCAI (pp. 4041-4047).

%%% method

%% initialization
[numNt,~]=size(Xt);
[numN,numL]=size(Y);
%%% Method
time=cell(3,1);
time{end}=0;
conf=zeros(numNt,numL);
%% COCOA =  BR + triClass
%BR, set paramters    
newmethod.name{1}='BR';
newmethod.param=method.param{1}.BR.param;
newmethod.base=method.param{1}.BR.base;
newmethod.th=method.param{1}.BR.th;
%call BR
[tmpconf,time{1}]=feval([newmethod.name{1},'_test'],X,Y,Xt,model{1},newmethod);
 conf=conf+tmpconf;   
%triClass, set parameters
newmethod.name{1}='triClass';
%triClass calls next model thus set next models correctly
count=0;
for i=1:length(method.param{1}.Tri.name)
    %Some users may set 'triClass' in the param.Tri.name{1}
    if strcmpi(method.param{1}.Tri.name{i},'triClass')
        continue;
    end
    count=count+1;
    newmethod.name{count+1}=method.param{1}.Tri.name{i};
    newmethod.param{count+1}=method.param{i};
end

newmethod.param{1}=method.param{1}.Tri.param;
newmethod.base=method.param{1}.Tri.base;
newmethod.th=method.param{1}.Tri.th;
%call triClass
[tmpconf,time{2}]=feval([newmethod.name{1},'_test'],X,Y,Xt,model{2},newmethod);
conf=tmpconf+conf;

conf=conf/2;








