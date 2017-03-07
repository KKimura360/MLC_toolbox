function[model,time]=COCOA_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.BR is paramters for BR
%method.param{x}.Tri is parameters for triClass
%% Output
%model: A learned model (cell(method.param{x}.numM,1))
%model{1:numM}: MLC classifiers (depends on called method)
%% Reference (APA style from google scholar)
%Zhang, M. L., Li, Y. K., & Liu, X. Y. (2015, July). Towards Class-Imbalance Aware Multi-Label Learning. In IJCAI (pp. 4041-4047).

%%% Method
time=cell(3,1);
time{end}=0;
%% COCOA = Ensemble of BR + triClass
%BR, set paramters    
newmethod.name{1}='BR';
newmethod.param=method.param{1}.BR.param;
newmethod.base=method.param{1}.BR.base;
newmethod.th=method.param{1}.BR.th;
%call BR
[model{1},time{1}]=feval([newmethod.name{1},'_train'],X,Y,newmethod);
    
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
[model{2},time{2}]=feval([newmethod.name{1},'_train'],X,Y,newmethod);

