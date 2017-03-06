function[model,time]=MLCC_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.ClsMethod
%method.param{x}.numCls: number of label clusters
%% Output
%model: A learned model (cell(numL,1))
%model{1:numL}: Larned classifier, corresponding to a label
%% Reference (APA style from google scholar)
%Meta-Label Classifier Chain
%Sun, L., Kudo, M., & Kimura, K. (2016, August). A Scalable Clustering-Based Local Multi-Label Classification Method. In ECAI 2016: 22nd European Conference on Artificial Intelligence, 29 August-2 September 2016, The Hague, The Netherlands-Including Prestigious Applications of Artificial Intelligence (PAIS 2016) (Vol. 285, p. 261). IOS Press.

%%% Method
%% Initialization
numCls=method.param{1}.numCls;
time=cell(numCls+1,1);
tmptime=cputime;
%Label Clustering
switch method.param{1}.ClsMethod
    case 'litekmeans'
        [assign,centroid]=litekmeans(Y','MaxIter',20);
    case 'SC'
        if strcmpi(method.param{1}.sim.type,'CLMLC')
            W=constructSimMat(X,Y,method.param{1});
            [assign]=Spectral_Clustering(W,numCls,method.param{1});
        else
            method.param{1}.sim.type='Ins-nn';
            W=constructSimMat(Y',X,method.param{1});
            [assign]=Spectral_Clustering(W,numCls,method.param{1});
        end
    otherwise
        error('%s is not supported',method.param{1}.ClsMethod);
end

%MLCC needs order of the classifier chain of meta-labels
model=cell(numCls+2,1);
%randomly obtain classifier chain (1-order)
chainorder=randperm(numCls);
%keep this at last cell of model
model{numCls+1}=chainorder;
model{numCls+2}=assign;
%Learning model
time{end}=cputime-tmptime;
fprintf('CALL: %s\n',method.name{2});
for Clscount=1:numCls
    
    if Clscount>1
        %the parent meta-label 
        ind= (assign==chainorder(Clscount-1));
        %problem transformation, adding label info. to the feature
        tmpX= [X Y(:,ind)];
    else
        % the first label does not have any parents. 
        tmpX=X;
    end
    %current meta-label
    ind=(assign==chainorder(Clscount));
    tmpY=Y(:,ind);
    % CALL next model
    [model{Clscount},time{Clscount}]=feval([method.name{2},'_train'],tmpX,tmpY,Popmethod(method));
end



