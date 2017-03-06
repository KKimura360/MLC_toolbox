function[model]=FaHOMER_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.numCls:    a number of clusters
%method.param{x}.lambda:  for the ridgre regression used for the model
%method.param{x}.alpha :  a weight paramter for instance balance
%method.param{x}.beta  ;  a weight parameter for label balance
%% Output
%model: A learned model ( cell(numCls+2,1))
%model{1:numCls}: learned model on each cluster
%model{1:numCls+1}:assign vector of labels
%model{1:numCls+2}:model of this layer
%% Option
%method.param.Clsparam: parameter for the clustering % see clustering
%method file 
%% Reference (APA style from google scholar)
%Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008, September). Effective and efficient multilabel classification in domains with large number of labels. In Proc. ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMDÅf08) (pp. 30-44).

%% Method
%error check 
if ~isfield(method.param{1},'numCls');
    error('FaHOMER needs a number of clusters\n' );  
end
 
    
%Initialization
numCls=method.param{1}.numCls;
lambda=method.param{1}.lambda;
alpha=method.param{1}.alpha;
beta=method.param{1}.beta;

%obtain the data info
[numN numF]=size(X);
[numNL,numL]=size(Y);
numL
%size check
sizeCheck;

%Clustering and model learning at the same time 
%Here, in this implementation, we consider ridge regression only for the
%model


%Ridge regression following not change while iterations
XX = [ones(numN, 1) X];
invX=inv(XX' * XX + lambda * eye(size(XX, 2))) * XX';

% if  the number of labels is smalle then the number of clusters
if numL <= numCls
    %assign labels
    assign=1:numL;
    % change the number of clsters size
    numCls=numL;
    %indicator matrix
    indMat=zeros(numL,numCls);
    for label=1:numL
     indMat(label,assign(label))=1;
    end
    %obtain target matrix V
    V=Y*indMat;
    V(V>0)=1;
    % obtain weight matrix
    W=invX*V;
else
    %main part of clustering
    % assign initialization
    % we consider balancing initalization,  we assign numCls labels a
    % time and repeat this procedure until all labels are assigned
    %keep the number of labels
    tmpL=numL;
    %index of label
    stInd=1;
    % initialize the assign vector
    assign=zeros(numL,1);
while(1)
    %if more than numCls labels are remaining
    if (tmpL-numCls)>=0
        %pick numCls labels and assign randomly 
        assign(stInd:(stInd+numCls-1))=randperm(numCls,numCls);
        stInd=stInd+numCls;
        tmpL=tmpL-numCls;
    %if remaining labels are less than numCls
    else
        % assign remaining labels randomly
        assign(stInd:end)=randperm(numCls,length(assign(stInd:end)));
        break;
    end
end

%indicator matrix
indMat=zeros(numL,numCls); % indMat is C in the slide
for label=1:numL
    indMat(label,assign(label))=1;
end

%obtain matrix multiplication of label matrix and indicator matrix
%it is neccesary to update boolean matrix multiplication, since we need to
%keep counts 
V=Y*indMat;
%practical setting,number of iterations, it will be removed or considered as a parameter
maxiter=20; 

for iter=1:maxiter
    %learning model
    %obtain target matrix for the model
    Z=V;    Z(Z>0)=1;
    %obtain model
    W=invX*Z;
    %obtain prediction results 
    XW= XX*W;
    %assign labels to clusters
    for label=1:numL
        %update matrix multiplication (to be matrix multilpication with out
        %label-th vectors of Y and indMat
        V=V- (Y(:,label)*indMat(label,:));
        %update indicator matrix 
        indMat(label,assign(label))=0;
        %initialization of a cost vector
        costvec=zeros(numCls,1);
        for k=1:numCls
            %calc a mask vector
            % this contains, instances are already assigend the the clutser
            % or not 
            % if mask(n)=1, nth instance is already assigned to this clsuster because of the other labels
            % if mask(n)=0, nth instance is not assigend to this cluster
            % yet
            %NOTE: HOMER duplicates instances among different clusters,
            %instances will be assigned to clusters if at least one labels are assigned to clusters 
            mask=V(:,k);
            mask(mask>0)=1;
            
            %calculate emprical risk
            tmp1st=XW(:,k)-Y(:,label);
            %remove insatnces which are already assigned by
            %because it does not matter, either this label is assigned or
            %not
            tmp1st(logical(mask))=0;
            %calculate the value of loss function (it depends on the chosen loss function)
            costvec(k)=norm(tmp1st,2)^2;
            
            %calculate instance-size balancing cost term
            tmp2nd= Y(:,label)+mask;
            %number of instances assigend to this cluster
            tmp2nd=sum(tmp2nd>0);
            
            %add cost with weighting parameter alpha
            costvec(k)=costvec(k) + alpha*tmp2nd;     
            %calculate label-size balancing cost term
            tmp3rd=(sum(indMat(:,k)))^2;
            %add cost with weighting parameter beta
            costvec(k)=costvec(k) + beta*tmp3rd;
        end
        %assign label to the minimum cost cluster
        [~,assign(label)]=min(costvec);
        % update indicator matrix
        indMat(label,assign(label))=1;
        % update matrix multiplication
        V=V+ (Y(:,label)*indMat(label,:));
    end
end

end
%output
model=cell(numCls+2,1);
model{numCls+1}=assign;

[model{numCls+2}]=W;

%Construct Hierarchcal structure
for i=1:numCls
    Lind=(assign==i);
    Nind=sum(Y(:,Lind),2)>0;
    tmpX=X(Nind,:);
    tmpY=Y(Nind,Lind);
    if numL == numCls || numL==0
        % end hierarchy here
        [model{i}]=NaN;
    else
        %Call HOMER again with the shrinked problem
        [model{i}]=feval([method.name{1},'_train'],tmpX,tmpY,method);
    end
end



