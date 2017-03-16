function[model,time]=MLDA_train(X,Y,method)
%
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: regularization parameter
%method.param{x}.dim:  lower-dim of labels
%% Output
%model: A learned model (cell(2,1))
%time : computation time
%
%% Reference (APA style from google scholar)
%Wang, H., Ding, C., & Huang, H. (2010, September). Multi-label linear discriminant analysis. In European Conference on Computer Vision (pp. 126-139). Springer Berlin Heidelberg.

%%% Method

%% initialization
dim=method.param{1}.dim;
gamma=method.param{1}.gamma;
[numN,numF] = size(X);
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
if dim >= numF
    error('the number of dim is larger than original dim')
end
time=cell(2,1);
tmptime=cputime;
model=cell(3,1);

%% Learning model
newY  = Y(:,any(Y,1)); 
C     = 1 - pdist(newY','cosine');
C(isnan(C)) = 0;
C     = sparse(squareform(C));
C(logical(eye(size(C)))) = 1;
Yc    = newY * C;
Z     = bsxfun(@rdivide,Yc,sum(newY,2));
m     = sum(Yc'*X,1) ./ sum(sum(Yc));
tmpX    = X - ones(numN,1)*m;
numL  = size(newY,2);
W     = sparse(1:numL,1:numL,sum(Z,1).^-1,numL,numL);
L     = sparse(1:numN,1:numN,sum(Z,2),numN,numN);
Sxz   = tmpX' * Z;
Sb    = Sxz * W * Sxz';
St    = tmpX' * L * tmpX + gamma.*speye(numF);
[U,~] = eigs(Sb,St,dim); 

%% Feature projection
tmpX     = sparse(tmpX * U);
model{2} = U;
model{3} = m;
time{end}=cputime-tmptime;

%% CALL next Classifier
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));