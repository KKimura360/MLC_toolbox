function[model,time]=OPLS_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(2,1))
%time : computation time
%% Reference (APA style from google scholar)
%Sun, L., Ceran, B., & Ye, J. (2010, July). A scalable two-stage approach for a class of dimensionality reduction techniques. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 313-322). ACM.
%Sun, L., Kudo, M., & Kimura, K. (2016, August). A Scalable Clustering-Based Local Multi-Label Classification Method. In ECAI 2016: 22nd European Conference on Artificial Intelligence, 29 August-2 September 2016, The Hague, The Netherlands-Including Prestigious Applications of Artificial Intelligence (PAIS 2016) (Vol. 285, p. 261). IOS Press.
%https://github.com/futuresun912/CLMLC

%%% Method
%% initialization
% number of reduced dim
% no other parameters
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end
time=cell(2,1);
tmptime=cputime;
model=cell(2,1);
% to normalized and shift
Xmean= mean(X,1);
X  = bsxfun(@minus,X,Xmean);
tmpY  = bsxfun(@minus,Y,mean(Y,2));
% to use svd(), we use non-sparse form
%NOTE: This implementation must be changed in the future
X=full(X);
tmpY=full(tmpY);

%Stage 1: solv the regularized least squares problem
[W1,S1,V1] = svd(X','econ');
r1 = rank(S1);
W1 = W1(:,1:r1); S1 = S1(1:r1,1:r1); V1 = V1(:,1:r1);
s1 = diag(S1);
U1 = W1 * diag(s1./(s1.^2+1)) * V1' * tmpY;

% Stage 2: solve the resulting optimization problem
H  = tmpY' * X * U1;
[UH,SH,~] = svd(H,'econ');
rH = min(size(Y,2),dim);
UH = UH(:,1:rH); SH = SH(1:rH,1:rH);
sH = diag(SH);
U2 = UH * diag(1./sqrt(sH));
U  = U1 * U2;

%Update X
tmpX=sparse(X * U);
model{2}=U;
time{end}=cputime-tmptime;
%CALL next Classifier
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


