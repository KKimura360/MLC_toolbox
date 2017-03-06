
function[P D] = MLDA(X,Y,K);

opts.tol=1e-2;
X=X'; %% feature x data matrix  Y, data x label
tmpvec=sum(Y);
Y(:,find(tmpvec==0))=[];
M= X * (Y * ones(size(Y,2),1));
M = M./ sum(sum(Y));
X= X- M * ones(size(M,2),size(X,2));

tmp=normalize_factor(Y,1);
C=Y'*Y;
Z=Y*C;
tmp= diag(1./sum(Y));
Z= Z *tmp;

 
W=diag(1./sum(Z));
L=diag(sum(Z,2));

Sb=(X*Z)*W* (Z' *X');
Sw= X * L*X';

T=pinv(Sw-Sb) * Sb;
[P D]=eigs(T,K,'lm',opts);


