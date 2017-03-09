function[model,time]=MLLEM_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method.param{x}.
%dim:  reduced dim
%opsWI: 1,2 to obtain Instance-Instance relationship
%opsWL: 1,2,3 to obtain Label-Label relationship
%type : 'L' or 'NL', to map test instances
%k1   : number of k-nn for Instance-Instance
%k2   : number of k-nn for Label-Label
%k3   : number of k-nn for Nonlinear simulation (only for NL)
%SCtype= 1,2,3  Laplacian type, unnormalized, normalized by shi, normalized
%by jordan
%alpha: weighting parameter for Instance-Instance
%beta : wrighting paramter for Label-Label
%lambda: ridge paramter for Linear simulation (only for L) 
%% Output
%model: weight matrix
%time: computation time
%% Reference
% Kimura, K., Kudo, M., & Sun, L. (2016, November). Simultaneous Nonlinear Label-Instance Embedding for Multi-label Classification. In Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR) and Structural and Syntactic Pattern Recognition (SSPR) (pp. 15-25). Springer International Publishing.

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
model=cell(numL,1);
time=cell(numL+1,1);
tmptime=cputime;
%paramters 
dim=method.param{1}.dim;
opsWI=method.param{1}.opsWI;
opsWL=method.param{1}.opsWL;
k1=method.param{1}.k1;
k2=method.param{1}.k2;
type=method.param{1}.type;
SCtype=method.param{1}.SCtype;
alpha=method.param{1}.alpha;
beta=method.param{1}.beta;
lambda=method.param{1}.lambda;

%% error check 
if numN < k1
    k1=numN;
end
if numL < k2
    k2=numL;
end


%Learning model
%Instance-Instance Relationship
switch opsWI
    case 1 %Nearest neighbor on feature space
        WI=sparse(adjacency(X,'nn',k1)); %% Using adjacency function provided by http://web.cse.ohio-state.edu/~mbelkin/algorithms/algorithms.html
        WI(WI>0)=1;  
    case 2 %Nearest neichbor on label space
        WI=sparse(zeros(size(X,1)));
        tmp=Y*Y';  % N x N innerproduct matrix
        tmp=tmp-diag(diag(tmp)); %Delete diagnoal elements
        [tmpmat, tmpind]=sort(tmp,2,'descend'); %% find k-largest innerproducts
        %k-nn 
        for i=1:numN
            %flag k-nearest neighbors
            WI(i,tmpind(i,1:k1))=1;
            WI(tmpind(i,1:k1),i)=1;
            %delete pseudo k-nn (othogonal instances)
            tmpvec=tmpmat(i,:);
            zerovec=(tmpvec<=0);
            WI(i,zerovec)=0;
            WI(zerovec,i)=0;
        end        
    otherwise
        error('opsWI is not surrported')
end

%Label-Label Relationship
WL=sparse(numL,numL);

switch opsWL
    case 1 % inner product 
        tmp=Y'*Y;  % N x N innerproduct matrix
        tmp=tmp-diag(diag(tmp)); %Delete diagnoal elements
        [tmpmat, tmpind]=sort(tmp,2,'descend');
        for i=1:numL;
            WL(i,tmpind(i,1:k2))=1;
            WL(tmpind(i,1:k2),i)=1;
            tmpvec=tmpmat(i,:);
            zerovec=(tmpvec<=0);
            WL(i,zerovec)=0;
            WL(zerovec,i)=0;
        end
    case 2 % jaccard similarity 
        tmp=squareform(pdist(Y','jaccard'));
        tmp=tmp+eye(size(tmp,1));
        [tmpmat tmpind]=sort(tmp,2);
        for i=1:numL;
            WL(i,tmpind(i,1:k2))=1;
            WL(tmpind(i,1:k2),i)=1;
            tmpvec=tmpmat(i,:);
            zerovec=(tmpvec==1);
            WL(i,zerovec)=0;
            WL(zerovec,i)=0;
        end
    case 3 % Euclidean distance
		WL=adjacency(Y','nn',k2);
		WL(WL>0)=1;
		WL=sparse(WL);
    otherwise
        error('opsWL is not supported')
end

% Laplacian eigen map
W=[ (alpha * WI) Y; Y' (beta *WL)];
sumW=sum(W,2);

D=spdiags(sumW,0,size(W,1),size(W,2));
L= D - W;

switch SCtype
    case 2
        sumW(sumW==0)=1e-08;
        D =spdiags(1./sumW,0,size(W,1),size(W,1));
        L= D * L;
    case 3 
        sumW(sumW==0)=1e-08;
        D=spdiags(1./(sumW.^0.5), 0, size(D, 1), size(D, 2));
        L = D * L * D;
end

L=L+eye(size(L,1),size(L,2))*1e-09;

opts.tol=1e-9;
opts.issym=1;
opts.disp=0;

[E,V]=eigs(L,(dim+2),'sm',opts);

zeroind=find(diag(V)<1e-06);
E(:,zeroind)=[];

if size(E,2) < dim
    dim=size(E,2);
end

G=E(1:numN,end-dim+1:end);
H=E(numN+1:end,end-dim+1:end);

model=cell(3,1);
switch type
    case {'L','linear','Linear'}
        XX=[ones(numN,1),X];
        W = inv(XX'*XX + lambda*eye(size(XX,2))) * XX' * G;
        model{1}=W;
    case {'NL','nonlinear','Nonlinear','NonLinear'}
        %kd-tree can be constructed here 
        model{1}='';
    otherwise
        error('type is not surpported')
end
model{2}=G;
model{3}=H;
time{end}=cputime-tmptime;
