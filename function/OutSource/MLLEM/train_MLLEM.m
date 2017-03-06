function [G H V] = train_MLLEM(X,Y,params);

%%Input: 
%  params.dim : the number of  dimension of embedding space
%  params.k1: the number of k-nn @ data 
%  params.k2: the number of k-nn @ label
%  params.opsWI: option to construct instance similarity
%               1. made by L2-distance on Feature space
%               2. made by innerproduct on Label space
%  params.opsWL: option to construct label similarity
%		1.innerproduct 
%		2.Jaccard similiarity
%		3.Euclid distance
% params.alpha: weight parameter for instance locality
% params.beta : weight parameter for label locality


%%Output
% G : N x K dimensional matrix
% H : L x K dimensional matrix
% V : eigen values



%% Construct instance locality and label locality


%% If we alread construct the same similalirties.

%% Construct Instance-Instance relationship
	tic;
  
if params.opsWI==1
    WI=sparse(adjacency(X,'nn',params.k1)); %% Using adjacency function provided by http://web.cse.ohio-state.edu/~mbelkin/algorithms/algorithms.html
	WI(WI>0)=1;                     
elseif params.opsWI==2 
    WI=sparse(zeros(size(X,1)));
    tmp=Y*Y';  % N x N innerproduct matrix
	tmp=tmp-diag(diag(tmp)); %Delete diagnoal elements
	[tmpmat, tmpind]=sort(tmp,2,'descend'); %% find k-largest innerproducts
	for i=1:size(tmp,1);
		WI(i,tmpind(i,1:params.k1))=1;
		WI(tmpind(i,1:params.k1),i)=1;
    end
else
    error('Wrong parameter pramas.WI')
    
end
tmptime=toc;
fprintf('Done WI: time: %1.4f \n',tmptime);

%% Construct Label-Label Relationship
tic;
WL=sparse(zeros(size(Y,2)));
if params.opsWL==1
	tmp=Y'*Y;  % N x N innerproduct matrix
	tmp=tmp-diag(diag(tmp)); %Delete diagnoal elements
	[tmpmat, tmpind]=sort(tmp,2,'descend');
    for i=1:size(Y,2);
        WL(i,tmpind(i,1:params.k2))=1;
        WL(tmpind(i,1:params.k2),i)=1;
    end
elseif params.opsWL==2

    tmp=squareform(pdist(Y','jaccard'));
    tmp=tmp+eye(size(tmp,1));
    [tmpmat tmpind]=sort(tmp,2);
	for i=1:size(Y,2);
        WL(i,tmpind(i,1:params.k2))=1;
		WL(tmpind(i,1:params.k2),i)=1;
    end
    
elseif params.opsWL==3
        fprintf('L2_distance is selected');
		WL=adjacency(Y','nn',params.k1);
		WL(WL>0)=1;
		WL=sparse(WL);
else
   error('Wrong parameter: pramas.WL')
end
tmptime=toc;
fprintf('Done WL: time: %1.4f \n',tmptime);

%% Laplcian Eigen Map
tic;
W=[ (params.alpha * WI) Y; Y' (params.beta *WL)];
D=sum(W(:,:),2);
L=spdiags(D,0,speye(size(W,1)))-W;
        
L=L+eye(size(L,1),size(L,2))*1e-09; % to avoid singularity 
opts.tol=1e-9;
opts.issym=1;
opts.disp=0;
[E V]=eigs(L,(params.dim+2),'sm',opts);
% if params.alpha= params.beta =0, there are 2 eigen vectors with 0 eigen values.
% Excluding eigen vectors with 0 eigen values
zeroind=find(diag(V)<1e-06);
E(:,zeroind)=[];
G=E(1:size(X,1),end-params.dim+1:end);
H=E(size(X,1)+1:end,end-params.dim+1:end);
    

tmptime=toc;
fprintf('Done Laplacian Eigen Map: time: %1.4f \n',tmptime);

