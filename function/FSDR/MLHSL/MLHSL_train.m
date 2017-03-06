function[model,time]=MLHSL_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%method:
%method.param{x}.beta: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%method.param{x}.type  : 
%% Output
%model: A learned model (cell(dim+2,1))
%% Reference (APA style from google scholar)
%Sun, L., Ji, S., & Ye, J. (2008, August). Hypergraph spectral learning for multi-label classification. In Proceedings of the 14th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 668-676). ACM.
%%% Method

%% Initialization
[numN numF]=size(X);
[numNL,numL]=size(Y);
%reduced dimension
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end

if dim >= numF
    error('the number of dim is larger than original dim')
end

type=method.param{1}.type;
lambda=method.param{1}.lambda;
model=cell(3,1);
time=cell(2,1);
tmptime=cputime;

%Learning model
switch type
    case 'clique'
        H = clique_expn_fact(Y);
    case 'star'
        H = star_expn_fact(Y);
    case  'zhou'  
        H = Zhou_Laplacian_fact(Y);
    otherwise 
        error('%s is not supported',type);
end

H = rowCenter(H);
W = solve_eig(X', H, lambda, dim);

% CALL base classfier
tmpX= X*W ;
model{3}=W;
time{end}=cputime-tmptime;

[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


