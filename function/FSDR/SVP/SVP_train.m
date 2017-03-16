function[model,time]=SVP_train(X,Y,method)
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

%%% Method
%% initialization
% number of reduced dim
[numN,numF]=size(X);
[numN,numL]=size(Y);
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim]);
    dim=ceil(dim);
end

%hard coding 
tol=1e-3;
%number of iterations of SVP
mxitr=200;
%verbosity
verbosity=1;
%thread
numThreads=2;
%
cost=0.1;

time=cell(2,1);
tmptime=cputime;
model=cell(2,1);
% to normalized and shift
normY = sqrt(sum((Y'.^2), 1)) + 1e-10;
tmpY = bsxfun(@rdivide, Y', normY);
tmpY=sparse(tmpY');

numk=method.param{1}.numk;
numk=min(numk,numN);
dim=min(dim,numN);
w_thresh=method.param{1}.w_thresh;
spParam=method.param{1}.sp_thresh;



[Om,OmVal,neighborIdx]=findKNN_test(tmpY',numk,numThreads);

neighborIdx=neighborIdx';
done=false;

[I,J]=ind2sub([numN numN],Om(:));
MOmega=sparse(I,J,OmVal(:),numN,numN);

while(~done)
    try
        [U, S, V]=lansvd(MOmega,dim,'L');
        Uinit=U*sqrt(S);
        Vinit=V*sqrt(S);
        [U, V]=WAltMin_asymm(Om(:), OmVal(:), mxitr, tol, Uinit, Vinit, numThreads);
	done=true;
    catch exception
         msgString = getReport(exception);
         disp(msgString);
         done = false;
    end
end

Zc = U;
Vc = V;
[W, alpha, mu] = computeW(X, Zc, 0.001, 0.01, 2, cost);
[W_I, W_J, W_lin] = find(W);
[W_I, W_J, W_lin] = find(W);
W_sort = sort(abs(W_lin));
w_idx = ceil(w_thresh*length(W_sort));
if(w_idx==0)
    w_idx = 1;
end
W_lin(abs(W_lin)<W_sort(w_idx)) = 0;
W = sparse(W_I, W_J, W_lin, size(X,2), dim);
  
Zct = full(X*W);
sp_thresh_v = zeros(dim, 1);

for sp_i= 1:dim
    [a, a_idx] = sort(abs(Zct(:, sp_i)));
    sp_thresh = a_idx(1:ceil(numN*spParam));
    Zct(sp_thresh, sp_i) = 0;
    sp_thresh_v(sp_i) = a(sp_thresh(end));
end

model{2}=W;

%Update X
tmpX=Zct;
time{end}=cputime-tmptime;
%CALL next Classifier
[model{1},time{1}]=feval([method.name{2},'_train'],tmpX,Y,Popmethod(method));


