function W= DMDDM(X, Y, opts)
% function W= MDDM(X, Y, opts)
% MDDM: Multilabel-Dimensional reduction via Dependecies Maximization 
% Only the projection for X is computed.
%
%
% Usage:
%     W = MDDM(X, Y, opts)
%     W = MDDM(X, Y)
% 
%    Input:
%        X       - Data matrix X. Each column of X is a data point. (d-by-n matrix)
%        Y       - Data matrix Y. Each column of Y is a data point. (k-by-n matrix)
%                 - n is number of samples, d is the X data dimensionality, and k is the Y data dimensionality.
%       
%    Output:
%        W: each column is a projection vector for X. (d-by-k matrix)
% 
%    Examples:
%        X = rand(15,10);
%        Y = rand(3, 10);
%        opts.reg_eig = 0.5;
%        W = OPLS(X, Y, opts);
% 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%
% 

% Step 1. Check the input opts using dr_opts function
if (~exist('opts','var'))
    opts = [];
end
opts.type = 'mddm';
opts = dr_opts(opts);

W = [];
if size(X,2)~=size(Y,2)
    disp('The numbers of samples in X and Y are not equal!');
    return;
end

% Centering of X


% Alternatingly optimize matrices Wx Wy 
sWx=cell(5,1);
sWy=sWx;
sDy=sWx;
sDx=sWx;
Wx= eye(size(X,1)); 
%Yt=pinv(Y);
[D N] = size(X);
for iter = 1:5
% First step learn Wy (Label Structure)  
   tmpX= Wx' * X;
   if  strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr')     
    Wy= TwoStage_DR( cenY,tmpX, opts); 
    else
    K= tmpX' * tmpX;
    tmpK = K - repmat(mean(K,1),N,1);
    HKH = tmpK - repmat(mean(tmpK,2),1,N);
% B= (Yt' * L) * Yt;
    D= (Y  * HKH) * Y';
    opts.k2=floor(size(Y,1)*0.9);
    [Wy tmp] = eigs(D,opts.k2,'lm');
    Wy=real(Wy);
   end
   sDy{iter}=D;
    sWy{iter}=Wy;
% Second step learng Wx    
    tmpY = Wy' * Y;
    if strcmpi(opts.alg, '2s') || strcmpi(opts.alg, '2s-lsqr') 
         Wx = TwoStage_DR(cenX, tmpY, opts); 
    else
    L= tmpY' *tmpY;
    tmpL = L - repmat(mean(L,1),N,1);
    HLH = tmpL - repmat(mean(tmpL,2),1,N);
    D= (X  * HLH) * X';
   opts.k=size(Y,1);
  %  opts.k=floor(size(X,1)*0.3);
    [Wx tmp] = eigs(D,opts.k,'lm');
    Wx=real(Wx);
    end
    sDx{iter}=D;
    sWx{iter}=Wx;
end
W= Wx;


save test.mat sWy sWx sDy sDx ;

    
    
end