function [model] = SSL(X,Y, opts)
%  [model] = ML_LS(X,Y, opts)
%
%  Implement the high-dimensional multi-label least square formulation
%  in the KDD08 paper.
%
% X: d-by-n matrix
% Y: k-by-n matrix
% n is the number of samples.
%
%  opts is the parameter stucture. Here are related fields:
%  required:
%   - opts.r: the dimension of shared dimensions   
%   - opts.alpha: the regularization parameter for non-shared part
%   - opts.beta: the regularization parameter for 2-norm regularization
%  
%  optional:
%   - opts.Ux, dx, Vx: the SVD of X by calling SVD(X, 'econ') 
%                       This is helpful for acceleration during parameter
%                       tuning. Note that dx is a vector(not matrix) of the
%                       singular vectors to save space.
%  
%  Return model with the following fields:
%   - model.W : U in the paper, a dxk matrix
%   - model.V : the coeffficient for shared dimensions
%   - model.theta: the learned transformation
%   - model.param: the related parameter information
%   - model.method: the method name
% 
%  For prediction: Y = sign(X * model.W)
%  
%  To include bias, please add a column of 1 to X before calling the
%  function. This should be done for prediction as well.
%
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

X = X';
Y = Y';

if (~exist('opts','var'))
    opts = [];
end
opts.type = 'ssl';
if ~isfield(opts, 'r')
    opts.r = 0;
elseif opts.r<0
    opts.r = 2;
end

if ~isfield(opts, 'alpha')
    opts.alpha = 0;
elseif opts.alpha<0
    opts.alpha = 0;
end

if ~isfield(opts, 'beta')
    opts.beta = 0;
elseif opts.beta<0
    opts.beta = 0;
end

r = opts.r;
alpha = opts.alpha;
beta = opts.beta;

% if size(X,2) < 300 %low dimensional data
%     [U,theta, V] = low_dimensional_LS(X,Y,r,alpha, beta);
%     return
% end

[n, dim] = size(X);

t = min(dim,n);


if isfield(opts, 'Ux') % the SVD compuation has been done
  U1 = opts.Ux;
  dx = opts.dx;
  V1 = opts.Vx;
  disp('No SVD is performed');
  opts=rmfield(opts, {'Ux', 'dx', 'Vx'});
else % perform SVD
   % convert to full matrix if possible to save computation time 
   if  issparse(X) && n*dim < 2000*30000
     X = full(X);
   end

  if issparse(X)
    [U1, Dx, V1] = svds(X, t);
  else 
    [U1, Dx, V1] = svd(X, 'econ');
  end
  dx = diag(Dx);  
end
clear X Dx; %X would not be used anymore.


base1 = dx.^2/n + beta;  % 1/n \sigma^2 + \beta I 
base2 = base1 + alpha; 
dinv1 = 1./base1;
dinv2 = 1./base2;  % [1/n*\sigma^2 + \beta*I]^-1
d1 = dinv1.*dx;
d2 = dinv2.*dx;
d = sqrt(base2.*dinv1);
d_tilde = sqrt(d1.*d2);

D = sparse(1:t, 1:t, d);
D_tilde = sparse(1:t, 1:t, d_tilde);

C = Y'*U1*D_tilde;
disp(size(C));
tic
[P1, Lambda, P2] = svds(C, r); % select the top r eigen vectors
toc


[theta,R] = qr(V1*D*P2, 0);
theta = theta';
R_inv = R^(-1);
P2 = P2*R_inv;


%% calculate M  this part has some problem, might run out of memory
%  so we use more efficient way as presented in our paper
%   Q = X'*X/n + (alpha + beta)* speye(dim) - alpha*theta'*theta;
%   b = X'*Y/n;
%   U = Q\b;


%Use Efficient way to calculate U
D2 = sparse(1:t, 1:t, d2);
MinvXtY = V1 * (D2 * (U1' * Y));

tempd = dinv2.*d;
part1 =V1 * (sparse(1:t, 1:t, tempd) * P2);

part2 = speye(r) - alpha * P2' * sparse(1:t, 1:t, dinv2) * P2;
part2 = part2^(-1);

% note that P2' * C' = Lambda * P1';
part3 = R_inv'*Lambda * P1';

U = (MinvXtY + alpha * part1 * part2 * part3)/n;
V = theta*U;



model.W = U;
model.V = V;
model.theta = theta;
model.param = opts;
model.method = 'ML-LS';







