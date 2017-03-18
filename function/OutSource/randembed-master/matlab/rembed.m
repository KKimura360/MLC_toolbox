function retval = rembed(Xtic, W, Ytic, k, varargin)
% retval = rembed(Xtic, W, Ytic, k) 

% approximate the top-k right singular subspace of (P_X Y),
% where P_X is the projection onto the left singular subspace of X,
% using randomized methods. The two views are represented
% by the matrices Xtic and Ytic.  Xtic and Ytic are (sparse) 
% design matrices whose rows are features and whose columns 
% are examples.  W is a column vector of importance weights.
%
% The return value is a struct containing the following fields:
%
%    projectx: function pointer to project features
%    projecty: function pointer to project labels
%    Wx: projection matrix for features (if trainpredictor=true)
%    bx: mean shift vector for features (if trainpredictor=true)
%    Wy: projection matrix for labels
%    by: mean shift vector for labels
%    sigma: vector of canonical correlations
%    usedX: if compression enabled, list of features
%
% Use the projectx and projecty function pointers to project data
% and you get the mean shift and compression as appropriate.  See
% opts.compress below for the definition of compression.
%
% retval = rembed(Xtic, W, Ytic, k, opts) takes an additional struct
% with extra options.
%
%   opts.verbose: if true, display progress information. default false.
%   opts.trainpredictor: if true, train predictor of embedding. default true.
%   opts.lambda: regularization parameter.  default is 1. 
%   opts.meanshift: if true, automatically subtract mean from
%                   Xtic and Ytic (without destroying sparsity).
%                   default is false.
%   opts.innerloop: number of passes in the inner loop of ALS.  
%                   increasing this might improve results, but
%                   will slow running time. default is 10.
%   opts.tmax: number of passes for the range finder.  default is 1.
%              2 is sometimes better.  more than 2 rarely helps.
%   opts.pre: ALS preconditioner.  possible values are:
%        'diag': use diagonal preconditioner (the default).
%        'identity': use no preconditioner.
%        @(Z,d,s): any function handle.  Z is the matrix to precondition.
%                  d is a (vector) diagonal of the view covariance.
%                  s is '1' for left view and '2' for right view.
%   opts.p: oversampling parameter.  default is 20.  larger 
%           values might improve results, but the default works well.
%   opts.compress: detect and ignore unused features.  default is false.
%                  when using hashing to generate features, this can save
%                  substantial memory.
%   opts.kbs: column block size for MEX-accelerated matrix operations.
%             smaller block sizes save memory but run slower.  default
%             block size is number of latent dimensions.
%   opts.bs: example block size for matrix operations.  smaller 
%            block sizes save memory but run slower.  default block
%            size is number of examples.
%   opts.precision: specify 'single' to attempt to save memory by hoping 
%                   that numerical difficulties do not arise.  default is
%                   'double'.
%
% HINT: opts.innerloop, opts.tmax, and opt.pre default values are tuned for 
% text problems.  For non-text data the first thing you should try is 
% a different preconditioner and/or increasing innerloop.

    if (nargin < 4)
      error('rembed:noargs', 'not enough arguments provided to function ''rembed''');
    end

    start=clock;

    if (exist('sparsequad','file') == 3 && ...
        exist('dmsm','file') == 3 && ...
        exist('sparseweightedsum','file') == 3)
      havemex=true;
    else
      havemex=false;
      if (~exist('OCTAVE_VERSION','builtin'))
        warning('rembed:nomex', 'MEX acceleration not available, have you compiled the mex?');
      end
    end

    [dx,nx]=size(Xtic);
    [dy,ny]=size(Ytic);
    [~,nw]=size(W);
    sumw=sum(W);

    if (nx ~= ny || ny ~= nw)
      error('rembed:shapeChk', 'arguments have incompatible shape');
    end

    k=min([dx;dy;k]);
    [lambda,p,tmax,innerloop,bs,kbs,compress,...
     meanshift,verbose,trainpredictor,precision]=parseArgs(nw,k,varargin{:});

    if (compress)
      if (verbose)
        startcompress=clock;
        fprintf('compressing features ...');
      end

      usedX=find(any(Xtic,2));
      [udx,~]=size(usedX);
      if (udx < dx)
        X=Xtic'; clear Xtic; Xtic=X(:,usedX)'; clear X;
        dx=udx;
      end
      k=min([dx;dy;k]);

      if (verbose)
        fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startcompress), etime(clock,start));
      end
    end

    kp=min([dx;dy;k+p]);

    if (verbose)
      startpreprocess=clock;
      fprintf('preprocessing data...');
    end

    meanX=zeros(1,dx);
    if (havemex && issparse(Xtic))
       if (meanshift)
         meanX=sparseweightedsum(Xtic,W,1)/sumw;
       end
       dXX=sparseweightedsum(Xtic,W,2)-sumw*meanX.*meanX;
    else
       if (meanshift)
         meanX=full(sum(bsxfun(@times,Xtic,W),2)')/sumw;
       end
       dXX=sum(bsxfun(@times,Xtic.*Xtic,W),2)'-sumw*meanX.*meanX;
    end  
    
    meanY=zeros(1,dy);
    if (havemex && issparse(Ytic))
      if (meanshift)
        meanY=sparseweightedsum(Ytic,W,1)/sumw;
      end
      dYY=sparseweightedsum(Ytic,W,2)-sumw*meanY.*meanY;
    else
      if (meanshift)
        meanY=full(sum(bsxfun(@times,Ytic,W),2)')/sumw;
      end
      if (~exist('OCTAVE_VERSION','builtin'))
        dYY=sum(bsxfun(@times,Ytic.*Ytic,W),2)'-sumw*meanY.*meanY;
      else
        % for some reason, octave runs out of memory with bsxfun here (?)
        dYY=-sumw*(meanY.*meanY);
        for ii=1:size(W,2)
          dYY=dYY+W(1,ii)*(Ytic(:,ii).*Ytic(:,ii))';
        end
      end
    end

    cx=lambda*sum(dXX)/dx;
    cy=lambda*sum(dYY)/dy;

    dXX=dXX+cx;
    dYY=dYY+cy;

    if (verbose)
      fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startpreprocess), etime(clock,start));
    end

    XticY=@(Z) XticYimpl(Z,bs,kbs,Xtic,meanX,Ytic,meanY,W,sumw,havemex);
    YticX=@(Z) XticYimpl(Z,bs,kbs,Ytic,meanY,Xtic,meanX,W,sumw,havemex);
    XticX=@(Z) XticYimpl(Z,bs,kbs,Xtic,meanX,Xtic,meanX,W,sumw,havemex);
    YticY=@(Z) XticYimpl(Z,bs,kbs,Ytic,meanY,Ytic,meanY,W,sumw,havemex);
    
    [preleft,preright]=parsePreconditioner(Xtic,Ytic,dXX,dYY,cx,cy,varargin{:});
    
    if (verbose)
      startinitialize=clock;
      fprintf('initializing basis...');
    end

    QY=initialize(Xtic,W,Ytic,XticX,dx,cx,YticY,dy,cy,kp,precision,varargin{:});
    if (size(varargin,1) == 1 && isfield(varargin{1}, 'project'))
        QY=varargin{1}.project(QY, 2);
    end

    if (verbose)
      fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startinitialize), etime(clock,start));
    end

    % 1. randomized range finder for Y'*(P_X Y)

    for ii=1:tmax
      if (verbose)
        startrangefind=clock;
        fprintf('rangefinding iteration...');
      end

      QX=cheesypcg(@(Z) XticX(Z)+cx*Z, preleft, zeros(kp,dx), XticY(QY), innerloop, verbose); clear QY; 
      YX=YticX(QX); clear QX;
      QY=modgs(YX,@(Z) Z*Z'); clear YX;

      if (verbose)
        fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startrangefind), etime(clock,start));
      end
    end

    % 2. final optimization in reduced basis
    
    if (verbose)
      startfinalopt=clock;
      fprintf('final optimization...');
    end

    if (tmax >= 0)
      QX=cheesypcg(@(Z) XticX(Z)+cx*Z, preleft, zeros(kp,dx), XticY(QY), innerloop, verbose); clear QY;
      T=YticX(QX); clear QX;
      [V,Ssq]=eig(T*T');
      ssq=diag(Ssq);
      [~,sind]=sort(ssq,'descend');
      V=V(:,sind);
      ssq=ssq(sind);
      sigma=sqrt(max(ssq(1:k),0))';
      maxsigma=max(sigma);
      tinysigma=1e-12*maxsigma.^2;
      invsigma=sigma./(tinysigma+sigma.^2);
      %T'=U*S*V'
      %U=T'*V*S^{-1}
      Wy=bsxfun(@times,T'*V(:,1:k),invsigma); clear T V;
      by=meanY*Wy;
    else % "compressed sensing mode"
      sigma=[];
      Wy=randn(dy,k);
      by=meanY*Wy;
    end

    if (verbose)
      fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startfinalopt), etime(clock,start));
    end
   
    % 3. least squares fit of X to projectY

    if (trainpredictor)
      if (verbose)
        startfit=clock;
        fprintf('fitting projection...');
      end
      Wx=cheesypcg(@(Z) XticX(Z)+1e-6*cx*Z, preleft, zeros(k,dx), XticY(Wy'), innerloop, verbose)'; 
      bx=meanX*Wx;
      if (verbose)
        fprintf(' stage time: %g (sec), total time: %g (sec)\n', etime(clock,startfit), etime(clock,start));
      end
    end

    if (compress)
      if (trainpredictor)
        retval=struct('Wx',Wx,'bx',bx,'Wy',Wy,'by',by,'sigma',sigma/sumw,...
                      'projectx',@(Z) project(Z,Wx,bx,usedX), ...
                      'projecty',@(Z) project(Z,Wy,by), ...
                      'usedX',usedX);
      else
        retval=struct('Wy',Wy,'by',by,'sigma',sigma/sumw,...
                      'projecty',@(Z) project(Z,Wy,by));
      end
    else
      if (trainpredictor)
        retval=struct('Wx',Wx,'bx',bx,'Wy',Wy,'by',by,'sigma',sigma/sumw,...
                      'projectx',@(Z) project(Z,Wx,bx), ...
                      'projecty',@(Z) project(Z,Wy,by));

      else
        retval=struct('Wy',Wy,'by',by,'sigma',sigma/sumw,...
                      'projecty',@(Z) project(Z,Wy,by));
      end 
    end
end

function [lambda,p,tmax,innerloop,bs,kbs,compress,meanshift,verbose,trainpredictor,precision] = parseArgs(n,k,varargin)
  lambda=1;
  if (size(varargin,1) == 1 && isfield(varargin{1},'lambda'))
    lambda=varargin{1}.lambda;
  end   
  p=20;
  if (size(varargin,1) == 1 && isfield(varargin{1},'p'))
    p=varargin{1}.p;
  end
  tmax=1;
  if (size(varargin,1) == 1 && isfield(varargin{1},'tmax'))
    tmax=varargin{1}.tmax;
  end
  innerloop=10;
  if (size(varargin,1) == 1 && isfield(varargin{1},'innerloop'))
    innerloop=varargin{1}.innerloop;
  end
  bs=n;
  if (size(varargin,1) == 1 && isfield(varargin{1},'bs'))
    bs=varargin{1}.bs;
  end
  kbs=k+p;
  if (size(varargin,1) == 1 && isfield(varargin{1},'kbs'))
    kbs=varargin{1}.kbs;
  end
  compress=false;
  if (size(varargin,1) == 1 && isfield(varargin{1},'compress'))
    compress=varargin{1}.compress;
  end
  meanshift=false;
  if (size(varargin,1) == 1 && isfield(varargin{1},'meanshift'))
    meanshift=varargin{1}.meanshift;
  end
  verbose=false;
  if (size(varargin,1) == 1 && isfield(varargin{1},'verbose'))
    verbose=varargin{1}.verbose;
  end
  trainpredictor=true;
  if (size(varargin,1) == 1 && isfield(varargin{1},'trainpredictor'))
    trainpredictor=varargin{1}.trainpredictor;
  end
  precision='double';
  if (size(varargin,1) == 1 && isfield(varargin{1},'precision'))
    precision=varargin{1}.precision;
  end
end

function [preleft,preright] = parsePreconditioner(Xtic,Ytic,dXX,dYY,cx,cy,varargin)
  if (size(varargin,1) == 0 || ~isfield(varargin{1}, 'pre') || strcmp(varargin{1}.pre, 'diag'))
    preleft=@(z) bsxfun(@rdivide,z,dXX);
    preright=@(z) bsxfun(@rdivide,z,dYY);
  elseif (strcmp(varargin{1}.pre, 'identity'))
    preleft=@(z) z;
    preright=@(z) z;
  else
    preleft=@(z) varargin{1}.pre(z, dXX, 1);
    preright=@(z) varargin{1}.pre(z, dYY, 2);
  end  
end
  
function QY = initialize(Xtic,W,Ytic,XticX,dx,cx,YticY,dy,cy,kp,precision,varargin)
  if (size(varargin,1) == 0 || ~isfield(varargin{1}, 'init') || ...
      strcmp(varargin{1}.init, 'randn'))
    if (exist('OCTAVE_VERSION','builtin'))
      % ugh
      if (strcmp(precision, 'single'))
        QY=single(randn(kp,dy));
      else
        QY=randn(kp,dy);
      end
    else
      QY=randn(kp,dy,precision);
    end
  else
    error('bad initialization spec');
  end
end

function F = XticYimpl(Z,bs,kbs,Xtic,meanX,Ytic,meanY,W,sumw,havemex)
  [~,n]=size(Xtic);
  [k,~]=size(Z);
  F=-sumw*((Z*meanY')*meanX);
  if (bs >= n)
    if havemex && issparse(Ytic) && issparse(Xtic)
      if (kbs >= k)
        F=F+sparsequad(Xtic,W,Ytic,Z);
      else
        for koff=1:kbs:k
          koffend=min(k,koff+kbs-1);
          F(koff:koffend,:)=F(koff:koffend,:)+sparsequad(Xtic,W,Ytic,Z(koff:koffend,:));
        end
      end
    elseif havemex && issparse(Ytic)      
      F=F+dmsm(Z,Ytic,W)*Xtic';
    elseif havemex && issparse(Xtic)
      F=F+dmsm(bsxfun(@times,Z*Ytic,W),Xtic');
    else
      F=F+bsxfun(@times,Z*Ytic,W)*Xtic';
    end
  else
    if havemex && issparse(Ytic) && issparse(Xtic)
      if (kbs >= k)
        F=F+sparsequad(Xtic,W,Ytic,Z,kbs);
      else
        for koff=1:kbs:k
          koffend=min(k,koff+kbs-1);
          F(koff:koffend,:)=F(koff:koffend,:)+sparsequad(Xtic,W,Ytic,Z(koff:koffend,:));
        end
      end       
    elseif havemex && issparse(Ytic)      
      for off=1:bs:n
        offend=min(n,off+bs-1);
        F=F+dmsm(Z,Ytic,W,off,offend)*Xtic(:,off:offend)';
      end
    elseif havemex && issparse(Xtic)
      for off=1:bs:n
        offend=min(n,off+bs-1);
        F=F+dmsm(bsxfun(@times,Z*Ytic(:,off:offend),W(off:offend)),Xtic(:,off:offend)');
      end
    else
      for off=1:bs:n
        offend=min(n,off+bs-1);
        F=F+bsxfun(@times,Z*Ytic(:,off:offend),W(off:offend))*Xtic(:,off:offend)';
      end
    end
  end
end

% objective is \| A Y - B \|^2
function Y = cheesypcg(Afunc,preAfunc,Y,b,iter,verbose)
  if (isa(b,'single'))
    tol=1e-4;
  else
    tol=1e-6;
  end
  
  r=b-Afunc(Y); clear b;
  z=preAfunc(r);
  p=z;
  rho=dot(r,z,2); clear z;
  initsumrr=sum(sum(r.*r));
  minY=Y;
  argminY=initsumrr;

  if (verbose)
    fprintf('\n');
  end

  for ii=1:iter
    Ap=Afunc(p);
    alpha=rho./max(dot(p,Ap,2),eps);
    Y=Y+bsxfun(@times,p,alpha);
    deltar=bsxfun(@times,Ap,alpha); clear Ap;
    r=r-deltar;
    newsumrr=sum(sum(r.*r));  
    
    if (newsumrr < argminY)
      minY=Y;
      argminY=newsumrr;
    end
    
    if (verbose)
      fprintf('iter = %u, newsumrr = %g, initsumrr = %g, relres = %g, argminY = %g\n',ii,newsumrr,initsumrr,newsumrr/initsumrr,argminY);
    end

    if (newsumrr<tol*initsumrr)
        break;
    end

    z=preAfunc(r);
    rho1=-(rho<0).*max(-rho,eps)+(rho>=0).*max(rho,eps);
    rho=-dot(deltar,z,2);
    beta=rho./rho1;
    p=z+bsxfun(@times,p,beta); clear z;
  end
  
  if (newsumrr > argminY)
    Y=minY;
  end
end

function Y = modgs( Y, W )
  if (isfloat(W))
    %W is a symmetric matrix
    C=Y*(W*Y');   
  else
    assert(isa(W,'function_handle'));
    %W is a function that computes Y'*(W*Y)
    C=W(Y);
  end
  [V,D]=eig(0.5*(C+C'));
  Y=pinv(sqrt(max(D,0)))*V'*Y;
end

function P = project(Z,Wx,bx,varargin)
  if (nargin == 4)
    Z=Z(:,varargin{1});
  end
  [~,dx]=size(Z);
  [dlx,~]=size(Wx);
  if (dx ~= dlx)
    error('rembed.project:shapeChk','argument has incompatible shape');
  end
  if (issparse(Z))
    persistent havedmsm;

    if (isempty(havedmsm))
      havedmsm=(exist('dmsm','file') == 3);
    end

    if (havedmsm)
      P=dmsm(Wx',Z')';
    elseif strcmp(class(Wx),'double')
      P=Z*Wx;
    else
      P=single(Z*double(Wx)); % :(
    end
    P=bsxfun(@minus,P,bx);
  else
    P=bsxfun(@minus,Z*Wx,bx);
  end
end
