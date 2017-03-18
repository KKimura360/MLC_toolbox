function w=megamls(xt,yt,varargin)
    [n,k]=size(yt);
    [~,d]=size(xt);

    if (exist('dmsm','file') == 3 && ...
        exist('chofactor','file') == 3 && ...
        exist('chosolve','file') == 3 && ...
        exist('fastsoftmax','file') == 3)
      havemex=true;
    else
      havemex=false;
      if (~exist('OCTAVE_VERSION','builtin'))
        warning('rembed:nomex', 'MEX acceleration not available, have you compiled the mex?');
      end
    end
    
    [lambda,logisticiter,monfunc,decay,eta,alpha,preflate]=parseArgs(k,varargin{:});

    C=xt'*xt;
    tr=sum(diag(C));
    C(sub2ind(size(C),1:size(C,1),1:size(C,1)))=...
      C(sub2ind(size(C),1:size(C,1),1:size(C,1)))+(lambda*tr/d);
    if (havemex)
      chofactor(C);
    else
      C=chol(C);
    end
    C=single(C);

    if (issparse(yt))
      if (havemex)
        xticy=single(dmsm(xt',yt));
      else
        xticy=single(double(xt)'*yt); % :(
      end
    else
      xticy=single(xt'*yt);
    end
    
    w=xticy+0;
    if (preflate)
      classcounts=full(mean(yt,1));
      maxp=max(classcounts);
      if (havemex)
        w=dmsm(w,spdiags(sqrt(10000*maxp)./sqrt(maxp+9999*classcounts'),0,k,k));
      else
        w=single(double(w)*spdiags(sqrt(10000*maxp)./sqrt(maxp+9999*classcounts'),0,k,k));
      end
      clear classcounts maxp;
    end
    if (havemex)
      chosolve(C,w);
    else
      w=C\(C'\w);
    end
    
    monfunc(w);
    eta=(1-alpha)*eta;
    if (alpha > 0)
      dwold=zeros(size(w,1),size(w,2),'single');
    end
    for i=1:logisticiter
      bs=min(ceil(n/10),ceil(1e+9/k));
      perm=randperm(n);
      for off=1:bs:n
        offend=min(n,off+bs-1);
        xntmptic=xt(perm(off:offend),:)';
        pttic=w'*xntmptic;
        if (havemex)
          fastsoftmax(pttic,max(pttic)); pptic=pttic; clear pttic; % probabilistic predictions
        else
          pttic=bsxfun(@minus,pttic,max(pttic));
          pttic=exp(pttic);
          pttic=bsxfun(@rdivide,pttic,sum(pttic));
          pptic=pttic;
          clear pttic;
        end
        g=-xntmptic*pptic'; clear xntmptic pptic;     % gradient
        g=g+(bs/n)*xticy;
        if (havemex)
          chosolve(C,g);
        else
          g=C\(C'\g);
        end
        g=(eta*n/bs)*g;
        if (alpha > 0)
          dwold=alpha*dwold;
          dwold=dwold+g;
          w=w+dwold;
        else
          w=w+g;
        end
        clear g;
      end
      monfunc(w);
      eta=eta*decay;
    end
end

function [lambda,logisticiter,monfunc,decay,eta,alpha,preflate]=parseArgs(k,varargin)
  lambda=1.0;
  if (isfield(varargin{1},'lambda'))
      lambda=varargin{1}.lambda;
  end
  logisticiter=0;
  if (isfield(varargin{1},'logisticiter'))
      logisticiter=varargin{1}.logisticiter;
  end
  monfunc=@(wr,b,ww) 1;
  if (isfield(varargin{1},'monfunc'))
      monfunc=varargin{1}.monfunc;
  end
  decay=1.0;
  if (isfield(varargin{1},'decay'))
      decay=varargin{1}.decay;
  end
  eta=1.0;
  if (isfield(varargin{1},'eta'))
      eta=varargin{1}.eta;
  end
  alpha=0.0;
  if (isfield(varargin{1},'alpha'))
      alpha=varargin{1}.alpha;
  end
  preflate=(logisticiter>0);
  if (isfield(varargin{1},'preflate'))
      preflate=varargin{1}.preflate;
  end  
end
