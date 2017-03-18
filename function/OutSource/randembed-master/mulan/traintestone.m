function res=traintestone(what)
  addpath('../matlab/');
  prefix='./';

  randn('seed',8675309);
  rand('seed',90210);
  
  start=tic;

  load(what);

  % octave loads ys/yt as structs of ir,jc,and data ... (?)

  if (isa(ys,'struct'))
    n=length(ys.jc)-1;
    ii=repelems(1:n,[1:n; diff(ys.jc)]);
    ys=sparse(ys.ir+1,ii,ys.data);
    clear n ii;
  end

  if (isa(yt,'struct'))
    n=length(yt.jc)-1;
    ii=repelems(1:n,[1:n; diff(yt.jc)]);
    yt=sparse(yt.ir+1,ii,yt.data);
    clear n ii;
  end

  trainright=xt;
  trainleft=yt;
  testright=xs;
  testleft=ys;

  % train-test split

  [n,d]=size(trainright);
  [m,~]=size(testright);
  
  v=sort(eig(trainleft'*trainleft),'descend');
  vs=cumsum(v);
  vs=vs/vs(end);
  embedd=find(vs>0.9, 1);

  % add constant feature
  trainright=horzcat(trainright,ones(n,1));
  testright=horzcat(testright,ones(m,1));
    
  ultraw=ones(1,size(trainleft,1));
  
  tic
  rs=rembed(trainright',ultraw,trainleft',embedd, ...
            struct('pre','identity','innerloop',100,'lambda',1e-4,...
                   'tmax',2,'trainpredictor',true,'verbose',false));
  toc

  trainh=horzcat(rs.projectx(trainright),ones(n,1));
  testh=horzcat(rs.projectx(testright),ones(m,1));
  
  perm=randperm(n);
  nsubtrain=ceil(0.9*n);
  subtrainh=trainh(perm(1:nsubtrain),:);
  subtrainleft=trainleft(perm(1:nsubtrain),:);
  valtrainh=trainh(perm(nsubtrain+1:end),:);
  valtrainleft=trainleft(perm(nsubtrain+1:end),:);
  subppos=full(sum(subtrainleft,1))/size(subtrainleft,1);
  
  maxiter=300;
  allf=1000+randi(7000,1,maxiter);
  allrfac=0.25+3.5*rand(1,maxiter);
  alllambda=exp(log(1e-4)+(log(1e-1)-log(1e-4))*rand(1,maxiter));
  alllogisticiter=1+randi(14,1,maxiter);
  alleta=0.25+1.0*rand(1,maxiter);
  allalpha=0.05+0.85*rand(1,maxiter);
  alldecay=0.9+0.15*rand(1,maxiter);
  kerns={'g','m3','m5','qg','qm3','qm5'};
  allkerns=randi(6,1,maxiter);
  maxshrink=1/sqrt(nsubtrain);
  minshrink=1/nsubtrain;
  allshrink=(1-maxshrink)+(maxshrink-minshrink)*rand(1,maxiter);

  for iter=1:maxiter
    f=allf(iter);
    rfac=allrfac(iter);
    lambda=alllambda(iter);
    logisticiter=alllogisticiter(iter);
    eta=alleta(iter);
    alpha=allalpha(iter);
    decay=alldecay(iter);
    kern=kerns{allkerns(iter)};
    shrink=allshrink(iter);
    
    clear wr b ww;
    try
      [wr,b,ww]=calmultimls(subtrainh,subtrainleft,...
                            struct('lambda',lambda,'f',f,'fbs',6000,...
                                   'rfac',rfac,'kernel',kern,...
                                   'logisticiter',logisticiter,'eta',eta,'alpha',alpha,'decay',decay,...
                                   'shrink',shrink));
    catch
      fprintf('*');
      continue
    end
    
    thres=(logisticiter==0)*0.5;
    fprintf('.');
    [~,~,~,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,subppos,valtrainh,valtrainleft,false);
    [teste,testelb,testeub,microF1,microF1lb,microF1ub]=multiHammingBoot(thres,wr,b,ww,valtrainh,valtrainleft,false);
        
    if (~exist('bestham','var'))
      bestham=struct('iter',iter,'loss',[teste,testelb,testeub]);
      fprintf('\niter = %u, bestham.loss = %g',iter,bestham.loss(1));
      bestmicro=struct('iter',iter,'loss',[microF1,microF1lb,microF1ub]);
      fprintf('\niter = %u, bestmicro.loss = %g',iter,bestmicro.loss(1));
      bestmacro=struct('iter',iter,'loss',[macroF1,macroF1lb,macroF1ub]);
      fprintf('\niter = %u, bestmacro.loss = %g',iter,bestmacro.loss(1));
    else
      if (teste < bestham.loss(1))
        bestham=struct('iter',iter,'loss',[teste,testelb,testeub]);
        fprintf('\niter = %u, bestham.loss = %g',iter,bestham.loss(1));
      end
      if (microF1 > bestmicro.loss(1))
        bestmicro=struct('iter',iter,'loss',[microF1,microF1lb,microF1ub]);
        fprintf('\niter = %u, bestmicro.loss = %g',iter,bestmicro.loss(1));
      end
      if (macroF1 > bestmacro.loss(1))
        bestmacro=struct('iter',iter,'loss',[macroF1,macroF1lb,macroF1ub]);
        fprintf('\niter = %u, bestmacro.loss = %g',iter,bestmacro.loss(1));
      end         
    end
  end
  
  fprintf('\n');
  which=1;
  ppos=full(sum(trainleft,1))/size(trainleft,1);
  for iter=[bestham.iter bestmicro.iter bestmacro.iter]
    f=allf(iter);
    rfac=allrfac(iter);
    lambda=alllambda(iter);
    logisticiter=alllogisticiter(iter);
    eta=alleta(iter);
    alpha=allalpha(iter);
    decay=alldecay(iter);
    kern=kerns{allkerns(iter)};
    shrink=allshrink(iter);
  
    tic
    clear wr b ww;
    [wr,b,ww]=calmultimls(trainh,trainleft,...
                          struct('lambda',lambda,'f',f,'fbs',6000,...
                                 'rfac',rfac,'kernel',kern,...
                                 'logisticiter',logisticiter,'eta',eta,'alpha',alpha,'decay',decay,...
                                 'shrink',shrink));
    thres=(logisticiter==0)*0.5;
    fprintf('iter=%u f=%g rfac=%g lambda=%g logisticiter=%u eta=%g alpha=%g decay=%g kern=%s shrink=%g\n',...
            iter,f,rfac,lambda,logisticiter,eta,alpha,decay,kern,shrink);
    if (which == 1 || which == 2)
      [teste,testelb,testeub,microF1,microF1lb,microF1ub]=multiHammingBoot(thres,wr,b,ww,testh,testleft,true);
      if (which == 1)
        res.calmls_embed_test_hamming=[testelb,teste,testeub];
      else
        res.calmls_embed_test_microF1=[microF1lb,microF1,microF1ub];
      end
    else
      [~,~,~,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,ppos,testh,testleft,true);
      res.calmls_embed_test_macroF1=[macroF1lb,macroF1,macroF1ub];
    end
    which=which+1;
  end

%  % calibration plot is for macrof1 model
%
%  figure
%  [trainbuckets,trainphatzero,trainphatone]=phat(wr,b,ww,trainh,trainleft,20);
%  plot(0.5*(trainbuckets(2:end)+trainbuckets(1:end-1)),(trainphatone./(trainphatone+trainphatzero))')
%  hold on
%  plot(logspace(-2,0,10),logspace(-2,0,10))
%  hold off
%
%  figure
%  [testbuckets,testphatzero,testphatone]=phat(wr,b,ww,testh,testleft,20);
%  plot(0.5*(testbuckets(2:end)+testbuckets(1:end-1)),(testphatone./(testphatone+testphatzero))')
%  hold on
%  plot(logspace(-2,0,10),logspace(-2,0,10))
%  hold off
  
  toc(start)
end

function [yhati,yhatj]=blockmultiinfer(X,th,wr,b,ww)
  [fplus1,~]=size(ww);
  f=fplus1-1;
  Z=cos(bsxfun(@plus,X*wr,b))*ww(1:f,:);
  Z=bsxfun(@plus,Z,ww(fplus1,:));
  [yhati,yhatj]=find(Z>th);
end

function [buckets,pyhatzero,pyhatone]=phat(wr,b,ww,trainh,trainy,numbuckets)
  persistent havemex;

  if (isempty(havemex))
    havemex = (exist('fastexpprob','file') == 3);
  end

  [~,f]=size(wr);
  [~,k]=size(ww);
  [n,~]=size(trainh);
  buckets=linspace(0,1,numbuckets);
  pyhatzero=zeros(numbuckets,1);
  pyhatone=zeros(numbuckets,1);
  
  bs=min(n,ceil(1e+9/(f*k)));
  for off=1:bs:n
      offend=min(n,off+bs-1);
      Z=cos(bsxfun(@plus,trainh(off:offend,:)*wr,b))*ww(1:f,:);
      Z=bsxfun(@plus,Z,ww(f+1,:));
      Z=min(Z,40); 
      if (havemex)
        fastexpprob(Z);
      else
        Z=exp(Z); Z=Z./(1+Z);
      end
      [i,j,~]=find(trainy(off:offend,:));
      oneidx=sub2ind(size(Z),i,j);
      allidx=1:numel(Z);
      zeroidx=setdiff(allidx,oneidx);
      for ii=1:numbuckets
          pyhatone(ii)=pyhatone(ii)+sum(Z(oneidx)<buckets(ii));
          pyhatzero(ii)=pyhatzero(ii)+sum(Z(zeroidx)<buckets(ii));
      end
  end
  pyhatone=diff(pyhatone);
  pyhatzero=diff(pyhatzero);
end

function [loss,truepos,falsepos,falseneg]=multiHammingImpl(th,wr,b,ww,X,Y,imp)
  [~,f]=size(wr);
  [n,k]=size(Y);
  [~,s]=size(imp);

  truepos=zeros(1,s);
  falsepos=zeros(1,s);
  falseneg=zeros(1,s);
  bs=min(n,ceil(1e+9/max(k,f)));
  for off=1:bs:n
      offend=min(n,off+bs-1);
      mybs=offend-off+1;
      [yhati,yhatj]=blockmultiinfer(X(off:offend,:),th,wr,b,ww);
      [i,j,~]=find(Y(off:offend,:));
      
      for ss=1:s
        weights=imp(i+(off-1),ss);
        megay=sparse(i,j,weights,mybs,k); clear weights;
        weightshat=imp(yhati+(off-1),ss);
        megayhat=sparse(yhati,yhatj,weightshat,mybs,k); clear weightshat;
        bstruepos=sum(megay(sub2ind(size(megay),yhati,yhatj)));
        bsfalsepos=sum(sum(megayhat))-bstruepos;
        bsfalseneg=sum(sum(megay))-bstruepos;
        truepos(ss)=truepos(ss)+bstruepos;
        falsepos(ss)=falsepos(ss)+bsfalsepos;
        falseneg(ss)=falseneg(ss)+bsfalseneg;
        clear megay megayhat;
      end
  end
  
  total=k*sum(imp,1);
  loss=(falsepos+falseneg)./total;
end

function [microF1,macroF1]=F1metric(truepos,falsepos,falseneg)
  classprec=truepos./(truepos+falsepos+1e-12);
  classrec=truepos./(truepos+falseneg+1e-12);
  % http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
  macroF1=mean(2*(classprec.*classrec)./(classprec+classrec+1e-12),1);
  % http://www.kaggle.com/c/lshtc/details/evaluation
  % MaP=mean(classprec,1);
  % MaR=mean(classrec,1);
  % macroF1=2*MaP*MaR./(MaP+MaR+1e-12);
  
  prec=sum(truepos,1)./(sum(truepos,1)+sum(falsepos,1)+1e-12);
  rec=sum(truepos,1)./(sum(truepos,1)+sum(falseneg,1)+1e-12);
  microF1=2*(prec.*rec)./(prec+rec+1e-12);
end

function [loss,cilb,ciub,microloss,microlb,microub]=multiHammingBoot(th,wr,b,ww,testh,testy,doprint)
  [m,~]=size(testh);
  imp=poissrnd(1,m,16);
  [testloss,testtruepos,testfalsepos,testfalseneg]=...
      multiHammingImpl(th,wr,b,ww,testh,testy,imp);
  
  [~,ind]=sort(testloss);
  loss=mean(testloss(ind(8:9)));
  cilb=testloss(ind(2));
  ciub=testloss(ind(15));
    
  [microF1,~]=F1metric(testtruepos,testfalsepos,testfalseneg);
  [~,ind]=sort(microF1);
  microloss=mean(microF1(ind(8:9)));
  microlb=microF1(ind(2));
  microub=microF1(ind(15));
 
  if (doprint)
    fprintf('per-ex inference: (hamming) [%g,%g,%g] (micro) [%g,%g,%g]\n',...
            cilb,loss,ciub,microlb,microloss,microub);
  end
end

function [microF1,macroF1]=multiF1Impl(wr,b,ww,trainh,trainy,ppos,imp)
  persistent havemex;

  if (isempty(havemex))
    havemex = (exist('fastexpprob','file') == 3);
  end

  [~,f]=size(wr);
  [n,k]=size(trainy);
  kbs=min(k,ceil(1e+9/n));
  [~,s]=size(imp);
  
  truepos=zeros(k,s);
  falsepos=zeros(k,s);
  falseneg=zeros(k,s);
  for off=1:kbs:k
    offend=min(k,off+kbs-1);
    Z=cos(bsxfun(@plus,trainh*wr,b))*ww(1:f,off:offend);
    Z=bsxfun(@plus,Z,ww(f+1,off:offend));
    Z=min(Z,40); 
    if (havemex)
      fastexpprob(Z);
    else
      Z=exp(Z); Z=Z./(1+Z);
    end
    for jj=off:offend
      [sp,ind]=sort(Z(:,jj-off+1),'descend');
      r=(n*ppos(jj)+1:n*ppos(jj)+n)';
      fscores=cumsum(sp)./r;
      [~,am]=max(fscores);
      i=find(trainy(:,jj));
      yhati=ind(1:am);

      bstruepos=sum(imp(intersect(i,yhati),:),1);
      bsfalsepos=sum(imp(setdiff(yhati,i),:),1);
      bsfalseneg=sum(imp(setdiff(i,yhati),:),1);
      truepos(jj,:)=truepos(jj,:)+bstruepos;
      falsepos(jj,:)=falsepos(jj,:)+bsfalsepos;
      falseneg(jj,:)=falseneg(jj,:)+bsfalseneg;
    end
  end  
  
  [microF1,macroF1]=F1metric(truepos,falsepos,falseneg);
end

function [microF1,microF1lb,microF1ub,macroF1,macroF1lb,macroF1ub]=multiF1Boot(wr,b,ww,ppos,testh,testy,doprint)
  [m,~]=size(testh);
  imp=poissrnd(1,m,16);
  [testmicroF1,testmacroF1]=multiF1Impl(wr,b,ww,testh,testy,ppos,imp);
  
  [~,ind]=sort(testmicroF1);
  microF1=mean(testmicroF1(ind(8:9)));
  microF1lb=testmicroF1(ind(2));
  microF1ub=testmicroF1(ind(15));
  
  [~,ind]=sort(testmacroF1);
  macroF1=mean(testmacroF1(ind(8:9)));
  macroF1lb=testmacroF1(ind(2));
  macroF1ub=testmacroF1(ind(15));

  if (doprint)
    fprintf('per-class inference: (micro) [%g,%g,%g] (macro) [%g,%g,%g]\n',...
            microF1lb,microF1,microF1ub,...
            macroF1lb,macroF1,macroF1ub);
  end
end
