function [Yt_pred, HL] = LSpaceTrans(DataSet, M, alg)
  if (isa(M, 'char'))
    M = str2num(M);
  end
  lambda = 0.1;

  %read dataset
  [Y, X, Yt, Xt] = read_dataset(DataSet);
  [N, K] = size(Y);
  [Nt, d] = size(Xt);

  % {0,1} -> {-1,1}
  if sum(sum(Y>0)) == sum(sum(Y))
    Y = 2*Y - 1;
    Yt = 2*Yt - 1;
  end

  %encoding scheme
  %for Binary Relevance with Random Discarding
  if (strcmp(alg, 'br'))
    [Z, Vm] = br_encode(Y, M);
  %for Principal Label Space Transformation
  elseif (strcmp(alg, 'plst'))
    [Z, Vm, shift] = plst_encode(Y, M);
  %for Conditional Principal Label Space Transformation
  elseif (strcmp(alg, 'cplst'))
    [Z, Vm, shift] = cplst_encode(Y, M, X, lambda);
  %for FaIE
  elseif (strcmp(alg, 'faie'))
    [Z, Vm] = FaIE_encode(Y, M, X, lambda);
  %for cssp
  elseif (strcmp(alg, 'cssp'))
    [Z, Vm] = cssp_encode(Y, M);
  else
    fprintf(1, 'ERROR, unrecognized coding scheme');
    return;
  end
  
  %ridge regression
  ww = ridgereg(Z, X, lambda);
  Zt_pred = [ones(Nt, 1) Xt] * ww;

  %decoding scheme
  %for Binary Relevance with Random Discarding
  if (strcmp(alg, 'br'))
    [Yt_pred, ~] = round_linear_decode(Zt_pred, Vm);
  %for Principal Label Space Transformation
  elseif (strcmp(alg, 'plst'))
    [Yt_pred, ~] = round_linear_decode(Zt_pred, Vm, shift);
  %for Conditional Principal Label Space Transformation
  elseif (strcmp(alg, 'cplst'))
    [Yt_pred, ~] = round_linear_decode(Zt_pred, Vm, shift);
  %for FaIE
  elseif (strcmp(alg, 'faie'))
    [Yt_pred, ~] = round_linear_decode(Zt_pred, Vm);
  elseif (strcmp(alg, 'cssp'))
    [Yt_pred, ~] = round_linear_decode(Zt_pred, Vm);
  else
    fprintf(1, 'ERROR, unrecognized coding scheme');
    return;
  end
  [~,~,~,HL,~] = evaluate(Yt_pred, Yt);
