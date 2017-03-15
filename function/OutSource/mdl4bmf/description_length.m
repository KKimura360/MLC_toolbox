function [L, LE] = description_length(D, A, B, k, errorMeasureNames)
% DESCRIPTION_LENGTH - computes description length for BMF
% 
% Usage:
% L = DESCRIPTION_LENGTH(D, B, X, k, errorMeasureNames)
%   Computes the description length of a BMF of matrix D as BoX
%   (with size k) using error measure defined in
%   errorMeasureNames. Returns a struct with field names
%   corresponding to the strings in cell array
%   errorMeasureNames. Possible error measures are
%		'naiveXor'		Use naive XOR error
%		'typedXor'		Use typed (0->1 & 1->0) errors in XOR
%		'naiveIndex'	Use indices of errors
%		'naiveFactor'	Use naive factorization of errors
%       'DtMnaiveXor'   Use naive XOR with Data-to-Model code
%       'DtMtypedXor'   Use typed XOR with Data-to-Model code 
%
% [L, LE] = DESCRIPTION_LENGTH(D, B, X, k, errorMeasureNames)
%   Also returns the error part of the description length; the structure
%   part is L - LE.
%					    

  % Process input arguments

  error(nargchk(5, 5, nargin));

  [n, m] = size(D);

  %% initialize output variables L and LE
  L = struct();
  LE = struct();

  possibleErrorMeasureNames = {'naiveXor', 'typedXor', 'naiveIndex', 'naiveFactor', 'DtMnaiveXor', 'DtMtypedXor'};

  %% D = A o B XOR E

  % These are constant over any error measure
  
  % the approx. L^0 cost of n and m
  %L_n = log2(n)+log2(log2(n));
  %L_m = log2(m)+log2(log2(m));
  
  % The Elias Delta of n and m
  %L_n = floor(log2(n)) + 2*floor(log2(log2(n)) + 1) + 1
  %L_m = floor(log2(m)) + 2*floor(log2(log2(m)) + 1) + 1
  
  % The L^0 cost of n and m
  L_n = L0(n);
  L_m = L0(m);
  
  L_k = log2(min(m,n));

  if k > 0
    E = double(xor(D, bprod(A, B))); 
  elseif k == 0
    E = D;
  else 
    error('k cannot be negative');
  end
    
  % Select correct error measure, only canonized names are accepted
  for i=1:length(errorMeasureNames)
      if all(all(E == 0))
          hasErr = false;
          L_E = 0;
      else
          hasErr = true;
      end
      em = errorMeasureNames{i};
      switch lower(em)
        case {'naivexor'}
          % Naive XOR method
          [L_A, L_B] = naiveModel(A, B, n, m, k);
          if hasErr
            L_E = naiveXor(E);
          end
        case {'typedxor'}
          % Typed XOR method
          [L_A, L_B] = naiveModel(A, B, n, m, k);
          if hasErr
            L_E = typedXor(E, A, B, k);
          end
        case {'naiveindex'}
          % Naive error index encoding method
          [L_A, L_B] = naiveModel(A, B, n, m, k);
          if hasErr
            L_E = naiveIndex(E);
          end
        case {'naivefactor'}
          [L_A, L_B] = naiveModel(A, B, n, m, k);
          if hasErr
            L_E = naiveFactor(E);
          end
        case {'dtmnaivexor'}
          % Naive XOR with Data-to-Model code
          [L_A, L_B] = DtMModel(A, B, n, m, k);
          if hasErr
            L_E = DtMnaiveXor(E);
          end
        case {'dtmtypedxor'}
          % Typed XOR with Data-to-Model code
          [L_A, L_B] = DtMModel(A, B, n, m, k);
          if hasErr
            L_E = DtMtypedXor(E, A, B, k);
          end
        otherwise
          emns = sprintf('%s ', possibleErrorMeasureNames{:});  
          error('Error measure "%s" is not implemented or ambiguous; possible values are: %s',... 
                em, emns);
      end

      % Total description lenght
      L.(em) = L_A + L_B + L_E + L_k + L_n + L_m;
      LE.(em) = L_E;
  end %% iter over errorMeasureNames
  return
%% end of the function!

%% L^0 cost function
function l = L0(n)
  l = 0;
  if n == 0
  	return;
  end
  tmp = log2(n);
  while tmp > 0
  	l = l + tmp;
 		tmp = log2(tmp);
  end
  l = l + log2(2.865064);

%% Model cost functions
function [L_A, L_B] = naiveModel(A, B, n, m, k)
    if k == 0
        L_A = 0;
        L_B = 0;
        return
    end
    sumA = sum(A);
    sumB = sum(B,2)';
    L_A1 = -log2(sumA/n);
    L_A0 = -log2((n-sumA)/n);
    % If some column is full of 0s or 1s, Inf appear; correct for that
    L_A1(L_A1 == Inf) = 0;
    L_A0(L_A0 == Inf) = 0;
    L_A = sum(sumA.*L_A1 + (n - sumA).*L_A0) + k*log2(n); % L(D | H_A) + L(H_A)

    L_B1 = -log2(sumB/m);
    L_B0 = -log2((m-sumB)/m);
    L_B1(L_B1 == Inf) = 0;
    L_B0(L_B0 == Inf) = 0;
    L_B = sum(sumB.*L_B1 + (m - sumB).*L_B0) + k*log2(m); 

function [L_A, L_B] = DtMModel(A, B, n, m, k)
    if k == 0
        L_A = 0;
        L_B = 0;
        return;
    end
    L_A = k*log2(n) + sum(nchooseklog2(n, sum(A)));
    L_B = k*log2(m) + sum(nchooseklog2(m, sum(B,2)'));
        
%% error mode functions; for easier reference
function L_E = naiveXor(E)
  [n,m] = size(E);
  Esum = sum(E(:));
  L_E1 = -log2(Esum/(n*m));
  L_E0 = -log2((n*m - Esum)/(n*m));
  L_E = Esum*L_E1 + (n*m - Esum)*L_E0;
  L_EH = log2(n*m); %% not Esum, you stupid Dutch bastard!
  L_EH(L_EH == Inf) = 0;
  L_E = L_E + L_EH;


function L_E = typedXor(E, A, B, k)
  [n, m] = size(E);
  if k == 0, 
    C = zeros(n,m);
  else
    C = bprod(A,B);
  end
  Csum = sum(C(:));
  %% 0s that should be 1s
  EpSum = sum(double(E(:) & ~C(:)));
  L_Ep_1 = -log2(EpSum / (n*m - Csum));
  L_Ep_0 = -log2(1 - EpSum / (n*m - Csum));
  %% their contribution
  if EpSum > 0,
      overcoverContrib = EpSum*L_Ep_1 + (n*m - EpSum - Csum)* ...
          L_Ep_0;
  else
      overcoverContrib = 0; % Nothing's overcovered
  end
  %% DEBUG
  %fprintf(2, 'EpSum = %f, L_Ep_1 = %f, L_Ep_0 = %f, overcoverContrib = %f\n',...
  %        EpSum, L_Ep_1, L_Ep_0, overcoverContrib);
  %% 1s that should be 0s
  EmSum = sum(double(E(:) & C(:)));
  L_Em_1 = -log2(EmSum / Csum);
  L_Em_0 = -log2(1 - EmSum / Csum);
  %% their contribution
  if EmSum > 0,
      undercoverContrib = EmSum*L_Em_1 + (Csum - EmSum)*L_Em_0;
  else
      undercoverContrib = 0; % everything's covered
  end
  %% DEBUG
  %fprintf(2, 'EmSum = %f, L_Em_1 = %f, L_Em_0 = %f, undercoverContrib = %f\n', ...
  %        EmSum, L_Em_1, L_Em_0, undercoverContrib);

  %% Contribution of Csum
  if Csum == 0,
      CsumContrib = 0;
  elseif Csum < n*m,
      %% CsumContrib = log2(EpSum) + log2(log2(EpSum)) + log2(EmSum) + log2(log2(EmSum));
      CsumContrib = log2(Csum) + log2(n*m - Csum);
  else
      %% CsumContrib = log2(EmSum) + log2(log2(EmSum)); %% this is correct, but gives NaNs.
      CsumContrib = log2(Csum);
  end

  %% Total
  L_E = overcoverContrib ...  % EpSum*L_Ep_1 + (n*m - EpSum - Csum)*L_Ep_0 ...
        + undercoverContrib ...  % + EmSum*L_Em_1 + (Csum - EmSum)*L_Em_0 ...
        + CsumContrib; % + log2(Csum) + log2(n*m - Csum);

function L_E = naiveIndex(E)
  [n, m] = size(E);
  L_E = sum(E(:))*log2(n*m);

function L_E = naiveFactor(E)
  [n, m] = size(E);
  L_Ec = log2(m) - (m-1)*log2((m-1)/m);
  L_Er = log2(n) - (n-1)*log2((n-1)/n);
  L_E = sum(E(:))*(L_Ec + L_Er);

function L_E = DtMnaiveFactor(E)
  [n, m] = size(E);
  L_Ec = log2(m);
  L_Er = log2(n);
  L_E = sum(E(:))*(L_Ec + L_Er);


function L_E = DtMnaiveXor(E)
  L_E = log2(numel(E)) + nchooseklog2(numel(E), sum(E(:)));
  
function L_E = DtMtypedXor(E, A, B, k)
  [n, m] = size(E);
  if k == 0, 
    C = zeros(n,m);
  else
    C = bprod(A,B);
  end
  Csum = sum(C(:));
  undercover = nchooseklog2(n*m-Csum, sum(double(E(:) & ~C(:))));
  overcover = nchooseklog2(Csum, sum(double(E(:) & C(:))));
  if Csum == 0 || Csum == n*m
      CsumContrib = log2(n*m);
  else
      CsumContrib = log2(Csum) + log2(n*m - Csum);
  end
  %Csum = sum(C(:));
  L_E = undercover + overcover + CsumContrib;