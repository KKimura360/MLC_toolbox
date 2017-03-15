function [bestK, bestT, bestL, kmatrix, errlengthmatrix, bestB, bestX] = mdl4bmf(A, T, varargin)
% MDL4BMF - computes BMF using ASSO and MDL
% 
% Usage
%  k = MDL4BMF(A, T) 
%    returns optimal size k of a BMF of A trying all threshold
%    parameters T
%
%  [k, t] = MDL4BMF(A, T)
%    as above, but returns also the threshold parameter t used
%
%  [k, t, L] = MDL4BMF(A, T)
%    as above, but returns also the minimum description length
%
%  [k, t, L, M] = MDL4BMF(A, T)
%    as above, but returns also the matrix M of all description lengths
%    with different t (rows) and k (columns); in particular, L = M(t, k)
%
%  [k, t, L, M, ME] = MDL4BMF(A, T) 
%    as above, but returns also the matrix ME of all error lengths with
%    different t and k; the structure length is M - ME
%
%  [k, t, L, M, ME, B, X] = MDL4BMF(A, T)
%    as above, but returns also the matrices B and X that yield the shortest
%    description length; does not work with ('errorMeasure', 'all') keyword-
%    value pair (see below)
%
%  k = MDL4BMF(A, T, 'keyword', 'value')
%    assigns value 'value' to keyword 'keyword'. Possible keywords
%    are:
%	KEYWORD			MEANING
%  	'maxK'			Largest k modelselector will try.
%
%	'maxItersSinceBetter'	Maximum number of larger values of k 
%	                modelselector will try after last decrease
%	                in description length.
%                   N.B. Works only with single error mesure.
%
%	'errorMeasure'          Method for computing the error in description
%	                lenght. If set to 'all', uses all error measures
%	                and returns structures k, t, L, M, and ME
%                   containing respective values for all measures.
%                   Possible values are
%       'all'               Run all tests
%       'naiveXor'          Use naive XOR error
%       'typedXor'          Use typed (0->1 & 1->0) errors in XOR
%       'naiveIndex'        Use indices of errors
%       'naiveFactor'       Use naive factorization of errors
%       'DtMnaiveXor'       Use naive XOR with Data-to-Model codes
%       'DtMtypedXor'       Use typed XOR with Data-to-Model codes (default)
%
%    'VerbosityLevel'       Controls the level of verbosity. 0 is no
%                   prints (default); 1 is some, 2 is more, etc.
%
%    'processes'            Number of processes to launch (default = 1)


error(nargchk(2, Inf, nargin));
if mod(size(varargin, 2), 2) ~= 0,
    error('Arguments must be Keyword-Value pairs');
end

% Input parameter checking
if any(A(A ~= 0) ~= 1)
    error('Input matrix A must be 0/1');
end

iA = uint8(A);

if ~isvector(T)
    error('Parameter T must be a scalar or vector')
end

[n, m] = size(A);

% Set default values:
maxK = min(n,m);
maxItersSinceBetter = Inf;
errorMeasure = 'DtMtypedXOR';
verbosity = 0;
processes = 1;

%% Supported keywords
keywordNames = {'maxK', 'maxItersSinceBetter', 'errorMeasure', ...
		'verbosityLevel', 'processes'};

%% Supported error measure names in the order they are computed for all
errorMeasureNames = {'naiveXor', 'typedXor', 'naiveIndex', 'naiveFactor', 'DtMnaiveXor', 'DtMtypedXor'};

for i=1:2:(size(varargin,2)-1),
  switch lower(make_canonical(varargin{i}, keywordNames))
    case {'maxk'}
      maxK = varargin{i+1};
    case {'maxiterssincebetter'}
      maxItersSinceBetter = varargin{i+1};
    case {'errormeasure'}
      errorMeasure = varargin{i+1};
    case {'verbositylevel'}
      verbosity = varargin{i+1};
    case {'processes'}
      processes = varargin{i+1};
    otherwise
      error(['Unknown parameter: ' varargin{i}]);
  end
end

% sanity check
if maxK > min(n,m), 
  warning('MDL4BMF:largeK', 'Setting maximum k to min(n, m) = %i', min(n,m));
  maxK = min(n,m);
end

% Process error measure names
% uses temporarily C and t for local purposes - do not mix with
% later t and C!
C = errorMeasureNames;
C{end+1} = 'all';
t = make_canonical(errorMeasure, C);
if isempty(errorMeasure),
    emns = sprintf('%s ', C{:});
    error(['Error measure type "%s" is either unknown or ambiguous: ' ...
           'possible values are: %s'], errorMeasure, emns);
end
errorMeasure = t;
% remove confusing variables
clear t C

returnBandX = false;

%% process output variables
if nargout > 5,
    if strcmp(errorMeasure, 'all')
        warning('MDL4BMF:returnValues', 'Only returning ''k'', ''t'', ''L'', ''M'', and ''ME'' are currently implemented if using ''all'' as an error measure');
    else
        returnBandX = true;
    end
end

useAllErrorMeasures = false;

% Store space for output variables
% MDL scores shall be saved to a matrix of size |T| x maxK;
% different MDL methods are saved in a struct with similar matrix
% for each different method

if strcmp(errorMeasure, 'all')
    useAllErrorMeasures = true;
    kmatrix = struct();
    for i=1:size(errorMeasureNames, 2),
        kmatrix.(errorMeasureNames{i}) = Inf(length(T), maxK);
        errlengthmatrix.(errorMeasureNames{i}) = Inf(length(T), maxK);
    end
    if isfinite(maxItersSinceBetter)
      warning('MDL4BMF:maxItersSinceBetter', 'Using maxItersSinceBetter is only supported with one error measure');
    end
else
    kmatrix = Inf(length(T), maxK);
    errlengthmatrix = Inf(length(T), maxK);
    bestL = Inf;
    bestK = -1;
    bestT = -1;
    if returnBandX
        bestB = [];
        bestX = [];
    end
end



%% To the actual computation


Corig = A*A'; % Association confidences
Corig = Corig./repmat(diag(Corig)', n, 1); % Normalize

%% DEBUG
%disp('Corig')
%disp(Corig)
%% END DEBUG

% Iterate over all values in T
for titer=1:length(T)
    t = T(titer);
    if (verbosity > 0),
      fprintf(1, 't = %f', t);
    end
    B = nan(n,maxK);
    X = nan(maxK,m);
    available_cols = 1:n;
    C = uint8(Corig >= t);
    %% DEBUG
    %disp(t)
    %disp(C)
    %% END DEBUG
    mask = zeros(n, m, 'uint8');
    for k=1:maxK,
        if (verbosity > 1),
	      fprintf(1, '  k = %i', k);
	    end
        %% select column of C (from available columns) to B and
        % corresponding row to X. avail_id is index to
        % `available_cols', not to C.
	    % If processes > 1, use parallel version
	    if processes > 1,
	      [avail_id, r] = select_best_column_par(iA, ...
                                                 C(:,available_cols), ...
                                                 mask, ...
		                        				 processes);
	    else
          [avail_id, r] = select_best_column(iA, ...
                                             C(:,available_cols), ...
                                             mask);
	    end
        B(:,k) = double(C(:,available_cols(avail_id)));
        X(k,:) = r;
        %% DEBUG
        %disp('B')
        %disp(B)
        %disp('X')
        %disp(X)
        %% END DEBUG
        mask = uint8(bprod(B(:,1:k), X(1:k,:)));
        % Update available_cols;
        foo = [available_cols(1:avail_id-1) available_cols(avail_id+1:end)];
        available_cols = foo;
        %% Compute description lenght
        if useAllErrorMeasures
            [L, LE] = description_length(A, B(:,1:k), X(1:k,:), k, errorMeasureNames);
            %% kmatrix is a struct
            for i=1:length(errorMeasureNames)
                em=errorMeasureNames{i};
                kmatrix.(em)(titer, k) = L.(em);
                errlengthmatrix.(em)(titer, k) = LE.(em);
            end
        else
            %% Only one error measure, kmatrix is a matrix
            [L, LE] = description_length(A, B(:,1:k), X(1:k,:), k, {errorMeasure});
            kmatrix(titer, k) = L.(errorMeasure);
            errlengthmatrix(titer, k) = LE.(errorMeasure);
            %% Keep an eye on the bestL
            if L.(errorMeasure) < bestL,
                bestL = L.(errorMeasure);
                bestK = k;
                bestT = T(titer);
                if returnBandX,
                    bestB = B(:, 1:k);
                    bestX = X(1:k, :);
                end
            end

	        %% check if we haven't got better for a while
	        % inefficient, but simple
	        if isfinite(maxItersSinceBetter) && k > maxItersSinceBetter,
	            if all(kmatrix(titer, k-maxItersSinceBetter:k-1) < L.(errorMeasure)) 
	              %k - find(kmatrix(titer,:) == min(kmatrix(titer,:))) > maxItersSinceBetter, %% removed condition
	              % all previous maxItersSinceBetter results are better than
	              % this one
	              break; %% breaks for k=1:maxK
                end
	        end
        end
    end %% for k=1:maxK
    if (verbosity > 0),
      fprintf(1, '\n');
    end
end %% for t=T

%% Return bestK
if strcmp(errorMeasure, 'all')
    bestK = struct();
    bestT = struct();
    bestL = struct();
    for i=1:size(errorMeasureNames, 2),
        em = errorMeasureNames{i};
        M = kmatrix.(em);
        bestL.(em) = min(M(:));
        [titer,k] = find(M == bestL.(em), 1); %% first column-wise,
                                             %% i.e. the smallest k
        bestK.(em) = k;
        bestT.(em) = T(titer);
    end
%% This part is done within the for t=T loop
%else
%    bestL = min(kmatrix(:));
%    [titer, k] = find(kmatrix == bestL, 1);
%    bestK = k;
%    bestT = T(titer);
end
