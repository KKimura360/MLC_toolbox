function [bestK, bestT, bestErr, E, bestB, bestX] = cv4bmf(A, T, varargin)
% CV4BMF - computes BMF using ASSO and cross validation
% 
% Usage
%  k = CV4BMF(A, T) 
%    returns optimal size k of a BMF of A trying all threshold
%    parameters T.
%  [k, t] = CV4BMF(A, T)
%    as above, but returns also the threshold parameter t used.
%  [k, t, err] = CV4BMF(A, T)
%    as above, but returns also the minimum error.
%  [k, t, err, E] = CV4BMF(A, T)
%    as above, but returns also the matrix M of all errors
%    with different t (rows) and k (columns); in particular, err = E(t, k)
%  k = CV4BMF(A, T, 'keyword', 'value')
%    assigns value 'value' to keyword 'keyword'. Possible keywords
%    are:
%	KEYWORD                 MEANING
%  	'maxK'                  Largest k modelselector will try.
%	'maxItersSinceBetter'	Maximum number of larger values of k 
%                           modelselector will try after last decrease
%                           in error.
%   'folds'                  Perform n-fold CV. Default is 10.
%   'VerbosityLevel'        Controls the level of verbosity. 0 is no
%                           prints (default); 1 is some, 2 is more, etc.
%   'processes'             Number of processes to launch (default = 1)


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
folds = 10;
verbosity = 0;
processes = 1;

%% Supported keywords
keywordNames = {'maxK', 'maxItersSinceBetter', 'folds', ...
		'verbosityLevel', 'processes'};

%% Supported error measure names in the order they are computed for all
errorMeasureNames = {'naiveXor', 'typedXor', 'naiveIndex', 'naiveFactor'};

for i=1:2:(size(varargin,2)-1),
  switch lower(make_canonical(varargin{i}, keywordNames))
    case {'maxk'}
      maxK = varargin{i+1};
    case {'maxiterssincebetter'}
      maxItersSinceBetter = varargin{i+1};
    case {'folds'}
      folds = varargin{i+1};
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
  warning('Setting maximum k to min(n, m) = %i', min(n,m));
  maxK = min(n,m);
end


%% process output variables
if nargout > 4,
    warning('Only returning ''k'', ''t'', ''err'', and ''E'' are currently implemented');
end

% Store space for output variables
% Reconstruction errors shall be saved to a matrix of size |T| x maxK;

E = zeros(length(T), maxK);


%% Compute the folds; leave out some columns
rp = randperm(m);
for i=1:folds-1,
    I{i} = rp(((i-1)*floor(m/folds)+1):i*floor(m/folds));
end
I{folds} = rp(((folds-1)*floor(m/folds)+1):end);


%% To the actual computation



%% DEBUG
%disp('Corig')
%disp(Corig)
%% END DEBUG



for f=1:folds
    if (verbosity > 0), fprintf(1, '\nfold = %i\n', f); end
    ids=true(1,m);
    ids(I{f}) = false;
    Tr=A(:, ids); % Training data
    iTr=iA(:, ids); % Interger training data
    TrM = size(Tr, 2);
    Te=A(:, ~ids); % Testing data
    iTe=iA(:, ~ids); % Interger testing data

    Corig = Tr*Tr'; % Association confidences
    Corig = Corig./repmat(diag(Corig)', n, 1); % Normalize

    % Iterate over all values in T
    for titer=1:length(T)
        t = T(titer);
        if (verbosity > 0),
            fprintf(1, 't = %f', t);
        end
        
        B = nan(n,maxK);
        X = nan(maxK,TrM);
        available_cols = 1:n;
        C = uint8(Corig >= t);
        %% DEBUG
        %disp(t)
        %disp(C)
        %% END DEBUG
        mask = zeros(n, TrM, 'uint8');
        for k=1:maxK,
            if (verbosity > 1),
                fprintf(1, 'k = %i  ', k);
            end
            %% select column of C (from available columns) to B and
            %% corresponding row to X. avail_id is index to
            %% `available_cols', not to C.
            %% If processes > 1, use parallel version
            if processes > 1,
                [avail_id, r] = select_best_column_par(iTr, ...
                                                       C(:,available_cols), ...
                                                       mask, ...
                                                       processes);
            else
                [avail_id, r] = select_best_column(iTr, ...
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
            %% Compute generalized X
            genX = [X(1:k, :) solve_x_greedy_matrix(Te, B(:, 1:k))];
            
            E(titer, k) = E(titer, k) + bnorm([Tr Te], B(:, 1:k), genX);
            %% DEBUG
            %fprintf(2, 'training err = %i\n', bnorm(Tr, B(:, 1:k), X(1:k, :)));
            %fprintf(2, 'o/a err      = %i\n', bnorm([Tr Te], B(:, 1:k), genX));
            %% END DEBUG
            
            %% check if we haven't got better for a while
            %% inefficient, but simple
            % Not DONE Here
            %if isfinite(maxItersSinceBetter) && k > maxItersSinceBetter,
                %if all(kmatrix(titer, k-maxItersSinceBetter:k-1) < L.(errorMeasure)) 
	      	        %% all previous maxItersSinceBetter results are better than
                    %% this one
                    %break; %% breaks for k=1:maxK
                %end
            %end
        end % for k=1:maxK
        if (verbosity > 0),
            fprintf(1, '\n');
        end
    end % for titer=1:length(T)    
end % for f=1:folds

% Normalize E
E = E./folds;

%% Return bestK
bestErr = min(E(:));
[titer, k] = find(E == bestErr, 1);
bestK = k;
bestT = T(titer);

