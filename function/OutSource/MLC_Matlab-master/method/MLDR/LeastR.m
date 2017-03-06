function [x, funVal, ValueL]=LeastR(A, y, z, opts)
%
%%
% Function LeastR
%      Least Squares Loss with the L1-norm Regularization
%
%% Problem
%
%  min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
%
%  By default, rsL2=0.
%  When rsL2 is nonzero, this correspons the well-know elastic net.
%
%% Input parameters:
%
%  A-         Matrix of size m x n
%                A can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  y -        Response vector (of size mx1)
%  z -        L_1 norm regularization parameter (z >=0)
%  opts-      Optimal inputs (default value: opts=[])
%
%% Output parameters:
%  x-         Solution
%  funVal-    Function value during iterations
%
%% Copyright (C) 2009-2010 Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last modified 5 September 2009.
%
% Related functions:
%  sll_opts, initFactor, LeastC
%%

%% Verify and initialize the parameters
%%

% Verify the number of input parameters
if (nargin <3)
    error('\n Inputs: A, y and z should be specified!\n');
elseif (nargin==3)
    opts=[];
end

% Get the size of the matrix A
[m,n]=size(A);

% Verify the length of y
if (length(y) ~=m)
    error('\n Check the length of y!\n');
end

% Verify the value of z
if (z<0)
    error('\n z should be nonnegative!\n');
end

% run sll_opts to set default values (flags)
opts=sll_opts(opts);

%% Detailed initialization
%% Normalization

% Please refer to sll_opts for the definitions of mu, nu and nFlag
%
% If .nFlag =1, the input matrix A is normalized to
%                     A= ( A- repmat(mu, m,1) ) * diag(nu)^{-1}
%
% If .nFlag =2, the input matrix A is normalized to
%                     A= diag(nu)^{-1} * ( A- repmat(mu, m,1) )
%
% Such normalization is done implicitly
%     This implicit normalization is suggested for the sparse matrix
%                                    but not for the dense matrix
%

if (opts.nFlag~=0)
    if (isfield(opts,'mu'))
        mu=opts.mu;
        if(size(mu,2)~=n)
            error('\n Check the input .mu');
        end
    else
        mu=mean(A,1);
    end

    if (opts.nFlag==1)
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=n)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,1)/m).^(0.5); nu=nu';
        end
    else % .nFlag=2
        if (isfield(opts,'nu'))
            nu=opts.nu;
            if(size(nu,1)~=m)
                error('\n Check the input .nu!');
            end
        else
            nu=(sum(A.^2,2)/n).^(0.5);
        end
    end

    ind_zero=find(abs(nu)<= 1e-10);    nu(ind_zero)=1;
    % If some values in nu is typically small, it might be that,
    % the entries in a given row or column in A are all close to zero.
    % For numerical stability, we set the corresponding value to 1.
end

if (~issparse(A)) && (opts.nFlag~=0)
    fprintf('\n -----------------------------------------------------');
    fprintf('\n The data is not sparse or not stored in sparse format');
    fprintf('\n The code still works.');
    fprintf('\n But we suggest you to normalize the data directly,');
    fprintf('\n for achieving better efficiency.');
    fprintf('\n -----------------------------------------------------');
end

%% Starting point initialization

% compute AT y
if (opts.nFlag==0)
    ATy=A'*y;
elseif (opts.nFlag==1)
    ATy=A'*y - sum(y) * mu';  ATy=ATy./nu;
else
    invNu=y./nu;              ATy=A'*invNu-sum(invNu)*mu';
end

% process the regularization parameter

% L2 norm regularization
if isfield(opts,'rsL2')
    rsL2=opts.rsL2;
    if (rsL2<0)
        error('\n opts.rsL2 should be nonnegative!');
    end
else
    rsL2=0;
end

% L1 norm regularization
if (opts.rFlag==0)
    lambda=z;
else % z here is the scaling factor lying in [0,1]
    if (z<0 || z>1)
        error('\n opts.rFlag=1, and z should be in [0,1]');
    end

    lambda_max=max(abs(ATy));
    lambda=z*lambda_max;

    rsL2=rsL2*lambda_max; % the input rsL2 is a ratio of lambda_max
end

% initialize a starting point
if opts.init==2
    x=zeros(n,1);
else
    if isfield(opts,'x0')
        x=opts.x0;
        if (length(x)~=n)
            error('\n Check the input .x0');
        end
    else
        x=ATy;  % if .x0 is not specified, we use ratio*ATy,
        % where ratio is a positive value
    end
end

% compute A x
if (opts.nFlag==0)
    Ax=A* x;
elseif (opts.nFlag==1)
    invNu=x./nu; mu_invNu=mu * invNu;
    Ax=A*invNu -repmat(mu_invNu, m, 1);
else
    Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
end

if (opts.init==0) % If .init=0, we set x=ratio*x by "initFactor"
    % Please refer to the function initFactor for detail

    x_norm=sum(abs(x)); x_2norm=x'*x;
    if x_norm>=1e-6
        ratio=initFactor(x_norm, Ax, y, lambda,'LeastR', rsL2, x_2norm);
        x=ratio*x;    Ax=ratio*Ax;
    end
end

%% The main program

%% The Armijo Goldstein line search scheme + accelearted gradient descent
if (opts.mFlag==0 && opts.lFlag==0)
    
    bFlag=0; % this flag tests whether the gradient step only changes a little

    L=1 + rsL2;
    % We assume that the maximum eigenvalue of A'A is over 1

    % assign xp with x, and Axp with Ax
    xp=x; Axp=Ax; xxp=zeros(n,1);

    % alphap and alpha are used for computing the weight in forming search point
    alphap=0; alpha=1;

    for iterStep=1:opts.maxIter
        % --------------------------- step 1 ---------------------------
        % compute search point s based on xp and x (with beta)
        beta=(alphap-1)/alpha;    s=x + beta* xxp;

        % --------------------------- step 2 ---------------------------
        % line search for L and compute the new approximate solution x

        % compute the gradient (g) at s
        As=Ax + beta* (Ax-Axp);

        % compute AT As
        if (opts.nFlag==0)
            ATAs=A'*As;
        elseif (opts.nFlag==1)
            ATAs=A'*As - sum(As) * mu';  ATAs=ATAs./nu;
        else
            invNu=As./nu;                ATAs=A'*invNu-sum(invNu)*mu';
        end

        % obtain the gradient g
        g=ATAs-ATy + rsL2 * s;

        % copy x and Ax to xp and Axp
        xp=x;    Axp=Ax;

        while (1)
            % let s walk in a step in the antigradient of s to get v
            % and then do the l1-norm regularized projection
            v=s-g/L;

            % L1-norm regularized projection
            x=sign(v).*max(abs(v)-lambda / L,0);

            v=x-s;  % the difference between the new approximate solution x
            % and the search point s

            % compute A x
            if (opts.nFlag==0)
                Ax=A* x;
            elseif (opts.nFlag==1)
                invNu=x./nu; mu_invNu=mu * invNu;
                Ax=A*invNu -repmat(mu_invNu, m, 1);
            else
                Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
            end

            Av=Ax -As;
            r_sum=v'*v; l_sum=Av'*Av;
            
            if (r_sum <=1e-20)
                bFlag=1; % this shows that, the gradient step makes little improvement
                break;
            end

            % the condition is ||Av||_2^2 <= (L - rsL2) * ||v||_2^2
            if(l_sum <= r_sum * (L-rsL2))
                break;
            else
                L=max(2*L, l_sum/r_sum + rsL2);
                % fprintf('\n L=%5.6f',L);
            end
        end

        ValueL(iterStep)=L;
        
        % --------------------------- step 3 ---------------------------
        % update alpha and alphap, and check whether converge
        alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;

        xxp=x-xp;   Axy=Ax-y;
        funVal(iterStep)=Axy'* Axy/2 + rsL2/2 * x'*x + sum(abs(x)) * lambda;

        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol)
                    break;
                end
            case 4
                norm_xp=sqrt(xp'*xp);    norm_xxp=sqrt(xxp'*xxp);
                if ( norm_xxp <=opts.tol * max(norm_xp,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
    end
end

%% Reformulated problem + Nemirovski's scheme

% .mFlag=1, and .lFlag=0
%  refomulate the problem as the constrained convex optimization
%  problem, and then apply Armijo Goldstein line search scheme

% Problem:
%    min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * t' * 1
%    s.t.   |x| <= t

if(opts.mFlag==1 && opts.lFlag==0)    
    
    bFlag=0; % this flag tests whether the gradient step only changes a little
    
    L=1 + rsL2;
    % We assume that the maximum eigenvalue of A'A is over 1

    % assign xp with x, and Axp with Ax
    xp=x; Axp=Ax; xxp=zeros(n,1);
    t=abs(x); tp=t; 
    % t is the upper bound of absolute value of x

    % alphap and alpha are used for computing the weight in forming search point
    alphap=0; alpha=1;

    for iterStep=1:opts.maxIter
        % --------------------------- step 1 ---------------------------
        % compute search point s based on xp and x (with beta)
        beta=(alphap-1)/alpha;    s=x + beta* xxp; s_t= t + beta * (t -tp);

        % --------------------------- step 2 ---------------------------
        % line search for L and compute the new approximate solution x

        % compute the gradient (g) at s
        As=Ax + beta* (Ax-Axp);

        % compute AT As
        if (opts.nFlag==0)
            ATAs=A'*As;
        elseif (opts.nFlag==1)
            ATAs=A'*As - sum(As) * mu';  ATAs=ATAs./nu;
        else
            invNu=As./nu;                ATAs=A'*invNu-sum(invNu)*mu';
        end

        % obtain the gradient g
        g=ATAs-ATy + rsL2 * s;

        % copy x and Ax to xp and Axp
        xp=x;    Axp=Ax;
        tp=t;

        while (1)
            % let s walk in a step in the antigradient of s to get v
            % and then do the l1-norm regularized projection
            
            u=s-g/L;
            v= s_t - lambda / L;
          
            % projection
            [x, t]=ep1R(u, v, n);

            v=x-s;  % the difference between the new approximate solution x
                       % and the search point s
                       
            v_t=t-s_t;

            % compute A x
            if (opts.nFlag==0)
                Ax=A* x;
            elseif (opts.nFlag==1)
                invNu=x./nu; mu_invNu=mu * invNu;
                Ax=A*invNu -repmat(mu_invNu, m, 1);
            else
                Ax=A*x-repmat(mu*x, m, 1);     Ax=Ax./nu;
            end

            Av=Ax -As;
            r_sum=v'*v + v_t'*v_t; l_sum=Av'*Av + v'*v * rsL2;
            
            if (r_sum <=1e-20)
                bFlag=1; % this shows that, the gradient step makes little improvement
                break;
            end
            
            % the condition is ||Av||_2^2 + rsL2 * ||v||_2^2
            %                       <= L * (||v||_2^2 + ||v_t|| _2^2 )
            if(l_sum <= r_sum * L)
                break;
            else
                L=max(2*L, l_sum/r_sum);
                % fprintf('\n L=%5.6f',L);
            end
        end
        
        ValueL(iterStep)=L;

        % --------------------------- step 3 ---------------------------
        % update alpha and alphap, and check whether converge
        alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;

        xxp=x-xp;   Axy=Ax-y;
        funVal(iterStep)=Axy'* Axy/2 + rsL2/2 * x'*x + sum(t) * lambda;

        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp=sqrt(xxp'*xxp + norm(t-tp)^2);
                if ( norm_xxp <=opts.tol)
                    break;
                end
            case 4
                norm_xp=sqrt(xp'*xp + tp'*tp);    norm_xxp=sqrt(xxp'*xxp+ norm(t-tp)^2);
                if ( norm_xxp <=opts.tol * max(norm_xp,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
    end
    
end


%% adaptive line search

% .mFlag=1, and .lFlag=1
%  refomulate the problem as the constrained convex optimization
%  problem, and then apply adaptive line search scheme

% Problem:
%    min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * t' * 1
%    s.t.   |x| <= t

if(opts.mFlag==1 && opts.lFlag==1)
    
    bFlag=0; % this flag tests whether the gradient step only changes a little

    L=1 + rsL2;
    % We assume that the maximum eigenvalue of A'A is over 1
    
    gamma=1;
    % we shall set the value of gamma = L,
    % where L is appropriate for the starting point

    xp=x; Axp=Ax;
    % store x and Ax
    xxp=zeros(n,1);
    % the difference of x and xp    
    t=abs(x); tp=t;
    % t is the upper bound of absolute value of x
    
    % compute AT Ax
    if (opts.nFlag==0)
        ATAx=A'*Ax;
    elseif (opts.nFlag==1)
        ATAx=A'*Ax - sum(Ax) * mu';  ATAx=ATAx./nu;
    else
        invNu=Ax./nu;                ATAx=A'*invNu-sum(invNu)*mu';
    end
    
    % We begin the adaptive line search in the following
    %
    % Note that, in the line search, L and beta are changing
    
    for iterStep=1:opts.maxIter

        ATAxp=ATAx;
        % store ATAx to ATAxp

        if (iterStep~=1)
            % compute AT Ax
            if (opts.nFlag==0)
                ATAx=A'*Ax;
            elseif (opts.nFlag==1)
                ATAx=A'*Ax - sum(Ax) * mu';  ATAx=ATAx./nu;
            else
                invNu=Ax./nu;                ATAx=A'*invNu-sum(invNu)*mu';
            end
        end

        %--------- Line Search for L begins
        while (1)
            if (iterStep~=1)
                alpha= (-gamma+ sqrt(gamma*gamma + 4* L * gamma)) / (2*L);
                beta= (gamma - gamma* alphap) / (alphap * gamma + alphap* L * alpha);
                % beta is the coefficient for generating search point s

                s=x + beta* xxp;   s_t= t + beta * (t -tp);
                As=Ax + beta* (Ax-Axp);
                ATAs=ATAx + beta * (ATAx- ATAxp);
                % compute the search point s, A * s, and A' * A * s
            else
                alpha= (-1+ sqrt(5)) / 2;
                beta=0; s=x; s_t=t; As=Ax; ATAs=ATAx;
            end

            g=ATAs-ATy + rsL2 * s;
            % compute the gradient g
           
            % let s walk in a step in the antigradient of s 
            u=s-g/L;
            v= s_t - lambda / L;

            % projection
            [xnew, tnew]=ep1R(u,v,n);

            v=xnew-s;  % the difference between the new approximate solution x
                            % and the search point s
            v_t=tnew-s_t;
            
            % compute A xnew
            if (opts.nFlag==0)
                Axnew=A* xnew;
            elseif (opts.nFlag==1)
                invNu=xnew./nu; mu_invNu=mu * invNu;
                Axnew=A*invNu -repmat(mu_invNu, m, 1);
            else
                Axnew=A*xnew-repmat(mu*xnew, m, 1);     Axnew=Axnew./nu;
            end

            Av=Axnew -As;
            r_sum=v'*v + v_t'*v_t; l_sum=Av'*Av + v'*v * rsL2;
            
            if (r_sum <=1e-20)
                bFlag=1; % this shows that, the gradient step makes little improvement
                break;
            end
            
            % the condition is ||Av||_2^2 + rsL2 * ||v||_2^2
            %                       <= L * (||v||_2^2 + ||v_t|| _2^2 )
            if(l_sum <= r_sum * L)
                break;
            else
                L=max(2*L, l_sum/r_sum);
                % fprintf('\n L=%5.6f',L);
            end
        end
        %--------- Line Search for L ends

        gamma=L* alpha* alpha;    alphap=alpha;
        % update gamma, and alphap
        
        ValueL(iterStep)=L;

        tao=L * r_sum / l_sum;
        if (tao >=5)
            L=L*0.8;
        end
        % decrease the value of L

        xp=x;    x=xnew; xxp=x-xp;
        Axp=Ax;  Ax=Axnew;
        % update x and Ax with xnew and Axnew        
        tp=t; t=tnew;
        % update tp and t       
        
        Axy=Ax-y;
        funVal(iterStep)=Axy' * Axy/2 + rsL2/2 * x'*x + lambda * sum(t);
        % compute function value

        switch(opts.tFlag)
            case 0
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <= opts.tol)
                        break;
                    end
                end
            case 1
                if iterStep>=2
                    if (abs( funVal(iterStep) - funVal(iterStep-1) ) <=...
                            opts.tol* funVal(iterStep-1))
                        break;
                    end
                end
            case 2
                if ( funVal(iterStep)<= opts.tol)
                    break;
                end
            case 3
                norm_xxp=sqrt(xxp'*xxp+ norm(t-tp)^2);
                if ( norm_xxp <=opts.tol)
                    break;
                end
            case 4
                norm_xp=sqrt(xp'*xp + tp'*tp);    norm_xxp=sqrt(xxp'*xxp+ norm(t-tp)^2);
                if ( norm_xxp <=opts.tol * max(norm_xp,1))
                    break;
                end
            case 5
                if iterStep>=opts.maxIter
                    break;
                end
        end
    end
end


%%
if(opts.mFlag==0 && opts.lFlag==1)
    error('\n The function does not support opts.mFlag=0 & opts.lFlag=1!');
end


%% Auxiliary Functions

function opts = sll_opts(opts)

% Options for Sparse Learning Library
%
% Notice:
% If one or several (even all) fields are empty, sll_opts shall assign the
% default settings.
%
% If some fields of opts have been defined, sll_opts shall check the fields
% for possible errors.
%
%
% Table of Options.  * * indicates default value.
%
%% FIELD            DESCRIPTION
%% Starting point
%
% .x0               Starting point of x. 
%                   Initialized according to .init.
%
% .c0               Starting point for the intercept c (for Logistic Loss)
%                   Initialized according to .init.
%
% .init             .init specifies how to initialize x.  
%                       * 0 => .x0 is set by the function initFactor *
%                         1 => .x0 and .c0 are defined
%                         2 => .x0= zeros(n,1), .c0=0
%
%% Termination
%
% .maxIter          Maximum number of iterations.
%                       *1e4*
%
% .tol              Tolerance parameter.
%                       *1e-4*
%
% .tFlag            Flag for termination.
%                       * 0 => abs( funVal(i)- funVal(i-1) ) <= .tol *
%                         1 => abs( funVal(i)- funVal(i-1) ) 
%                              <= .tol max( funVal(i-1), 1)
%                         2 => funVal(i) <= .tol
%                         3 => norm( x_i - x_{i-1}, 2) <= .tol
%                         4 => norm( x_i - x_{i-1}, 2) <= 
%                              <= .tol max( norm( x_{i-1}, 2), 1 )
%                         5 => Run the code for .maxIter iterations
%
%% Normalization
%
% .nFlag            Flag for implicit normalization of A.
%                       * 0 => Do not normalize A *
%                         1 => A=(A-repmat(mu, m, 1))*diag(nu)^{-1}
%                         2 => A=diag(nu)^{-1}*(A-repmat(mu,m,1)
%
% .mu               Row vector to be substracted from each sample.
%                           (.mu is used when .nFlag=1 or 2)
%                       If .mu is not specified, then
%                            * .mu=mean(A,1) *
%
% .nu               Weight (column) vector for normalization
%                           (.mu is used when .nFlag=1 or 2)
%                       If .nu is not specified, then
%                       * .nFlag=1 => .nu=(sum(A.^2, 1)'/m.^{0.5} *
%                       * .nFlag=2 => .nu=(sum(A.^2, 2)/n.^{0.5} *
%
%% Regularization
%
% .rFlag            Flag for regularization
%                           (.rFlag is used for the functions with "R")
%                        * 0 => lambda is the regularization parameter *
%                          1 => lambda = lambda * lambda_{max}
%                               where lambda_{max} is the maximum lambda
%                               that yields the zero solution
% .rsL2              Regularization parameter value of the squared L2 norm
%                           (.rsL2 is used only for l1 regularization)
%                        *.rsL2=0*
%                    If .rFlag=0, .rsL2 is used without scaling
%                       .rFlag=1, .rsL2=.rsL2 * lambda_{max}
%
%% Method & Line Search
% .lFlag
%
%% Grooup & Others
%
% .ind              Indices for k groups (a k+1 row vector)
%                   For group lasso only
%                   Indices for the i-th group are (ind(i)+1):ind(i+1)
%
% .q                Value of q in L1/Lq regularization
%                      *.q=2*
%
% .sWeight          The sample (positive and negative) weight
%                   For the Logistic Loss only
%                   Positive sample: .sWeight(1)
%                   Negative sample: sWeight(2)
%                   *1/m for both positive and negative samples*
%
% .gWeight          The weight for different groups
%                      *.gWeight=1*
%
% .fName            The name of the function
%
%% Copyright (C) 2009-2010 Jun Liu, and Jieping Ye
%
% You are suggested to first read the Manual.
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last modified 7 August 2009.

%% Starting point

if isfield(opts,'init')
    if (opts.init~=0) && (opts.init~=1) && (opts.init~=2)
        opts.init=0; % if .init is not 0, 1, or 2, then use the default 0
    end
    
    if ~isfield(opts,'x0') && (opts.init==1)
        opts.init=0; % if .x0 is not defined and .init=1, set .init=0
    end
else
    opts.init = 0; 
                     % if .init is not specified, use "0"
end

%% Termination

if isfield(opts,'maxIter')
    if (opts.maxIter<1)
        opts.maxIter=10000;
    end
else
    opts.maxIter=10000;
end

if ~isfield(opts,'tol')
    opts.tol=1e-3;
end

if isfield(opts,'tFlag')
    if opts.tFlag<0
        opts.tFlag=0;
    elseif opts.tFlag>5
        opts.tFlag=5;
    else
        opts.tFlag=floor(opts.tFlag);
    end
else
    opts.tFlag=0;
end

%% Normalization

if isfield(opts,'nFlag')
    if (opts.nFlag~=1) && (opts.nFlag~=2)
        opts.nFlag=0;
    end
else
    opts.nFlag=0;
end

%% Regularization

if isfield(opts,'rFlag')
    if (opts.rFlag~=1)
        opts.rFlag=0;
    end
else
    opts.rFlag=0;
end
%% Method (Line Search)

if isfield(opts,'lFlag')
    if (opts.lFlag~=1)
        opts.lFlag=0;
    end
else
    opts.lFlag=0;
end

if isfield(opts,'mFlag')
    if (opts.mFlag~=1)
        opts.mFlag=0;
    end
else
    opts.mFlag=0;
end



%% Auxiliary Function 2
function ratio=initFactor(x_norm, Ax , y, z, funName, rsL2, x_2norm)
% 
%% function initFactor
%     compute the an optimal constant factor for the initialization
%
%
% Input parameters:
% x_norm-      the norm of the starting point
% Ax-          A*x, with x being the initialization point
% y-           the response matrix
% z-           the regularization parameter or the ball
% funName-     the name of the function
%
% Output parameter:
% ratio-       the computed optimal initialization point is ratio*x
%
%% Copyright (C) 2009-2010 Jun Liu, and Jieping Ye
%
% For any problem, please contact with Jun Liu via j.liu@asu.edu
%
% Last revised on August 2, 2009.

switch(funName)
    case 'LeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'LeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        %fprintf('\n ratio=%e',ratio);
        
    case 'glLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'mcLeastR'
        ratio=  (Ax(:)'*y(:) - z * x_norm) / norm(Ax,'fro')^2;
        %fprintf('\n ratio=%e',ratio);
        
    case 'mtLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax);
        %fprintf('\n ratio=%e',ratio);
        
    case 'nnLeastR'
        ratio=  (Ax'*y - z * x_norm) / (Ax'*Ax + rsL2 * x_2norm);
        ratio=max(0,ratio);
        
    case 'nnLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax'*y / (Ax'*Ax + rsL2 * x_2norm);

        if ratio_optimal<0
            ratio=0;
        elseif ratio_optimal<=ratio_max
            ratio  =  ratio_optimal;
        else
            ratio  =  ratio_max;
        end
        % fprintf('\n ratio=%e,%e,%e',ratio,ratio_optimal,ratio_max);
        
    case 'mcLeastC'
        ratio_max     = z / x_norm;
        ratio_optimal = Ax(:)'*y(:) / (norm(Ax'*Ax,'fro')^2);
        
        if abs(ratio_optimal)<=ratio_max
            ratio  =  ratio_optimal;
        elseif ratio_optimal<0
            ratio  =  -ratio_max;
        else
            ratio  =  ratio_max;
        end
        
    otherwise
        fprintf('\n The specified funName is not supprted');
end
