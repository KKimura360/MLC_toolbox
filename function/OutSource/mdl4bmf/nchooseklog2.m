function x = nchooseklog2(n, k)
%NCHOOSEKLOG2 Computes the base-2 logarithm of binomial coefficient
%   x = NCHOOSEKLOG2(n, k) where n and k are scalars, returns
%   log2(nchoosek(n, k)) avoiding overflows.
%
%   X = NCHOOSEKLOG2(n, K) where K is a vector or a matrix
%   returns X of same shape as K such that X(i) = NCHOOSEKLOG2(n, K(i)).
%
%   See also   NCHOOSEK, GAMMALN

error(nargchk(2, 2, nargin));

if ~isscalar(n)
    error('First input argument must be a scalar');
end

x = (gammaln(n+1) - gammaln(k+1) - gammaln((n+1)-k))./log(2);

end

