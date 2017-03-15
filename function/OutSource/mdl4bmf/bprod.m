function A = bprod(B, C)
% BPROD - a Boolean product of two matrices
%
% Usage: A = BPROD(B, C) returns A = B o C. Works also for 
% non-Boolean B and C.

error(nargchk(2, 2, nargin));

if size(B,2) ~= size(C, 1)
    error(['First matrix must have same number of columns as the ' ...
           'second has rows.']);
end

B(B~=0) = 1;
C(C~=0) = 1;
A = min(1, B*C);
