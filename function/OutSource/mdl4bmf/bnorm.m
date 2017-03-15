function e = bnorm(A, B, C);
% BNORM - Returns the "Boolean norm".
% e = BNORM(A, B, C) returns sum(sum(abs(A - min(1,B*C)))).
  
  e = sum(sum(abs(A - min(1,B*C))));