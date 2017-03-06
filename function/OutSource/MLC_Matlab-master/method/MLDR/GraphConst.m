function[W D] =GraphConst(Y);

%% Dot product weighting.
%% Y is label x data mat

Y(Y>0)=1;
Y(Y<1)=0;

W= Y *Y';
D= diag(sum(W,1));

