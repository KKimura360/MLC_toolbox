function [U,V]= NMF(X,dim,iter)

%Initialization
U=sprand(size(X,1),dim,1);
V=sprand(size(X,2),dim,1);

for i = 1:iter
    U=updateU(X,U,V);
    V=updateU(X',V,U);
    [U,V]=normalize_factor(U,V);
end
