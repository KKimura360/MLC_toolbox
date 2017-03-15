function [scoreMat] = getScoreMat(Y, NN, maxLbl)
%Y: lxn
%NN: kXnt
%maxLbl: estimate average label on per instance (the space allocated per test point to form the label vector will be k*maxLbl)
[I, J, S] = formScoreMat(Y, NN, maxLbl);
nxi = nnz(I);
I = I(1:nxi);
J = J(1:nxi);
S = S(1:nxi);
scoreMat = sparse(I, J, S, size(NN, 2), size(Y, 1));
end