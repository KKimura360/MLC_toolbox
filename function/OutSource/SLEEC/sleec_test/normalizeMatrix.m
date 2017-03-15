%Normalize matrix by making each row unit norm
%Z is in sparse format, so normalizing by taking transpose

function [ZZ] = normalizeMatrix(Z)
    Zt = Z';
    normZ = sqrt(sum((Zt.^2), 1));
    ZZ =  bsxfun(@rdivide, Zt, normZ);
    ZZ = ZZ';
end