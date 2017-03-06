function [nDCG] = evalnDCG(scoreMat, lblMat, k)
%scoreMat and lblMat are nt x L

nDCG = zeros(k, 1);
sumY = sum(lblMat, 2);
Ypred = zeros(size(lblMat, 1), k);
normFac = zeros(size(lblMat, 1), k);
for i = 1:k
    [~, Jidx] = max(scoreMat, [], 2);
    Iidx = (1:length(Jidx))';
    linIdx = sub2ind(size(lblMat), Iidx, Jidx);
    lbls = lblMat(linIdx);
    scoreMat(linIdx) = 0;
    Ypred(:, i) = lbls/log2(1+i);
    sY = sum(Ypred(:, 1:i), 2);
    
    normFac(:, i) = ((sumY >= i)+ eps)/log2(1+i);
    sF = sum(normFac(:, 1:i), 2);
    
    
    nDCG(i) = sum(sY./sF)/size(lblMat, 1);
end
end