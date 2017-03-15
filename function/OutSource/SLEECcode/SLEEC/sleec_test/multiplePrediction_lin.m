function [result,predictAcc,predictLabels, tim_test, KNN] = multiplePrediction_lin(data, assign_mat, clusterCenters, SVPModel, SVPMLparams, NNtest, T, numThreads)

if(SVPMLparams.AG == 2)
    XTAG = normalizeMatrix(data.Xt);
else
    XTAG = data.Xt;
end

XTAGtr = XTAG';
tData.Xt = XTAGtr;

numNeighPerClus = NNtest;
[n, d]= size(data.X);
[nt, l] = size(data.Yt);

XXT_t = tData.Xt;
KNN = zeros(nt, T*numNeighPerClus);

tt = tic;
for t = 1:T
   
   assign = assign_mat(t, :);
   numCluster = max(assign)+1;
   cc = clusterCenters{t};
   [tassign] = identifyClusterDP_FtSp_sparse(XXT_t, cc, numCluster);
   
   for clusIter = 0:max(assign)
       dp = find(assign == clusIter);
       dpt=find(tassign == clusIter);
       
       dpl = length(dp);
       
       W = SVPModel{t}.alpha{clusIter+1};
       ztrain = SVPModel{t}.trEmbed{clusIter+1};
       
       ztest = (XXT_t(:, dpt)'*W);
       ztest = full(ztest);
       numNeigh = min(numNeighPerClus, dpl);
       [KNNidx] = findKNN_rf_ed(ztrain', ztest', numNeigh, numThreads);
       %[KNNidx] = findKNN_rf_dp(ztrain', ztest', numNeigh);
       if(numNeighPerClus > numNeigh)
        if((numNeighPerClus-numNeigh) < dpl)
            KNNidx = [KNNidx(1:(numNeighPerClus- numNeigh), :); KNNidx];
        else
            for my_i = 1:numNeighPerClus-numNeigh
                KNNidx = [KNNidx(1, :); KNNidx];
            end
        end
       end
       
       KNN(dpt, (t-1)*numNeighPerClus + 1:t*numNeighPerClus) = dp(KNNidx)';
   end
  
end
tim_test = toc(tt);
[predictAcc, predictLabels] = evalPrec_rf(data.Y', data.Yt', KNN', numNeighPerClus*T, numThreads);
result = sum(predictAcc, 2)/nt;

end