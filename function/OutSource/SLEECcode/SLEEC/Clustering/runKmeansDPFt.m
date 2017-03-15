function [assign] = runKmeansDPFt(Ytr, numClusters, numIters, numThreads)
rperm = randperm(size(Ytr,2));
selPoints = rperm(1:numClusters);
%keyboard;
%selPoints = randi( [1, size(Y, 1)], numClusters, 1);

clusterCenters = full( Ytr(:, selPoints));

%[assign] = kmeansDPFT_mm(Y', clusterCenters', numClusters, numIters);
[assign] = kmeansDP_FtSp(Ytr, clusterCenters, numClusters, numIters, numThreads);
%keyboard
%hist(assign, numClusters);
end