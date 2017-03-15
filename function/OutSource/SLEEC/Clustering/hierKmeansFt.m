%Function to run hierarchical kmeans clustering
function [assign, totalClusters, hierstruct] = hierKmeansFt(Ytr, numIters, numClustersInit, mxPtsCluster, totalClusters, fid, frac, numThreads)

[l,n] = size(Ytr);
totalClusters = totalClusters + numClustersInit;

fprintf(fid, 'Number of Datapoints : %d NumClusters : %d \n', n, numClustersInit);

[assign] = runKmeansDPFt(Ytr, numClustersInit, numIters, numThreads);
hierstruct.assign = assign;

c = hist(assign, max(assign+1));

for i = 1:length(c)
    fprintf(fid, '\t\tCluster %d Points %d \n', i, c(i));
end
fprintf(fid, '\n\n');

for i = 1:numClustersInit
   
   clusterPts = find(assign == i-1);
   numPts = length(clusterPts);
   if(numPts <= mxPtsCluster)
       hierstruct.subcluster{i} = struct([]);
       continue
   end
   
   numClusterHier = ceil(frac*numPts/mxPtsCluster);
   [assign2, totalClusters2, hierstructtemp] = hierKmeansFt(Ytr(:, clusterPts), numIters, numClusterHier, mxPtsCluster, totalClusters, fid, frac, numThreads);
   hierstruct.subcluster{i} = hierstructtemp;
   
   assign(clusterPts) = assign2 + max(assign+1);
   totalClusters = totalClusters2;
end
end