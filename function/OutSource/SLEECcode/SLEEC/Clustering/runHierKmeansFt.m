% KMFTparams are the parameters
% KMFTparams.numIters;
% KMFTparams.numClus;
% KMFTparams.mxPts;
% KMFTparams.norm;
% KMFTparams.frac;
% KMFTparams.outfile;

numIters = KMFTparams.numIters;
numClustersInit = KMFTparams.numClus;
mxPtsCluster = KMFTparams.mxPts;
totalClusters = 0;
frac = KMFTparams.frac;
numThreads = KMFTparams.numThreads;
fid = fopen(KMFTparams.outfile, 'w');

[assign1, cc, hierstruct] = hierKmeansFt(Xin_tr, numIters, numClustersInit, mxPtsCluster, totalClusters, fid, frac, numThreads);
assign = zeros(size(Xin, 1), 1);

clusterCount = 0;
for i = 0:max(assign1)
    clsidx = find(assign1 == i);
    if(isempty(clsidx))
        continue
    end
    assign(clsidx) = clusterCount;
    clusterCount = clusterCount+1;
end

fclose(fid);