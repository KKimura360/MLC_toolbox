function [assign_mat, clusterCenters, tim, normData, KMFTparams] = multipleClustering(data, T, KMFTparams, fname)

[n, d] = size(data.X);
[nt, ~] = size(data.Xt);

if(nargin <= 1)
    T = 1;
end

if(nargin <=2)
    KMFTparams.numIters = 5;
    KMFTparams.numClus = 5;
    KMFTparams.mxPts = 10000000;
    KMFTparams.norm = 1;
    KMFTparams.frac = 1;
    KMFTparams.numThreads = 32;
end

if(nargin <= 3)
    fname = 'x1_cluster_wiki10_';
end
XX = normalizeMatrix(data.X);
XXT = normalizeMatrix(data.Xt);

assign_mat = zeros(T, n);

if(KMFTparams.frac == 1)
    Xin = XX;
    Xtin = XXT;
else
    Xin = data.X;
    Xtin = data.Xt;
end

clusterCenters = {};
cstart = tic;

Xin_tr = Xin';
Xtin_tr = Xtin';
tim = zeros(T,1);
for t = 1:T
    filename = [fname, num2str(t), '.txt']; 
    KMFTparams.outfile = filename;
    tim_t = tic;
    fprintf('Starting Clustering\n');
    runHierKmeansFt;
    tim(t) = toc(tim_t);
    assign_mat(t, :) = assign;
    fprintf('Clustering done\n');
    [centerLb] = test_clusAssignFt_sparse(Xin_tr, Xtin_tr, assign);
    clusterCenters{t} = sparse(centerLb);
end
telapsed = toc(cstart);
normData.X = XX;
normData.Xt = XXT;
end