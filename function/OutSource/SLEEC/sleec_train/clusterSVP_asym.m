w_thresh = SVPMLparams.w_thresh;
spParam = SVPMLparams.sp_thresh;

Yt = data.Yt; Y = data.Y;

if(SVPMLparams.AG == 2)
    XAG = XX;
    XTAG = XXT;
else
    XAG = data.X;
    XTAG = data.Xt;
end

normY = sqrt(sum((Y'.^2), 1)) + 1e-10;
Y = bsxfun(@rdivide, Y', normY);
Y = Y';
%Y = data.Y;
[n, l] = size(Y);
[nt, ~] = size(Yt);

%Set the parameters for svp
outDim = SVPMLparams.outDim;
params.tol=1e-3;
params.mxitr=SVPMLparams.mxitr;
params.verbosity=1;

%storing precisions (5xnumClusters) and numer of datapoints(numClustersx1)
numClusterAssigned = max(assign) + 1;
dpClusters = zeros(numClusterAssigned, 1);
embedTr = zeros(size(data.X,1), outDim);
svpEmbedCluster = {};
polyKernelAlpha = {};
svpTime = 0;
regressionTime = 0;

lambda = SVPMLparams.lambda;
fid = fopen(SVPMLparams.outfile, 'w');

Ytr = Y';
XAGtr = XAG';
dytr = data.Y';
XTAGtr = XTAG';

numThreads = SVPMLparams.numThreads;
%Form the svp embedding for each cluster obtained
for clusterIter = 0:max(assign)
    %Create separate dataset
    tClus = tic;
    tic;
    dp = find(assign == clusterIter);
    ds = Ytr(:, dp)';
    dsY = dytr(:, dp)';
    nc = length(dp);
    dpClusters(clusterIter+1) = nc;
    dsx = XAGtr(:, dp)';
    t = toc;
    svpTime = svpTime + t;
    
    tic;
    numNeighbors = SVPMLparams.SVPneigh;
    outDim = SVPMLparams.outDim;
    numNeighbors = min(numNeighbors, nc);
    outDim  = min(outDim, nc);
    
    fprintf('\n Cluster : %d NumTrainPoints : %d outDim : %d\n', clusterIter, nc, outDim);
    fprintf(fid, '\n Cluster : %d NumTrainPoints : %d\n', clusterIter, nc);
    
    [Om, OmVal, neighborIdx] = findKNN_test(ds', numNeighbors, numThreads);
    
    %Setup svp for this dataset
    neighborIdx = neighborIdx';
    done = false;
    
    [I,J]=ind2sub([nc nc],Om(:));
    MOmega=sparse(I, J,OmVal(:), nc, nc);
    t = toc;
    svpTime = svpTime + t;
    while(~done)
        try
            tic;
            [U, S, V]=lansvd(MOmega,outDim, 'L');
            Uinit=U*sqrt(S);
            Vinit=V*sqrt(S);
            [U, V]=WAltMin_asymm(Om(:), OmVal(:), params.mxitr, params.tol, Uinit, Vinit, numThreads);
            t = toc;
            done  = true;
        catch exception
            msgString = getReport(exception);
            disp(msgString);
            done = false;
        end
    end
    svpTime = svpTime + t;
    
    tic;
    Zc = U;
    Vc = V;
   
    [W, alpha, mu] = computeW(dsx, Zc, 0.001, 0.01, 2, SVPMLparams.c);
    [W_I, W_J, W_lin] = find(W);
    W_sort = sort(abs(W_lin));
    w_idx = ceil(w_thresh*length(W_sort));
    if(w_idx==0)
        w_idx = 1;
    end
    W_lin(abs(W_lin)<W_sort(w_idx)) = 0;
    W = sparse(W_I, W_J, W_lin, size(XX, 2), outDim);
    
    polyKernelAlpha{clusterIter + 1} = W;
    Zct = full(dsx*W);
    sp_thresh_v = zeros(outDim, 1);

    for sp_i= 1:outDim
       [a, a_idx] = sort(abs(Zct(:, sp_i)));
       sp_thresh = a_idx(1:ceil(nc*spParam));
       Zct(sp_thresh, sp_i) = 0;
       sp_thresh_v(sp_i) = a(sp_thresh(end));
    end

    t = toc;
    regressionTime = regressionTime + t;
    svpEmbedCluster{clusterIter +1} = Zct;
    
    toc(tClus);
end
numNeighbors = SVPMLparams.SVPneigh;
outDim = SVPMLparams.outDim;
fclose(fid);