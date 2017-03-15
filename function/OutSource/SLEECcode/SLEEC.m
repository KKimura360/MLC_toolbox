function [result_out] = SLEEC(data, SLEECparams)

% data : structure containing the complete (obtained using readData.m)
% data.X : sparse n x d matrix containng train features
% data.Xt : sparse nt x d matrix containng test features
% data.Y : sparse n x l matrix containng train labels
% data.Yt : sparse n x l matrix containng test labels

    rng('default')
    cd SLEEC
    
    cd Clustering
    KMFTparams.numIters = 7;
    KMFTparams.numClus = SLEECparams.num_clusters;
    KMFTparams.mxPts = 10000000;
    KMFTparams.norm = 1;
    KMFTparams.frac = 1;
    KMFTparams.numThreads = SLEECparams.num_threads;
    
    [assign_mat, clusterCenters, tim_clus, normData, KMFTparams] = multipleClustering(data, SLEECparams.num_learners, KMFTparams, SLEECparams.fname);
    cd ../sleec_train
    SVPMLparams.AG = SLEECparams.normalize +1;
    SVPMLparams.SVPneigh = SLEECparams.SVP_neigh;
    SVPMLparams.outDim = SLEECparams.out_Dim;
    SVPMLparams.mxitr = 200;
    SVPMLparams.lambda = 1;
    SVPMLparams.w_thresh = SLEECparams.w_thresh;
    SVPMLparams.sp_thresh = SLEECparams.sp_thresh;
    SVPMLparams.c = SLEECparams.cost;
    SVPMLparams.numThreads = SLEECparams.num_threads;
    
    [SVPModel, SVPtime_mat, regressiontime_mat, SVPMLparams] = multipleSVP_lin(data, assign_mat, normData, SVPMLparams, SLEECparams.fname);
    
    cd ../sleec_test
    [result,predictAcc,predictLabels, tim_test, KNN] = multiplePrediction_lin(data, assign_mat, clusterCenters, SVPModel, SVPMLparams, SLEECparams.NNtest, SLEECparams.num_learners, SLEECparams.num_threads);
    
    result_out.assign_mat = assign_mat;
    result_out.clusterCenters = clusterCenters;
    result_out.tim_clus = tim_clus;
    result_out.normData = normData;
    result_out.KMFTparams = KMFTparams;
    result_out.SVPModel = SVPModel;
    result_out.SVPtime_mat = SVPtime_mat;
    result_out.regressiontime_mat = regressiontime_mat;
    result_out.SVPMLparams = SVPMLparams;
    result_out.precision = result;
    result_out.predictAcc = predictAcc;
    result_out.predictLabels = predictLabels;
    result_out.tim_test = tim_test;
    result_out.test_KNN = KNN;
    
    cd ../../

end