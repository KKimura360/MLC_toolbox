% T is the number of random clusterings
% assign_mat, tassign_mat, XX, XXT are available
% For each clustering, save
function [SVPModel, SVPtime_mat, regressiontime_mat, SVPMLparams] = multipleSVP_lin(data, assign_mat, normData, SVPMLparams, fname)
addpath(genpath('prcpy'));
XX=normData.X;
XXT=normData.Xt;
T = size(assign_mat, 1);

if(nargin <= 4)
    SVPMLparams.AG = 2;
    SVPMLparams.SVPneigh = 50;
    SVPMLparams.outDim = 75;
    SVPMLparams.mxitr = 200;
    SVPMLparams.lambda = 1;
    SVPMLparams.w_thresh = 0.75;
    SVPMLparams.sp_thresh = 0.5;
    SVPMLparams.c = 0.1;
    SVPMLparams.numThreads = 32;
end

if(nargin < 5)
    fname = 'x1_train_wikiLshtc_';
end

SVPModel = {};
SVPtime_mat = zeros(T, 1);
regressiontime_mat = zeros(T, 1);

f_id = fopen('CumulResults.txt', 'w');
for iterCount = 1:T
    fprintf('Clustering Number %d starting...\n', iterCount);
    SVPMLparams.outfile = [fname, num2str(iterCount), '.txt'];
    assign = assign_mat(iterCount, :);
    
    clusterSVP_asym;
    SVPModel{iterCount}.trEmbed = svpEmbedCluster;
    SVPModel{iterCount}.alpha = polyKernelAlpha;
    
    SVPtime_mat(iterCount) = svpTime;
    regressiontime_mat(iterCount) = regressionTime;
end
fclose(f_id);
end