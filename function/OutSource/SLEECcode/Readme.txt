**The code  requires prior installation of libinear
  The windows 64 bit binary of 'train' is already included in the sleec_train folder
  To run on other platforms, compile liblinear and use the matlab train executable for linux in sleec_train
  For parallelization, we have used openmp library, just make sure that the compiler you are using
  is OpenMP compatible**
  
To compile SLEEC, run the following command on the Matlab prompt
> make_SLEEC
SLEEC training and prediction code have been combined with the SLEEC functions. To run SLEEC, try
> load data.mat
> SLEECparams
> [result] = SLEEC(data, params);


The following explains the matlab structure for the data, parameters and the output result

SLEEC Data

	data    : structure containing the complete dataset
	data.X  : sparse n x d matrix containing train features
	data.Xt : sparse m x d matrix containing test features
	data.Y  : sparse n x L matrix containing train labels
	data.Yt : sparse m x L matrix containing test labels
SLEEC Parameters

	params.num_learners : number of SLEEC learners in ensemble (default 10)
	params.num_clusters : Initial number of clusters (default 300)
	params.num_threads  : Number of threads for parallelization (default 32)
	params.SVP_neigh    : Number of nearest neighbours to be preserved (default 15)
	params.out_Dim      : embedding dimensions (default 50)
	params.w_thres      : 1-w_thresh is the sparsity of regressors w (default 0.7)
	params.sp_thresh    : 1-sp_thresh is the sparsity of embeddings (default 0.7)
	params.cost         : liblinear cost coefficient (default 0.1)
	params.NNtest       : number of nearest neighbours to consider while testing (default 10)
	params.normalize    : 1 for normalized data, 2 for unnormalized data (only for mediamill)
	params.fname        : filename for logging purposes
SLEEC Result

	result.clusterCenters     : cluster centers for the different learners
	result.tim_clus           : time taken for clustering
	result.SVPModel           : model for different learners containing embeddings and regressors
	result.SVPtime_mat        : time taken for performing SVP for each learner
	result.regressiontime_mat : time taken for learning regressors
	result.precision          : overall precision accuracy
	result.predictAcc         : precision accuracy per test point
	result.predictLabels      : Top-k labels predicted per test point
	result.tim_test           : time taken for the testing procedure
	result.test_KNN           : kNN Matrix for the test points
Toy Example

The zip file containing the source code also include the BibTeX dataset as a toy example. The following are the instructions to run SLEEC on the BibTeX dataset

> cd Toy_Example 
> load bibtex.mat 
> bibtexParams 
> cd .. 
> [result] = SLEEC(data, params); 

To verify that you have run the code successfully, please compare the precision that you obtain with the result structure that has been provided in the Toy_Example folder.