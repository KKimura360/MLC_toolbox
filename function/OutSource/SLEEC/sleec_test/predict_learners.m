numLearners = 15;
predictTimeMat = zeros(numLearners, 1);
precMat = zeros(numLearners, 5);
modelSizeMat = zeros(numLearners, 1);
nDCGMat = zeros(numLearners, 5);


outfile = 'wiki10TestLog';

fid = fopen(outfile, 'w');

prec = zeros(5, 1);
fprintf(fid, 'numTree\ttest_time\tp1\tp2\tp3\tp4\tp5\tnDCG1\tnDCG2\tnDCG3\tnDCG4\tnDCG5\n');
%numneigh = [1, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6];
numneigh = [35, 30, 30, 30, 25, 25, 25, 25, 20, 20, 15, 15, 15, 10, 10, 10];
for numTree = 1:numLearners
    fprintf('TESTING : Learner %d\n', numTree);
    
    [result,predictAcc,predictLabels, tim_test, KNN] = multiplePrediction_lin(data, tData, assign_mat, clusterCenters, SVPModel, numneigh(numTree), numTree);
   cd Tools\
   [scoreMat] = getScoreMat(data.Y', KNN', 50);
   nDCG = evalnDCG(scoreMat, data.Yt, 5);
   for k = 1:5
      prec(k) = get_metrics_at_k(scoreMat', data.Yt', k, 1); 
   end
   cd ..
   
   predictTimeMat(numTree, 1) = tim_test;
   precMat(numTree, :) = prec';
   nDCGMat(numTree, :) = nDCG';
   
   fprintf(fid, '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', numTree, tim_test, prec(1), prec(2), prec(3), prec(4), prec(5), nDCG(1), nDCG(2), nDCG(3), nDCG(4), nDCG(5));
end

fclose(fid);