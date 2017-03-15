%Function to identify clusters given data and clusters
function [clusterCenter] = test_clusAssignFt_sparse(Xtr, Xt_tr, clusAssign, clusCenInp)
    
    [d, ~] = size(Xt_tr);
    numCluster = max(clusAssign + 1);
    if(nargin == 4)
        clusterCenter = clusCenInp;
    else
        tic;
        clusterCenter = sparse([], [], [], d, numCluster, numCluster*100000);
        for i = 1:numCluster
            dp = find(clusAssign == i-1);
            numdp = length(dp);
            cluCen = (sum(Xtr(:, dp), 2))/numdp;
            clusterCenter(:, i) = cluCen;
        end
        toc;
    end
end