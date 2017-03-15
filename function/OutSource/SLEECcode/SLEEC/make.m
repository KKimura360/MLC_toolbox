cd Clustering
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims kmeansDP_FtSp.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"

cd ../sleec_train
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims findKNN_test.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims updateU.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims updateV.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims compute_X_Omega.c smat.cpp COMPFLAGS="/openmp $COMPFLAGS" 

cd ../sleec_test
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims evalPrec_rf.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims findKNN_rf_dp.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims findKNN_rf_ed.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims identifyClusterDP_FtSp_sparse.cpp smat.cpp COMPFLAGS="/openmp $COMPFLAGS" 

cd ..