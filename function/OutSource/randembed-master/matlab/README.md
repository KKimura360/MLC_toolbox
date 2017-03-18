matlab
==========
This directory contains the matlab/octave code.

Files:

 * [rembed.m](rembed.m): randomized embedding code.  Given features X and labels Y, computes the top right singular vectors of (U<sub>X</sub><sup>T</sup> Y), where U<sub>X</sub> are the left singular vectors of X.  
 * [calmultimls.m](calmultimls.m): (primal approximation to) kernel multiclass and multilabel fitting routine.  uses preconditioned SGD.

To see examples of these two routines composed to solve problems, look at the multilabel experiments in [../mulan](../mulan) and the multiclass experiment in [../aloi](../aloi).

compiling the mex
-----------------
If you are using matlab, you should compile the mex to make everything go faster.  Hopefully you can just type
> make NUM\_THREADS=6
 
and all the mex will be compiled for you.  Adjust NUM\_THREADS based upon how much parallelism is appropriate for your setup.  (Sorry, it's not cool enough to auto-detect this).

If you lack a reasonable shell environment, you can execute the mex commands directly from matlab, e.g., under Windows:

> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack sparsequad.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack dmsm.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack sparseweightedsum.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack chofactor.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack chosolve.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack fastexpprob.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
> &gt;&gt; mex OPTIMFLAGS="&#47;O2" -largeArrayDims -DNUM\_THREADS=2 -lmwblas -lmwlapack fastsoftmax.cpp  
> Building with 'Microsoft Visual C++ 2012'.  
> MEX completed successfully.  
