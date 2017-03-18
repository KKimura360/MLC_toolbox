%% Compile MEX functions
clear all;

%% liblinear
cd function/base/linearSVM/liblinear-2.1/matlab/
make
cd ../../../../../

%% libsvm 
cd function/base/SVM/libsvm-3.21/matlab/
make
cd ../../../../../

%% MIToolbox
cd function/OutSource/MIToolbox-3.0.0/matlab/
CompileMIToolbox
cd ../../../../

%% SLEEC
try
    cd function/OutSource/SLEECcode/
    make_SLEEC
    cd ../../../
catch
	warning('you need to download SLEEC code and set function/OutSource/SLEECcode, see README');
end

%% FastXMLs
try
    cd function/OutSource/FastXML_PfastreXML/Tools/matlab
    make
    cd ../../../../../
catch 
    warning('you need to download FastXML_PfastreXML and set function/OutSource/, see README');
end

%% BMaD (asso)
cd function/OutSource/mdl4bmf/
try
	makeasso
catch
	warning('failed to compile asso functions')
end
cd ../../../

%% Rembrandit  % gave up to read the source codes 
% cd function/OutSource/randembed-master/matlab
% if ispc
%     try
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 chofactor.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 chosolve.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 dmsm.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 fastexpprob.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 fastsoftmax.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 sparsequad.cpp -lmwblas -lmwlapack
%     mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=1 sparseweightedsum.cpp -lmwblas -lmwlapack
%     catch
%         warning('rembrandit failed')
%     end
% elseif isunix
%    !make
% else
%     error('rembrendit does not support MAC hehehe')
% end
%Add newmethods
