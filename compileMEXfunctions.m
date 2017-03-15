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
cd function/OutSource/SLEEC/
make
cd ../../../

%% BMaD (asso)
cd function/OutSource/mdl4bmf/
makeasso
cd ../../../
%Add newmethods
