function[param]=SetSVPParameter(pseudo)

%BIBTEX
%param.dim=100;
%param.numk=15;
%param.w_thresh=0.1;
%param.sp_thresh=0.01;

param.dim=100; % SLEEC default
param.numk=15;% SLEEC default

param.w_thresh=0.7; %SLEEC default
param.sp_thresh=0.7; %SLEEC default

if exist('function/OutSource/SLEECcode')==0
    error('you need to download SLEECcode from... http://www.manikvarma.org/ and set to funcion/OutSource/SLEECcode');
end    