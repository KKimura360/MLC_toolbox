function[param]=SetfRAkELParameter(pseudo)
%SetfRAkELParameter
%number of sampled labels
param.numK=3;
% not RAkEL-d
param.type='normal';

%classifier for the 1st layer
param.MLC.name={'BR'};
param.MLC.base.name='ridge';
param.MLC.base.param.lambda=10;
param.MLC.th.type='Scut';
param.MLC.th.param=0.3;

%base classifier
param.base.name='ridge';
param.base.param.lambda=10;

param.th.type='Scut';
param.th.param=0.5;