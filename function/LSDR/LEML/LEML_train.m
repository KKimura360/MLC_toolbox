function[model,time]=LEML_train(X,Y,method)
%% Input
%X: Feature Matrix (NxF)
%Y: Label   Matrix (NxL)
%param.type controls its loss function
%0:Squared Loss
%1:Logistic Loss
%2: Squared Hinge Loss
%10: Sqaured Loss with fully obeservation, This is the fastest.  (see the paper);
%method.param{x}.lambda: a weight parameter
%method.param{x}.dim   : lower-dim of labels
%% Output
%model: A learned model (cell(dim+2,1))
%% Reference (APA style from google scholar)

%%% Method
% error check

%% Initialization
[numN. numF]=size(X);
[numNL,numL]=size(Y);
% reduced dim (number of latent labels)
dim=method.param{1}.dim;
if ischar(dim)
    eval(['dim=',method.param{1}.dim,';']);
    dim=ceil(dim);
end
% is a weighting paramter for reproductablity term see the paper 
lambda=method.param{1}.lambda;
type=method.param{1}.type;  

model=cell(2,1);
time=cell(2,1);
tmptime=cputime;
%size check
sizeCheck;

%parameter set
ops=['-s',blanks(1),num2str(type),blanks(1),'-k',blanks(1),num2str(dim),blanks(1),'-t',blanks(1),'10',blanks(1)...
 '-l',blanks(1),num2str(lambda)'];

%% Leaning model
% to avoid use test instances and labels in this black box function
% we use random variables 
Xt= sprand(10,numF,0.1);
Yt= sprand(10,numL,0.1);
[W, H, wall_time]=train_ml(sparse(Y),sparse(X),Xt,Yt,ops);

model{1}=W;
model{2}=H;

