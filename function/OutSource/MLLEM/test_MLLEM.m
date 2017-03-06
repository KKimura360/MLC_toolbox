
function [Results]=test_MLLEM(Xtr,Ytr,Xts,Yts,G,H,params)

%% Input
% G  N x K matrix
% H  N x K matrix 
% params.k3= number of k-nn for NL
% params.lambda ridge parameter
% params.method : option to classification
% 		'L': Linear
%		'NL': Nonlinear
% params.dim : the number of dimension to use
% params.data: the dataset
%% In addition, parameters used in train_MLLEM.m is also needed.

%%Output
% Result.auc  AUC
% Result.top1 top1 precision
% Result.top3 top3 precision
% Result.top5 top5 precision

%% Check the dimension
dim=min(size(G,2),params.dim);
G=G(:,(end-dim)+1:end);
H=H(:,(end-dim)+1:end);

%% Linear
	% Learning regression for testdata projection
if params.method=='L'
    % Conduct ridge regression
    [ww]=ridgereg(G, Xtr, params.lambda);
 	% Projection test instances into K-dimensional space 
	Gts= [ones(size(Xts,1),1), Xts] * ww;

   % Compute distance between test instances and labels on K-dimensional space
	Y_hat=L2_distance(Gts',H');
	Y_hat=max(max(Y_hat))-Y_hat; % reversing scores

elseif params.method=='NL'                

  %% Non-linear
   % Find k nearest neighbors 
   tmp=L2_distance(Xts',Xtr');
   [val Ind]=sort(tmp,2);
   Fpt=zeros(size(tmp,1),dim); 

   for i= 1:size(tmp,1);
	Fpt(i,:)=squeeze(mean(G(Ind(i,1:params.k3),:),1)); %Mean of k-nearest of neighbors
   end
         Y_hat=L2_distance(Fpt',H');
         Y_hat=max(max(Y_hat))-Y_hat; % reversing scores
else
    error('Wrong parameter: params.method, it must be L or NL');
end

%% Evalute score ranking
Results.top1=TopPrec(Yts,Y_hat,1);
Results.top3=TopPrec(Yts,Y_hat,3);
Results.top5=TopPrec(Yts,Y_hat,5);  
[Results.auc]=AVGauc(Yts,Y_hat);




