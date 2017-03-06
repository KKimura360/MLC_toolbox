
function W = NMF(X,Y,opts);
%% X = feature x data matrix 
%% Y = label x data   matrix
%% J = size of U and S 

lambda=0.1;
iter=20;
eps=1e-12;
%J=opts.k;
J=size(Y,1);
U=sprand(size(X,1),J,1);
S=sprand(J,size(Y,1),1);
%Y=normalize_factor(Y',1);
%Y=Y';
%% Normalize Data
%X=normalize_factor(X,1);

[W D]=GraphConst(Y);


for i =1:iter
	U= U .* (X* (Y' *S')) ./ ((((U *S)*Y)*Y') *S'+ eps); 
%	S= ((U'*X) * Y') ./ ((( (U'*U) * S) * YY) +eps);
	%% Graphi Regualized Constraint With Graphs
    YW= lambda .*   S*W;
    YD= lambda .*   S*D;
	S= S.* (((U' * X) * Y') + YW) ./  ((((U'*U) * S) * Y)*Y' + YD +eps);
%	S=normalize_factor(S,2);
end 
W= U*S; 