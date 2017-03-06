function distance = distM(X,Y,M)
% square norm of the euclidean distance between row vectors in X, Y
nx	= size(X,1);
ny	= size(Y,1);
distance=diag(X*M*X')*ones(1,ny)+ones(nx,1)*(diag(Y*M*Y'))' - 2*(X*M*Y');