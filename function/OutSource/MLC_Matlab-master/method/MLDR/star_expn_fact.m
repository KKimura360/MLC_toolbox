function [A, degree_vector, IL]  = star_expn_fact(H, edgeWeight)
% function [A, degree_vector, IL]  = star_expn_fact(H, edgeWeight)
% this function converts a hypergraph into a star expansion graph. 
% In addition, we factorize IL = A*A' were IL = I - L, where I is the identify matrix and L is the Laplacian matrix.
% We return the matrix A. 
% ==============    Input Description
% H is the incidence matrix for a hypergraph
% H's size is |V|*|E|, where V is the set of vertices, and E is the set of hyperedges
% edgeWeight is a edgeNum-by-1 vector, it represents the weight for each hyperedge
%
% ==============    Output Description
% A: |V|-by-|E| matrix, and it satisfies IL = A * A';
% IL: |V|-by-|V| matrix, it is the matrix used to run Spectral_Ng algorithm, and it is I - L, where L is the normalized
% Laplacian
% degree_vector: |V|-by-1 vector, it contains the degree of each vertex in the star expansion graph
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 0. Get some basic information
[vertexNum, edgeNum] = size(H);

if nargin == 1
    edgeWeight = ones(edgeNum, 1);
end
edgeDegree = sum(H);
edgeDegree = edgeDegree';


edgeWeight_normalize = edgeWeight ./ edgeDegree;
weight_matrix = H * diag(edgeWeight_normalize);


starEdgeDegree = sum(weight_matrix);
starVertexDegree = sum(weight_matrix, 2);
if vertexNum > 3000
    starDeg_matrix = sparse(1:vertexNum, 1:vertexNum, starVertexDegree.^(-1/2), vertexNum, vertexNum);
else
    starDeg_matrix = diag(starVertexDegree.^(-1/2));
end
A = starDeg_matrix * weight_matrix * diag(starEdgeDegree.^(-1/2));
% A = diag(starVertexDegree.^(-1/2)) * weight_matrix * diag(starEdgeDegree.^(-1/2));

degree_vector = sum(weight_matrix);
degree_vector = degree_vector';
degree_vector = weight_matrix * degree_vector;
% 
% degree_vector = weight_matrix * weight_matrix';
% degree_vector = sum(degree_vector, 2);

if nargout == 3
    IL = A * A';
end





