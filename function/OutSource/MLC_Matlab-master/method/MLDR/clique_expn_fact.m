function [A, degree_vector, IL] = clique_expn_fact(H, edgeWeight)
% function [A, degree_vector, IL] = clique_expn_fact(H, edgeWeight)
% this function converts a hypergraph into a standard clique expansion graph, and then compute IL of
% this graph. Next, we factorize IL(or C) = A*A', and the matrix A is returned. 
% ==============    Input Description
% H is the incidence matrix for a hypergraph
% H's size is |V|*|E|, where V is the set of vertices, and E is the set of hyperedges
% edgeWeight is a |E|-by-1 vector, it represents the weight for each hyperedge
%
% ==============    Output Description
% A: |V|-by-|E| matrix, and it satisfies IL = A * A';
% IL: |V|-by-|V| matrix, it is the matrix used to run Spectral_Ng algorithm, and it is I - L, where L is the normalized
% Laplacian
% degree_vector: |V|-by-1 vector, it contains the degree of each vertex in the clique expansion graph. 
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 0. Get some basic information
[vertexNum, edgeNum] = size(H);

% if no weight is given, all edges will have the same weight
if nargin == 1
    edgeWeight = ones(edgeNum, 1);
end
edgeDegree_matrix_half = diag(edgeWeight.^(1/2));

weight_matrix = H * diag(edgeWeight);
% temp = H' * ones(size(H, 1), 1);
temp = sum(H);
temp = temp';
degree_vector = weight_matrix * temp;

% % weight_matrix = weight_matrix - diag(diag(weight_matrix));
% for i = 1:size(weight_matrix)
%     weight_matrix(i,i) = 0;
% end
% degree_vector = sum(weight_matrix, 2);

degree_matrix_halfInv = sparse(1:vertexNum, 1:vertexNum, degree_vector.^(-1/2), vertexNum, vertexNum);
% degree_matrix_halfInv = diag(degree_vector.^(-1/2));

A = degree_matrix_halfInv * H * edgeDegree_matrix_half;
if nargout == 3
    IL = A * A';
end