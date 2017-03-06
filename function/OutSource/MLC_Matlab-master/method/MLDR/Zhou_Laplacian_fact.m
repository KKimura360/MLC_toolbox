function [A, IL] = Zhou_Laplacian_fact(H, edgeWeight)
% function [A, degree_vector, IL] = clique_expn_fact(H, edgeWeight)
% this function converts a hypergraph into a standard clique expansion graph, and then compute IL of
% this graph. Next, we factorize IL(or C) = A*A', and the matrix A is returned. Compared with 
% clique_expn, we do not construct the clique expansion graph explicitly. 
% ==============    Input Description
% H is the incidence matrix for a hypergraph
% H's size is |V|*|E|, where V is the set of vertices, and E is the set of hyperedges
% edgeWeight is a edgeNum-by-1 vector, it represents the weight for each hyperedge
%
% ==============    Output Description
% A: |V|-by-|E| matrix, and it satisfies IL = A * A';
% IL: the matrix used to run Spectral_Ng algorithm, and it is I - L, where L is the normalized
% Laplacian
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% Step 0. Get some basic information
[vertexNum, edgeNum] = size(H);

% if no weight is given, all edges will have the same weight
if nargin == 1
    edgeWeight = ones(edgeNum, 1);
end
d_e = sum(H);
% Make sure all vectors are column vectors.
d_e = d_e';
weight_matrix = H * diag(edgeWeight);
d_v = sum(weight_matrix, 2);

d_e_weight = sqrt(edgeWeight) ./ (sqrt(d_e));

A = diag(d_v.^(-1/2)) * H * diag(d_e_weight);

if nargout == 2
    IL = A * A';
end