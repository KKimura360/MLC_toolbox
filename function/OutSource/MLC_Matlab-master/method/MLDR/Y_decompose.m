function H = Y_decompose(Y, tech_option)
% This function deals with the general eigenvalue problem
%        (XX^T)^+ (X S X^T) w = lambda * w, where S = HH^T
% Note that S is n-by-n, and H is n-by-k. Specificaly, this function decompose S = H*H'
% where S is defined using Y, and the matrix H is returned in this function.
% ~~~~~~~~~~ The techniques implememted by this function (or interface) incldue
% 1. CCA
% 2. Orthonormal partial least squares
% 3. Hypergraph spectral learning
%       3.1 Clique Expansion
%       3.2 Star Expansion
% ============= Input Description ============
% Y: k-by-n matrix, the label matrix. Y(i, j) is either 1 or 0.
% ============= Output Description ===========
% H: n-by-k matrix, where k is usually the number of labels (classes).
%
% Copyright(c) Liang Sun (sun.liang@asu.edu), Shuiwang Ji (shuiwang.ji@asu.edu), and Jieping Ye (jieping.ye@asu.edu), Arizona State Univerisity
%

% The default option is CCA.
if nargin < 2
    tech_option = 'cca';
end
if (strcmpi(tech_option, 'cca')==0) && (strcmpi(tech_option, 'pls')==0) && (strcmpi(tech_option, 'star')==0) ...
        && (strcmpi(tech_option, 'clique')==0)  && (strcmpi(tech_option, 'zhou')==0) && (strcmpi(tech_option, 'lda')==0)
    tech_option = 'cca';
end

% Compute H matrix for different dimensionality reduction algorithms.
if strcmpi(tech_option, 'cca')
    % CCA
    [U, S, V] = svd(Y, 'econ');
    H = V * U';
elseif strcmpi(tech_option, 'hsci')
    %HSCI with inner product kernel
    %  H*L*H = (HY) * (Y'H')  H is 
    H = colCenter(Y)'; 
elseif strcmpi(tech_option, 'pls')
    % PLS
    H = Y';
elseif strcmpi(tech_option, 'clique')
    % HSL--Clique expansion
    incidence_matrix = Y';
    incidence_matrix(incidence_matrix ~= 1) = 0;
    H = clique_expn_fact(incidence_matrix);
elseif strcmpi(tech_option, 'star')
    % HSL--Star expansion
    incidence_matrix = Y';
    incidence_matrix(incidence_matrix ~= 1) = 0;
    H = star_expn_fact(incidence_matrix);
elseif strcmpi(tech_option, 'zhou')
    % HSL-Zhou's Laplacian
    incidence_matrix = Y';
    incidence_matrix(incidence_matrix ~= 1) = 0;    
    H = Zhou_Laplacian_fact(incidence_matrix);
elseif strcmpi(tech_option, 'lda')
    % LDA
    n_array = sum(Y, 2);
    k = length(n_array);
    n = size(Y,2);
    H = zeros(n, k);
    h = 1./sqrt(n_array);
    
    for i = 1:k
        H(Y(i,:)==1,i) = h(i);
    end    
end

H = rowCenter(H);