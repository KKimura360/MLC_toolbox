function X = myDisc( X,num_state,factor )
%MYDIST Discretize the training data for computing mutual information
% Divide the values of numeric features into three or five categories
% according to [mean +/- factor*std].
%
% 
% X:          Training data in N x F
% num_state:  Number of states of discretized features
% factor:     Scaling factor

num_F = size(X,2);
if num_state == 3
    values = [-1,0,1];
elseif num_state == 5
    values = [-2,-1,0,1,2];
else
    disp('#State for discretization is set as 3.')
    num_state = 3;
end
for i = 1 : num_F
    array_fea = X(:,i);
%     if max(array_fea) ~= 1
    if  max(array_fea(array_fea~=1)) ~= 0
        meanF = mean(array_fea);
        stdF1 = factor*std(array_fea);
        if num_state == 3
            edges = [-inf; meanF-stdF1; meanF+stdF1; inf];
        else
            stdF2 = 2*stdF1;
            edges = [-inf; meanF-stdF2; meanF-stdF1; meanF+stdF1; meanF+stdF2; inf];
        end
        %% NOTE new version round function is needed
          edges = round(edges,3);
        X(:,i) = discretize(array_fea,edges,values);
    end
end

end
