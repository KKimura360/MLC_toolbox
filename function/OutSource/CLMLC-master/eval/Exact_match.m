function ExactMatch = Exact_match(Pre_Labels,test_target)
%EXACTMATCH Exact Match
%
%      Syntax:
% 
%          ExactMatch = Exact_match(Pre_Labels,test_target)
%
%      Input:
%
%          Pre_Labels          L x Nt predicted label matrix           
%          test_target         L x Nt groundtruth label matrix
%
%      Output:
%
%          ExactM              Exact-Match

[~,num_instance] = size(Pre_Labels);
match_pairs = 0;
for i = 1:num_instance
    if Pre_Labels(:,i) == test_target(:,i)
        match_pairs = match_pairs + 1;
    end
end
ExactMatch = match_pairs/num_instance;

end

