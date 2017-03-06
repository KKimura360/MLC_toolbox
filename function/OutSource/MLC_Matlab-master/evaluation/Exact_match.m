function ExactMatch = Exact_match(Pre_Labels,test_target)
%EXACTMATCH 此处显示有关此函数的摘要
%   此处显示详细说明

[~,num_instance] = size(Pre_Labels);
match_pairs = 0;

for i = 1:num_instance
    if Pre_Labels(:,i) == test_target(:,i)
        match_pairs = match_pairs + 1;
    end
end

ExactMatch = match_pairs/num_instance;

end

