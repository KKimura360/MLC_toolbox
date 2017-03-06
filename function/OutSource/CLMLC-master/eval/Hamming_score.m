function HammingScore = Hamming_score(Pre_Labels,test_target)
%HAMMINGSCORE Hamming Score
%
%      Syntax:
% 
%          HammingScore = Hamming_score(Pre_Labels,test_target)
%
%      Input:
%
%          Pre_Labels          L x Nt predicted label matrix           
%          test_target         L x Nt groundtruth label matrix
%
%      Output:
%
%          HammingScore        Hamming Score 

    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels==test_target));
    HammingScore=miss_pairs/(num_class*num_instance);
    
end