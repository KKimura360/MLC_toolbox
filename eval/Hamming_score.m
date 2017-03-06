function HammingLoss = Hamming_score(Pre_Labels,test_target)

    [num_class,num_instance]=size(Pre_Labels);
    miss_pairs=sum(sum(Pre_Labels==test_target));
    HammingLoss=miss_pairs/(num_class*num_instance);