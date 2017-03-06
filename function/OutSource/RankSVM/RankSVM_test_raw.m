function [Outputs,Threshold,Pre_Labels]=RankSVM_test_raw(test_data,test_target,svm,Weights,Bias,SVs,Weights_sizepre,Bias_sizepre)
%RansSVM_test tests a multi-label ranking svm using the method described in [1] and [2].
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Threshold,Pre_Labels]=RankSVM_test(test_data,test_target,Weights,Bias,SVs,Wegihts_sizepre,Bias_sizepre)
%
%    Description
%
%       RankSVM_test takes,
%           test_data        - An MxN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           svm              - svm.type gives the type of svm used in testing, which can take the value of 'RBF', 'Poly' or 'Linear'; svm.para gives the corresponding parameters used for the svm:
%                              1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-Gamma*|x(i)-x(j)|^2)
%                              2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x(i),x(j)>+coefficient)^degree.
%                              3) if svm.type is 'Linear', then svm.para is [].
%           Weights          - The value for beta(ki) as described in the appendix of [1] is stored in Weights(k,i)
%           Bias             - The value for b(i) as described in the appendix of [1] is stored in Bias(1,i)
%           SVs              - The ith support vector is stored in SVs(:,i)
%           Weights_sizepre  - An 1xQ weight for the size predictor as described in [2]
%           Bias_sizepre     - The bias for the size predictor as described in [2]
%      and returns,
%           Outputs          - The output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Threshold        - The threshold of the ith testing instance for assessing class membership is stored in Threshold(1,i)
%           Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is 0 
%
%    For more details,please refer to [1], [2] and [3].   
%
%    [1] Elisseeff A, Weston J. Kernel methods for multi-labelled classfication and categorical regression problems. Technical Report, BIOwulf Technologies, 2001.
%    [2] Elisseeff A,Weston J. A kernel method for multi-labelled classification. In: Dietterich T G, Becker S, Ghahramani Z, eds. Advances in Neural Information Processing Systems 14, Cambridge, MA: MIT Press, 2002, 681-687.
%    [3] Schapire R. E., Singer Y. BoosTexter: a boosting based system for text categorization. Machine Learning, 39(2/3): 135-168, 2000.

[num_testing,tempvalue]=size(test_data);
[num_class,tempvalue]=size(test_target);
[tempvalue,num_training]=size(SVs);

Label=cell(num_testing,1);
not_Label=cell(num_testing,1);
Label_size=zeros(1,num_testing);
size_alpha=zeros(1,num_testing);
for i=1:num_testing
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    size_alpha(1,i)=Label_size(1,i)*(num_class-Label_size(1,i));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

kernel=zeros(num_testing,num_training);
if(strcmp(svm.type,'RBF'))
    for i=1:num_testing
        for j=1:num_training
            gamma=svm.para;
            kernel(i,j)=exp(-gamma*sum((test_data(i,:)'-SVs(:,j)).^2))
        end
    end
else
    if(strcmp(svm.type,'Poly'))
        for i=1:num_testing
            for j=1:num_training
                gamma=svm.para(1);
                coefficient=svm.para(2);
                degree=svm.para(3);
                kernel(i,j)=(gamma*test_data(i,:)*SVs(:,j)+coefficient)^degree;
            end
        end
    else
        for i=1:num_testing
            for j=1:num_training
                kernel(i,j)=test_data(i,:)*SVs(:,j);
            end
        end
    end
end

Outputs=zeros(num_class,num_testing);
for i=1:num_testing
    for k=1:num_class
        temp=0;
        for j=1:num_training
            temp=temp+Weights(k,j)*kernel(i,j);            
        end
        temp=temp+Bias(k);
        Outputs(k,i)=temp;
    end
end
Threshold=([Outputs',ones(num_testing,1)]*[Weights_sizepre,Bias_sizepre]')';
Pre_Labels=zeros(num_class,num_testing);
for i=1:num_testing
    for k=1:num_class
        if(Outputs(k,i)>=Threshold(1,i))
            Pre_Labels(k,i)=1;
        else
            Pre_Labels(k,i)=0;
        end
    end
end
