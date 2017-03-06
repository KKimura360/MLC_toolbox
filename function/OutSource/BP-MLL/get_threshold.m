function Threshold=get_threshold(train_data,train_target,test_data,net)
%get_threshold gets the threshold value for each testing instance
%    The same strategy as described in [1] and [2] is used to train the threshold predictor 
%
%    Syntax
%
%       Threshold=get_threshold(train_data,train_target,test_data,net)
%
%    Description
%
%       get_threshold takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           net              - The trained multi-label neural network
%      and returns,
%           Threshold        - The threshold of the ith testing instance for assessing class membership is stored in Threshold(1,i)
%
%[1] Elisseeff A, Weston J. Kernel methods for multi-labelled classfication and categorical regression problems. Technical Report, BIOwulf Technologies, 2001.
%[2] Elisseeff A,Weston J. A kernel method for multi-labelled classification. In: Dietterich T G, Becker S, Ghahramani Z, eds. Advances in Neural Information Processing Systems 14, Cambridge, MA: MIT Press, 2002, 681-687.

    [num_class,num_training]=size(train_target);
    [num_testing,tempvalue]=size(test_data);
    
    Label=cell(num_training,1);
    not_Label=cell(num_training,1);
    for i=1:num_training
        temp=train_target(:,i);
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1}=[Label{i,1},j];
            else
                not_Label{i,1}=[not_Label{i,1},j];
            end
        end
    end

    Left=sim(net,train_data')';
    Right=zeros(num_training,1);
    for i=1:num_training
        temp=Left(i,:);
        [temp,index]=sort(temp);
        candidate=zeros(1,num_class+1);
        candidate(1,1)=temp(1)-0.1;
        for j=1:num_class-1
            candidate(1,j+1)=(temp(j)+temp(j+1))/2;
        end
        candidate(1,num_class+1)=temp(num_class)+0.1;
        miss_class=zeros(1,num_class+1);
        for j=1:num_class+1
            temp_notlabels=index(1:j-1);
            temp_labels=index(j:num_class);
            [tempvalue,false_neg]=size(setdiff(temp_notlabels,not_Label{i,1}));
            [tempvalue,false_pos]=size(setdiff(temp_labels,Label{i,1}));
            miss_class(1,j)=false_neg+false_pos;
        end
        [temp_minimum,temp_index]=min(miss_class);
        Right(i,1)=candidate(1,temp_index);
    end
    Left=[Left,ones(num_training,1)];
    tempvalue=(Left\Right)';
    Weights_sizepre=tempvalue(1:num_class);
    Bias_sizepre=tempvalue(num_class+1);
    
    Outputs=sim(net,test_data');
    Threshold=([Outputs',ones(num_testing,1)]*[Weights_sizepre,Bias_sizepre]')';