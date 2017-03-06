function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=LIFT(train_data,train_target,test_data,test_target,ratio,svm)
%LIFT deals with multi-label learning problem by introducing label-specific features [1].
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=LIFT(train_data,train_target,test_data,test_target,ratio,svm)
%
%    Description
%
%       LIFT takes,
%           train_data       - An M1xN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           ratio            - The number of clusters (i.e. k1 for positive examples, k2 for negative examples) considered for the i-th class is set to
%                              k2=k1=min(ratio*num_pi,ratio*num_ni), where num_pi and num_ni are the number of positive and negative examples for the i-th class respectively.
%                              ***The default configuration is ratio=0.1***
%           svm              - A struct variable with two fields, i.e. svm.type and svm.para. 
%                              Specifically, svm.type gives the kernel type, which can take the value of 'RBF', 'Poly', 'Linear' or 'LibLinear'; 
%                              svm.para gives the corresponding parameters used for the specified kernel:
%                              1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-gamma*|x(i)-x(j)|^2)
%                              2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x(i),x(j)>+coefficient)^degree.
%                              3) if svm.type is 'Linear', then svm.para is [].
%                              *** The default configuration of svm is svm.type='Linear' with 'svm.para=[]' ***
%
%      and returns,
%           HammingLoss      - The hamming loss on testing data as described in [2]
%           RankingLoss      - The ranking loss on testing data as described in [2]
%           OneError         - The one-error on testing data as described in [2]
%           Coverage         - The coverage on testing data as described in [2]
%           Average_Precision- The average precision on testing data as described in [2]
%           Outputs          - The output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Pre_Labels       - If the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%
%  [1] M.-L. Zhang. LIFT: Multi-label learning with label-specific features, In: Proceedings of the 22nd International Joint Conference on Artificial Intelligence, 2011.
%
%  [2] R. E. Schapire, Y. Singer. BoosTexter: A boosting based system for text categorization. Machine Learning, 39(2/3): 135-168, 2000.

    if(nargin<6)
        svm.type='Linear';
        svm.para=[];
    end
    
    if(nargin<5)
        ratio=0.1;
    end
    
    if(nargin<4)
        error('Not enough input parameters, please type "help LIFT" for more information');
    end
    
    [num_train,dim]=size(train_data);
    [num_class,num_test]=size(test_target);
    
    P_Centers=cell(num_class,1);
    N_Centers=cell(num_class,1);
    
    %Find key instances of each label
    for i=1:num_class
        disp(['Performing clusteirng for the ',num2str(i),'/',num2str(num_class),'-th class']);
        
        p_idx=find(train_target(i,:)==1);
        n_idx=setdiff([1:num_train],p_idx);
        
        p_data=train_data(p_idx,:);
        n_data=train_data(n_idx,:);
        
        k1=min(ceil(length(p_idx)*ratio),ceil(length(n_idx)*ratio));
        k2=k1;
        
        if(k1==0)
            POS_C=[];
            [NEG_IDX,NEG_C]=kmeans(train_data,min(50,num_train),'EmptyAction','singleton','OnlinePhase','off','Display','off');
        else
            if(size(p_data,1)==1)
                POS_C=p_data;
            else
                [POS_IDX,POS_C]=kmeans(p_data,k1,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
            
            if(size(n_data,1)==1)
                NEG_C=n_data;
            else
                [NEG_IDX,NEG_C]=kmeans(n_data,k2,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
        end  
        
        P_Centers{i,1}=POS_C;
        N_Centers{i,1}=NEG_C;
    end
    
    switch svm.type
        case 'RBF'
            gamma=num2str(svm.para);
            str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(svm.para(1));
            coef=num2str(svm.para(2));
            degree=num2str(svm.para(3));
            str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            str='-t 0 -b 1';
        otherwise
            error('SVM types not supported, please type "help LIFT" for more information');
    end
    
    Models=cell(num_class,1);
    
    %Perform representation transformation and training
    for i=1:num_class        
        disp(['Building classifiers: ',num2str(i),'/',num2str(num_class)]);
        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_train/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;train_data(low:high,:)];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_train;
            
            tmp_mat=[centers;train_data(low:high,:)];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];
        end
        
        training_instance_matrix=data;
        training_label_vector=train_target(i,:)';
        
        Models{i,1}=svmtrain(training_label_vector,training_instance_matrix,str);      
    end
    
    %Perform representation transformation and testing
    Pre_Labels=[];
    Outputs=[];
    for i=1:num_class        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_test/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;test_data(low:high,:)];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_test;
            
            tmp_mat=[centers;test_data(low:high,:)];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];
        end
        
        testing_instance_matrix=data;
        testing_label_vector=test_target(i,:)';
        
        [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,Models{i,1},'-b 1');
        if(isempty(predicted_label))
            predicted_label=train_target(i,1)*ones(num_test,1);
            if(train_target(i,1)==1)
                Prob_pos=ones(num_test,1);
            else
                Prob_pos=zeros(num_test,1);
            end
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,pos_index);
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        end
    end
    
    HammingLoss=Hamming_loss(Pre_Labels,test_target);
    RankingLoss=Ranking_loss(Outputs,test_target);
    OneError=One_error(Outputs,test_target);
    Coverage=coverage(Outputs,test_target);
    Average_Precision=Average_precision(Outputs,test_target);