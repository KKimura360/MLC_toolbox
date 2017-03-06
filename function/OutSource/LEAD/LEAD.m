function [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=LEAD(train_data,train_target,test_data,test_target,DAG,svm)
%LEAD implements the multi-label learning approach which explicitly exploits the label dependencies 
%with Bayesian network structure [1]. 
%
%Two important notes on this implementation:
%
%1) Due to proprietary reasons for various Bayesian network learning packages, the identification 
%process for the Bayesian network structure, i.e. steps 1 & 2 as shown in Subsection 2.2.2 of [1], 
%is NOT incorporated in this implementation.
%
%Therefore, we assume that the Bayesian network structure is areadly known and proceed to the 
%subsequent learning procedure, i.e.  steps 3 & 4 as shown in Subsection 2.2.2. The users can freely
%employ any package for Bayesian network structure learning before calling this function.
%
%2) Libsvm [2] is utilized as the base classifier for LEAD. 
%
%Threfore, the libsvm package SHOULD BE AVAILABLE under matlab path to properly run the LEAD function.
%
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels]=LEAD(train_data,train_target,test_data,test_target,DAG,svm)
%
%    Description
%
%       LEAD takes,
%           train_data       - An M1xd array, the feature set of the ith training example is stored in train_data(i,:)
%           train_target     - A QxM1 array, if the ith training example belongs to the jth class, then train_target(j,i) equals 1, otherwise train_target(j,i) equals 0
%           test_data        - An M2xd array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals 1, otherwise test_target(j,i) equals 0
%           DAG              - A QxQ binary matrix, if the i-th label is the parent of the j-th label as implied by the Bayesian network structure, then DAG(i,j)=1, otherwise, DAG(i,j)=0
%           svm              - A struct variable with two fields, i.e. svm.type and svm.para. 
%                              Specifically, svm.type gives the kernel type, which can take the value of 'RBF', 'Poly' or 'Linear'; 
%                              svm.para gives the corresponding parameters used for the specified kernel:
%                              1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-gamma*|x(i)-x(j)|^2)
%                              2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x(i),x(j)>+coefficient)^degree.
%                              3) if svm.type is 'Linear', then svm.para is [].
%                              *** The default configuration of svm is svm.type='Linear' ***
%
%      and returns,
%           HammingLoss       - The hamming loss on test data as defined in [3]
%           RankingLoss       - The ranking loss on test data as defined in [3]
%           OneError          - The one-error on test data as defined in [3]
%           Coverage          - The coverage on test data as defined in [3], which is further normalized by Q so as to make this metric vary between [0,1]
%           Average_Precision - The average precision on test data as defined in [3]
%           Outputs           - The output of the ith testing instance on the jth class is stored in Outputs(j,i)
%           Outputs           - A QxM2 array, the probability of the ith testing instance belonging to the jth class is stored in Outputs(j,i)
%           Pre_Labels        - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is 1, otherwise Pre_Labels(j,i) is 0
%
%
%[1] M.-L. Zhang and K. Zhang. Multi-label learning by exploiting label dependency. In: Proceedings 
%of the 16th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington D. C., 2010, 999-1007.
%
%[2] C.-C. Chang and C.-J. Lin. LIBSVM: a library for support vector machines, Technical Report, 
%2001. [http://www.csie.ntu.edu.tw/~cjlin/libsvm]
%
%[3] Schapire R. E., Singer Y. BoosTexter: a boosting based system for text categorization. Machine 
%Learning, 39(2/3): 135-168, 2000.

    if(nargin<6)
        svm.type='Linear';
    end
    
    if(nargin<5)
        error('Not enough input parameters');
    end
    
    %Implementing step 3 as shown in Subsection 2.2.2 [1]
    
    num_class=size(train_target,1);
    
    tmp_DAG=DAG;
    
    order=zeros(1,num_class);
    
    selected=[];
    
    for i=1:num_class
        b=(sum(tmp_DAG)==0);
        idx=find(b);
        idx=setdiff(idx,selected);
        if(isempty(idx))
            error('The graph is not a directed acyclic graph, please check again.');
        else            
            idx=idx(1);
            order(1,idx)=i;
        end
        tmp_DAG(idx,:)=0;
        selected=[selected,idx];
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
            gamma=1;
            str=['-t 2 -g ',gamma,' -b 1'];
    end
    
    classifiers=cell(1,num_class);
    
    for i=1:num_class
        disp(['Training classifier: ',num2str(i),'/',num2str(num_class)]);
        tmp_parent=find(DAG(:,i))';
        classifiers{1,i}.parent=tmp_parent;
        
        training_instance_matrix=[train_data,train_target(tmp_parent,:)'];
        training_label_vector=train_target(i,:)';
        
        tmp_model = svmtrain(training_label_vector,training_instance_matrix,str);
        if(isempty(tmp_model.SVs))
            num_a=sum(train_target(i,:)==1);
            num_b=sum(train_target(i,:)==0);
            if(num_a>=num_b)
                tmp_model.majority_label=1;
            else
                tmp_model.majority_label=0;
            end
        end
        classifiers{1,i}.model=tmp_model;
    end
    
    %Implementing step 4 as shown in Subsection 2.2.2 [1]
    
    [num_class,num_test]=size(test_target);
    
    Outputs=zeros(num_class,num_test);    

    [tmp,idx]=sort(order);
    psize=20;
    
    for j=1:num_class
        disp(['Computing probabilities for the ',num2str(j),'-th class...']);
        nodes=idx(j);
        labels=-ones(1,num_class);
        labels(1,idx(j))=1;        
        Outputs(idx(j),:)=LEAD_Get_Prob(test_data,nodes,labels,order,classifiers,psize,Outputs);
    end
    
    Pre_Labels=zeros(num_class,num_test);
    for i=1:num_test
        for j=1:num_class
            if(Outputs(j,i)>=0.5)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=0;
            end
        end
    end
    HammingLoss=Hamming_loss((Pre_Labels-0.5)*2,(test_target-0.5)*2);

    RankingLoss=Ranking_loss(Outputs,(test_target-0.5)*2);
    OneError=One_error(Outputs,(test_target-0.5)*2);
    Coverage=coverage(Outputs,(test_target-0.5)*2);
    Coverage=Coverage/num_class;
    Average_Precision=Average_precision(Outputs,(test_target-0.5)*2);