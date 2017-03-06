function [Centroids,Sigma_value,Weights,tr_time]=ML_RBF_train(train_data,train_target,ratio,mu)
%ML_RBF_train trains a multi-label RBF learner as in [1]
%
%    Syntax
%
%       [Centroids,Sigma_value,Weights,tr_time]=ML_RBF_train(train_data,train_target,ratio,mu)
%
%    Description
%
%       ML_RBF_train takes,
%           train_data    - An MxN array, the i-th training instance is stored in train_data(i,:)
%           train_target  - A QxM array, if the i-th training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           ratio         - The number of centroids of the i-th class is set to be ratio*Ti, where Ti is the number of training instances with lable i
%           mu            - The ratio used to determine the standard deviation of the Gaussian activation function [1]
%      and returns,
%           Centroids    - A KxN matrix, where the k-th centroid of the RBF neural network is stored in Centroids(k,:)
%           Sigma_value  - A 1xK vector, where the sigma value for the k-th centroid is stored in Sigma_value(1,k)
%           Weights      - A (K+1)xQ matrix used for label prediction
%           tr_time      - The time spent in training
%
% [1] M.-L. Zhang. ML-RBF: RBF neural networks for multi-label learning. Neural Processing Letters, 2009, 29(2): 61-74.
   
    start_time=cputime;
    
    [num_class,num_train]=size(train_target);
    Dim=size(train_data,2);
    
    disp('First layer clustering...');
    num_cluster=zeros(1,num_class);
    for j=1:num_class
        num_cluster(1,j)=ceil(ratio*sum(train_target(j,:)==1));
    end
    num_centroid=sum(num_cluster);
    Centroids=zeros(num_centroid,Dim);
    for j=1:num_class
        disp(strcat(num2str(j),'/',num2str(num_class)));
        temp_index=find(train_target(j,:)==1);
        temp_train_data=train_data(~,:);
        [IDX,CEN]=kmeans(temp_train_data,num_cluster(1,j));        
        low=sum(num_cluster(1:j-1))+1;
        high=sum(num_cluster(1:j));
        Centroids(low:high,:)=CEN;        
    end
    
    distfun='euclidean';    
    Y1=pdist(Centroids,distfun);
    centroid_dist=squareform(Y1);
    numerator=sum(sum(triu(centroid_dist,1)));
    denominator=num_centroid*(num_centroid-1)/2;
    sigma=mu*(numerator/denominator);
    
    Sigma_value=zeros(1,num_centroid);
    counter=0;
    for j=1:num_class
        sigma_j=sigma;

        for k=1:num_cluster(j)
            counter=counter+1;
            Sigma_value(1,counter)=sigma_j;
        end
    end
    
    disp('Second layer optimization...');
    A=zeros(num_train,num_centroid+1);
    B=zeros(num_train,num_class);
    
    for i=1:num_train
        if(mod(i,100)==0)
            disp(strcat(num2str(i),'/',num2str(num_train)));
        end
        
        temp_vec=zeros(1,num_centroid);
        
        for k=1:num_centroid
            temp_sigma=Sigma_value(1,k);
            vec1=train_data(i,:);
            vec2=Centroids(k,:);
            tmp=sqrt((vec1-vec2)*(vec1-vec2)');
            temp_vec(1,k)=exp((-tmp^2)/(2*temp_sigma^2));
        end

        temp_vec=[1,temp_vec];
        A(i,:)=temp_vec;
                
        B(i,:)=train_target(:,i)';
    end
    
    Weights=A\B;
    
    tr_time=cputime-start_time;