function [nets,errors]=BPMLL_train_raw(train_data,train_target,hidden_neuron,alpha,epochs,intype,outtype,Cost,min_max)
%BPMLL_train trains a multi-label neural network
%
%    Syntax
%
%       [nets,errors]=BPMLL_train(train_data,train_target,hidden_neuron,alpha,epochs,intype,outtype,Cost,min_max)
%
%    Description
%
%       BPMLL_train takes,
%           train_data    - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target  - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           hidden_neuron - Number of hidden neurons used in the network
%           alpha         - Learning rate for updating weights and biases, default=0.05
%           epochs        - Maximum number of training epochs, default=100
%           intype        - The type of activation function used for the hidden neurons, 1 for 'logsig', 2 for 'tansig', default=2
%           outtype       - The type of activation function used for the output neurons, 1 for 'logsig', 2 for 'tansig', default=2
%           Cost          - Cost parameter used for regularization, default=0.1
%           min_max       - A Nx2 array, where the minimum value and maximum value of the ith input dimension are stored in min_max(i,1) and min_max(i,2) respectively, default=[zeros(N,1),ones(N,1)]
%      and returns,
%           nets          - nets{i,1} stores the trained multi-labe neural network after i iterations
%           errors        - errors{i,1} stores the global training error after i iterations

    rand('state',sum(100*clock));

%Check input parameters    
    if(nargin<9)
        min_max=minmax(train_data');
    end
    
    if(nargin<8)
        Cost=0.1;
    end
    
    if(nargin<7)
        outtype=2;
    end
    
    if(nargin<6)
        intype=2;
    end
    
    if(nargin<5)
        epochs=100;
    end
    
    if(nargin<4)
        alpha=0.05;
    end        

%Set activiation function types    
    if(intype==1)
        in='logsig';
    else
        in='tansig';
    end
    if(outtype==1)
        out='logsig';
    else
        out='tansig';
    end
    
%Initializing    
    [num_class,num_training]=size(train_target);
    [num_training,Dim]=size(train_data);
    
    Label=cell(num_training,1);
    not_Label=cell(num_training,1);
    Label_size=zeros(1,num_training);
    for i=1:num_training
        temp=train_target(:,i);
        Label_size(1,i)=sum(temp==ones(num_class,1));
        for j=1:num_class
            if(temp(j)==1)
                Label{i,1}=[Label{i,1},j];
            else
                not_Label{i,1}=[not_Label{i,1},j];
            end
        end
    end
    
    Cost=Cost*2;
    
%Initialize multi-label neural network    
    incremental=ceil(rand*100);
    for randpos=1:incremental
        net=newff(min_max,[hidden_neuron,num_class],{in,out});
    end
    old_goal=realmax;

%Training phase
    for iter=1:epochs
        disp(strcat('training epochs: ',num2str(iter)));
        tic;
        for i=1:num_training
            net=update_net_ml(net,train_data(i,:)',train_target(:,i),alpha,Cost/num_training,in,out);
        end
        
        cur_goal=0;
        for i=1:num_training
            if((Label_size(i)~=0)&(Label_size(i)~=num_class))
                output=sim(net,train_data(i,:)');
                temp_goal=0;
                for m=1:Label_size(i)
                    for n=1:(num_class-Label_size(i))
                        temp_goal=temp_goal+exp(-(output(Label{i,1}(m))-output(not_Label{i,1}(n))));
                    end
                end
                temp_goal=temp_goal/(m*n);
                cur_goal=cur_goal+temp_goal;
            end
        end
        cur_goal=cur_goal+Cost*0.5*(sum(sum(net.IW{1}.*net.IW{1}))+sum(sum(net.LW{2,1}.*net.LW{2,1}))+sum(net.b{1}.*net.b{1})+sum(net.b{2}.*net.b{2}));        

        disp(strcat('Global error after ',num2str(iter),' epochs is: ',num2str(cur_goal)));
        old_goal=cur_goal;

        nets{iter,1}=net;
        errors{iter,1}=old_goal;

        toc;
    end    

    disp('Maximum number of epochs reached, training process completed');