function [Weights,Bias,SVs,Weights_sizepre,Bias_sizepre,svm_used,iteration]=RankSVM_train_raw(train_data,train_target,svm,cost,lambda_tol,norm_tol,max_iter)
%RansSVM_train trains a multi-label ranking svm using the method described in [1] and [2].
%
%    Syntax
%
%       [Weights,Bias,SVs,Weights_size,Bias_size]=RankSVM_train(train_data,train_target,svm,cost,lambda_tol,norm_tol,max_iter)
%
%    Description
%
%       RankSVM_train takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           svm          - svm.type gives the type of svm used in training, which can take the value of 'RBF', 'Poly' or 'Linear'; svm.para gives the corresponding parameters used for the svm:
%                          1) if svm.type is 'RBF', then svm.para gives the value of gamma, where the kernel is exp(-Gamma*|x(i)-x(j)|^2)
%                          2) if svm.type is 'Poly', then svm.para(1:3) gives the value of gamma, coefficient, and degree respectively, where the kernel is (gamma*<x(i),x(j)>+coefficient)^degree.
%                          3) if svm.type is 'Linear', then svm.para is [].
%           cost         - The value of 'C' used in the SVM, default=1
%           lambda_tol   - The tolerance value for lambda described in the appendix of [1]; default value is 1e-6
%           norm_tol     - The tolerance value for difference between alpha(p+1) and alpha(p) described in the appendix of [1]; default value is 1e-4
%           max_iter     - The maximum number of iterations for RankSVM, default=50
%      and returns,
%           Weights          - The value for beta(ki) as described in the appendix of [1] is stored in Weights(k,i)
%           Bias             - The value for b(i) as described in the appendix of [1] is stored in Bias(1,i)
%           SVs              - The ith support vector is stored in SVs(:,i)
%           Weights_sizepre  - An 1xQ weight for the size predictor as described in [2]
%           Bias_sizepre     - The bias for the size predictor as described in [2]
%           svm_used         - The same as input svm, used for future testing
%      
%    For more details,please refer to [1] and [2].   
%
%    [1] Elisseeff A, Weston J. Kernel methods for multi-labelled classfication and categorical regression problems. Technical Report, BIOwulf Technologies, 2001.
%    [2] Elisseeff A,Weston J. A kernel method for multi-labelled classification. In: Dietterich T G, Becker S, Ghahramani Z, eds. Advances in Neural Information Processing Systems 14, Cambridge, MA: MIT Press, 2002, 681-687.

%Initializing
if(nargin<3)
    error('Not enough input parameters, please check again.');
end

if(nargin<7)
    max_iter=50;
end

if(nargin<6)
    norm_tol=1e-4;
end

if(nargin<5)
    lambda_tol=1e-6;
end

if(nargin<4)
    cost=1;
end

%Preprocessing input data
SVs=[];
target=[];
[num_training,tempvalue]=size(train_data);
[num_class,tempvalue]=size(train_target);
for i=1:num_training
    temp=train_target(:,i);
    if((sum(temp)~=num_class)&(sum(temp)~=-num_class))
        SVs=[SVs,train_data(i,:)'];
        target=[target,temp];
    end
end

[Dim,num_training]=size(SVs);
Label=cell(num_training,1);
not_Label=cell(num_training,1);
Label_size=zeros(1,num_training);
size_alpha=zeros(1,num_training);
for i=1:num_training
    temp=target(:,i);
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

kernel=zeros(num_training,num_training);
if(strcmp(svm.type,'RBF'))
    for i=1:num_training
        for j=1:num_training
            gamma=svm.para(1);
            kernel(i,j)=exp(-gamma*sum((SVs(:,i)-SVs(:,j)).^2));
        end
    end
else
    if(strcmp(svm.type,'Poly'))
        for i=1:num_training
            for j=1:num_training
                gamma=svm.para(1);
                coefficient=svm.para(2);
                degree=svm.para(3);
                kernel(i,j)=(gamma*SVs(:,i)'*SVs(:,j)+coefficient)^degree;
%                 kernel(i,j)=(gamma*sum(SVs(:,i)-SVs(:,j))+coefficient)^degree;
            end
        end
    else
        for i=1:num_training
            for j=1:num_training
                kernel(i,j)=SVs(:,i)'*SVs(:,j);
            end
        end
    end
end
svm_used=svm;

%Begin training phase

%data initializing
Alpha=zeros(1,sum(size_alpha));

c_value=cell(1,num_class);
for k=1:num_class
    c_value{1,k}=zeros(num_class,num_class);
    c_value{1,k}(k,:)=ones(1,num_class);
    c_value{1,k}(:,k)=-ones(num_class,1);
end

%Find the Alpha value using Franke and Wolfe method [1]
continuing=true;
iteration=0;
while(continuing)
%computing Beta    
    tic;
    iteration=iteration+1;
    disp(strcat('current iteration: ',num2str(iteration)));
    Beta=zeros(num_class,num_training);
    for k=1:num_class
        for i=1:num_training
            for m=1:Label_size(i)
                for n=1:(num_class-Label_size(i))
                    index=sum(size_alpha(1:i-1))+(m-1)*(num_class-Label_size(i))+n;
                    Beta(k,i)=Beta(k,i)+c_value{1,k}(Label{i,1}(m),not_Label{i,1}(n))*Alpha(index);
                end
            end
        end
    end
    
%computing gradient(ikl)
    inner=zeros(num_class,num_training);
    for k=1:num_class
        for j=1:num_training
            inner(k,j)=Beta(k,:)*kernel(:,j);
        end
    end
    gradient=[];
    for i=1:num_training
        for m=1:Label_size(i)
            for n=1:(num_class-Label_size(i))
                temp=inner(Label{i,1}(m),i)-inner(not_Label{i,1}(n),i)-1;
                gradient=[gradient,temp];
            end
        end
    end
    
%Find Alpha_new    
    Aeq=zeros(num_class,sum(size_alpha));
    for k=1:num_class
        counter=0;
        for i=1:num_training
            for m=1:Label_size(i)
                for n=1:(num_class-Label_size(i))
                    counter=counter+1;
                    Aeq(k,counter)=c_value{1,k}(Label{i,1}(m),not_Label{i,1}(n));
                end
            end
        end
    end
    beq=zeros(num_class,1);
    LB=zeros(sum(size_alpha),1);
    UB=zeros(sum(size_alpha),1);
    counter=0;
    for i=1:num_training
        for m=1:Label_size(i)
            for n=1:(num_class-Label_size(i))
                counter=counter+1;
                UB(counter,1)=cost/(size_alpha(i));
            end
        end
    end
    Alpha_new=linprog(gradient',[],[],Aeq,beq,LB,UB);
    Alpha_new=Alpha_new';
    
%Find Lambda    
     Lambda=fminbnd(@neg_dual_func,0,1,optimset('Display','iter'),Alpha,Alpha_new,c_value,kernel,num_training,num_class,Label,not_Label,Label_size,size_alpha);


%Test convergence    
    if((abs(Lambda)<=lambda_tol)|(Lambda*sqrt(sum((Alpha_new-Alpha).^2))<=norm_tol))
        continuing=false;
        disp('program terminated normally');
    else
        if(iteration>=max_iter)
            continuing=false;
            warning('maximum number of iterations reached, procedure not convergent');
        else
            Alpha=Alpha+Lambda*(Alpha_new-Alpha);
        end
    end  
    toc;
end

Weights=Beta;

%Computing Bias
Left=[];
Right=[];
for i=1:num_training
    for m=1:Label_size(i)
        for n=1:(num_class-Label_size(i))
            index=sum(size_alpha(1:i-1))+(m-1)*(num_class-Label_size(i))+n;
            if((abs(Alpha(index))>=lambda_tol)&&(abs(Alpha(index)-(cost/(size_alpha(i))))>=lambda_tol))
                vector=zeros(1,num_class);
                vector(1,Label{i,1}(m))=1;
                vector(1,not_Label{i,1}(n))=-1;
                Left=[Left;vector];
                Right=[Right;-gradient(index)];
            end            
        end
    end
end
if (isempty(Left))
    Bias=sum(train_target');
else    
    Bias=(Left\Right)';
end

%Computing the size predictor using linear least squares model [2]
Left=zeros(num_training,num_class);
for i=1:num_training
    for k=1:num_class
        temp=Beta(k,:)*kernel(:,i);
        temp=temp+Bias(k);
        Left(i,k)=temp;
    end
end
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
        [~,false_neg]=size(setdiff(temp_notlabels,not_Label{i,1}));
        [~,false_pos]=size(setdiff(temp_labels,Label{i,1}));
        miss_class(1,j)=false_neg+false_pos;
    end
    [~,temp_index]=min(miss_class);
    Right(i,1)=candidate(1,temp_index);
end
Left=[Left,ones(num_training,1)];
tempvalue=(Left\Right)';
Weights_sizepre=tempvalue(1:num_class);
Bias_sizepre=tempvalue(num_class+1);