function output=neg_dual_func(Lambda,Alpha_old,Alpha_new,c_value,kernel,num_training,num_class,Label,not_Label,Label_size,size_alpha);

    Alpha=Alpha_old+Lambda*(Alpha_new-Alpha_old);
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
    
    output=0;
    for k=1:num_class
        output=output+Beta(k,:)*(Beta(k,:)*kernel')';
    end
    output=0.5*output;
    output=output-sum(Alpha);