function new_net=update_net_ml(net,instance,label,alpha,Cost,in,out)
%update_net_ml updates the weights and biases of a multi-label neural network
%
%    Syntax
%
%       new_net=update_net_ml(net,instance,label,alpha,Cost,in,out)
%
%    Description
%
%       update_net_ml takes,
%           net          - An input multi-label neural network
%           instance     - An Nx1 input vector
%           label        - A Qx1 vector, if the ith class belongs to the input instance, then label(i)=+1, otherwise label(i)=-1
%           alpha        - Learning rate for updating weights and biases
%           Cost         - Cost parameter for each instance, used for regularization
%           in           - The activation function used for the hidden neurons, either 'logsig' or 'tansig'
%           out          - The activation function used for the output neurons, either 'logsig' or 'tansig'
%      and returns,
%           new_net      - The updated neural network
        
        [num_class,tempvalue]=size(label);
        [num_hidden,tempvalue]=size(net.IW{1});
        Labels=[];
        not_Labels=[];
        for i=1:num_class
            if(label(i)==1)
                Labels=[Labels,i];
            else
                not_Labels=[not_Labels,i];
            end
        end
        [tempvalue,num_labels]=size(Labels);
        [tempvalue,num_notlabels]=size(not_Labels);
        
        if((num_labels~=0)&(num_notlabels~=0))
            if(strcmp(in,'logsig'))
                b=logsig(net.IW{1}*instance+net.b{1});
            else
                b=tansig(net.IW{1}*instance+net.b{1});
            end
            c=sim(net,instance);

            d=zeros(1,num_class);
            for j=1:num_class
                if(ismember(j,Labels))
                    temp=0;
                    for n=1:num_notlabels
                        temp=temp+exp(-(c(j)-c(not_Labels(n))));
                    end
                else
                    temp=0;
                    for m=1:num_labels
                        temp=temp-exp(-(c(Labels(m))-c(j)));
                    end
                end
                if(strcmp(out,'logsig'))
                    d(j)=temp*c(j)*(1-c(j));
                else
                    d(j)=temp*(1+c(j))*(1-c(j));
                end
            end
            d=d*(1/(num_labels*num_notlabels));

            e=zeros(1,num_hidden);
            for i=1:num_hidden
                if(strcmp(in,'logsig'))
                    e(i)=b(i)*(1-b(i))*(d*net.LW{2,1}(:,i));
                else
                    e(i)=(1+b(i))*(1-b(i))*(d*net.LW{2,1}(:,i));
                end
            end

            update_w=(alpha*(b*d))'-Cost*net.LW{2,1};
            update_v=(alpha*(instance*e))'-Cost*net.IW{1};
            update_b2=alpha*d'-Cost*net.b{2};
            update_b1=alpha*e'-Cost*net.b{1};

            net.IW{1}=net.IW{1}+update_v;
            net.LW{2,1}=net.LW{2,1}+update_w;
            net.b{1}=net.b{1}+update_b1;
            net.b{2}=net.b{2}+update_b2;
        end
        new_net=net;