function Prob=LEAD_Get_Cond_Marginal(data,causal_order,labels,classifiers,CPT)

    if(isempty(causal_order))
        error('Input error for parameter causal_order, please check again!');
    end
    
    num_class=length(labels);
    
    num_inst=size(data,1);
    
    num_nodes=length(causal_order);
    
    if(num_nodes==1)
        node=causal_order(1);
        if(labels(1,node)==-1)
            Prob=ones(num_inst,1);
        else
            pa=classifiers{1,node}.parent;
            n_pa=setdiff(1:num_class,pa);
            
            if(sum(labels(pa)==-1)==0)
                template=dec2bin(0,num_class);
                template(pa)=dec2bin(labels(pa));
                template(n_pa)=[];
                
                if(isempty(template))
                    idx=1;
                else
                    idx=bin2dec(template);
                    idx=idx+1;
                end
                                
                Prob_pos=CPT{node,1}(:,idx,1);
                Prob_neg=CPT{node,1}(:,idx,2);
            else
                error('error in conditional probability computation');
            end
            
            if(labels(1,node)==1)
                Prob=Prob_pos;
            else
                Prob=Prob_neg;
            end
        end
    else
        node=causal_order(1);
        causal_order(1)=[];
        if(labels(1,node)==-1)
            pa=classifiers{1,node}.parent;
            n_pa=setdiff(1:num_class,pa);
            
            if(sum(labels(pa)==-1)==0)
                template=dec2bin(0,num_class);
                template(pa)=dec2bin(labels(pa));
                template(n_pa)=[];
                
                if(isempty(template))
                    idx=1;
                else
                    idx=bin2dec(template);
                    idx=idx+1;
                end
                                
                Prob_pos=CPT{node,1}(:,idx,1);
                Prob_neg=CPT{node,1}(:,idx,2);
            else
                error('error in conditional probability computation');
            end
            
            labels1=labels;
            labels1(1,node)=1;
            Prob1=LEAD_Get_Cond_Marginal(data,causal_order,labels1,classifiers,CPT);
            
            labels2=labels;
            labels2(1,node)=0;
            Prob2=LEAD_Get_Cond_Marginal(data,causal_order,labels2,classifiers,CPT);
            
            Prob=Prob_pos.*Prob1+Prob_neg.*Prob2;
        else
            pa=classifiers{1,node}.parent;
            n_pa=setdiff(1:num_class,pa);
            
            if(sum(labels(pa)==-1)==0)
                template=dec2bin(0,num_class);
                template(pa)=dec2bin(labels(pa));
                template(n_pa)=[];
                
                if(isempty(template))
                    idx=1;
                else
                    idx=bin2dec(template);
                    idx=idx+1;
                end
                                
                Prob_pos=CPT{node,1}(:,idx,1);
                Prob_neg=CPT{node,1}(:,idx,2);                                
            else
                error('error in conditional probability computation');
            end
            
            if(labels(1,node)==1)
                labels1=labels;
                labels1(1,node)=1;
                Prob1=LEAD_Get_Cond_Marginal(data,causal_order,labels1,classifiers,CPT);
                Prob=Prob_pos.*Prob1;
            else
                labels2=labels;
                labels2(1,node)=0;
                Prob2=LEAD_Get_Cond_Marginal(data,causal_order,labels2,classifiers,CPT);
                Prob=Prob_neg.*Prob2;
            end
        end
    end