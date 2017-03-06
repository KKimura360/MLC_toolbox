function Prob=LEAD_Get_Prob(data,nodes,labels,order,classifiers,psize,Outputs)

    if(isempty(nodes))
        error('Input error for parameter nodes, please check again!');
    end
    
    num_class=length(labels);
    num_inst=size(data,1);
    
    nodes=sort(nodes);
    
    Subgraph=LEAD_Get_Subgraph(nodes,classifiers);
    
    sub_order=order(Subgraph);
    
    [sorted_sub_order,idx]=sort(sub_order);
    
    Subgraph_causal_order=Subgraph(idx);
    
    %Computing conditional probability table (CPT) for each node in the subgraph
    
    subgraph_size=length(Subgraph);        

    if(subgraph_size<=psize)
        CPT=cell(num_class,1);
    
        for i=1:subgraph_size
            cur_node=Subgraph_causal_order(i);
            pa=classifiers{1,cur_node}.parent;
            n_pa=setdiff(1:num_class,pa);

            num_pa=length(pa);

            if(num_pa==0)
                CPT{cur_node,1}=zeros(num_inst,1,2);

                testing_instance_matrix=data;
                testing_label_vector=ones(num_inst,1);
                if(~isempty(classifiers{1,cur_node}.model.SVs))
                    [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,classifiers{1,cur_node}.model,'-b 1');
                    pos_index=find(classifiers{1,cur_node}.model.Label);
                    Prob_pos=prob_estimates(:,pos_index);
                else
                    majority_label=classifiers{1,cur_node}.model.majority_label;
                    if(majority_label==1)
                        Prob_pos=ones(num_inst,1);
                    else
                        Prob_pos=zeros(num_inst,1);
                    end
                end
                Prob_neg=1-Prob_pos;

                CPT{cur_node,1}(:,1,1)=Prob_pos;
                CPT{cur_node,1}(:,1,2)=Prob_neg;
            else
                idx=find(labels(pa)==-1);
                pa_var=pa(idx);
                pa_fix=setdiff(pa,pa_var);

                CPT{cur_node,1}=zeros(num_inst,2^num_pa,2);

                num_pa_var=length(pa_var);

                for pointer=1:2^num_pa_var
                    template=dec2bin(0,num_class);
                    template(pa_fix)=dec2bin(labels(pa_fix));
                    template(pa_var)=dec2bin(pointer-1,num_pa_var);
                    template(n_pa)=[];

                    idx=bin2dec(template);
                    idx=idx+1;

                    testing_instance_matrix=data;
                    for j=1:length(template)
                        testing_instance_matrix=[testing_instance_matrix,str2double(template(j))*ones(num_inst,1)];
                    end
                    testing_label_vector=ones(num_inst,1);
                    if(~isempty(classifiers{1,cur_node}.model.SVs))
                        [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,classifiers{1,cur_node}.model,'-b 1');
                        pos_index=find(classifiers{1,cur_node}.model.Label);
                        Prob_pos=prob_estimates(:,pos_index);
                    else
                        majority_label=classifiers{1,cur_node}.model.majority_label;
                        if(majority_label==1)
                            Prob_pos=ones(num_inst,1);
                        else
                            Prob_pos=zeros(num_inst,1);
                        end
                    end
                    Prob_neg=1-Prob_pos;

                    CPT{cur_node,1}(:,idx,1)=Prob_pos;
                    CPT{cur_node,1}(:,idx,2)=Prob_neg;
                end
            end
        end       
    
        Prob=LEAD_Get_Cond_Marginal(data,Subgraph_causal_order,labels,classifiers,CPT);
    else                
        Prob=ones(num_inst,1);
        
        Indexes=find(labels==1);
        
        for i=1:length(Indexes)
            cur_node=Indexes(i);
            
            pa=classifiers{1,cur_node}.parent;
            pa=sort(pa);
            num_pa=length(pa);
            Prior_pos=Outputs(pa,:)';
            Prior_neg=1-Prior_pos;
            
            Prob_pos=zeros(num_inst,1);
            
            for pointer=1:2^num_pa
                template=dec2bin(pointer-1,num_pa);
                vec=zeros(1,num_pa);
                Prior=ones(num_inst,1);
                for j=1:num_pa
                    vec(1,j)=str2double(template(j));
                    if(vec(1,j)==1)
                        Prior=Prior.*Prior_pos(:,j);
                    else
                        Prior=Prior.*Prior_neg(:,j);
                    end
                end
                pos_idx=find(vec);
                neg_idx=setdiff([1:num_pa],pos_idx);                
                
                testing_instance_matrix=data;
                aux_mat=concur(vec',num_inst)';
                testing_instance_matrix=[testing_instance_matrix,aux_mat];
                testing_label_vector=ones(num_inst,1);
                if(~isempty(classifiers{1,cur_node}.model.SVs))
                    [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,classifiers{1,cur_node}.model,'-b 1');
                    pos_index=find(classifiers{1,cur_node}.model.Label);
                    Prob_pos=Prob_pos+Prior.*prob_estimates(:,pos_index);
                else
                    majority_label=classifiers{1,cur_node}.model.majority_label;
                    if(majority_label==1)
                        Prob_pos=Prob_pos+Prior.*ones(num_inst,1);
                    else
                        Prob_pos=Prob_pos+Prior.*zeros(num_inst,1);
                    end
                end
            end
            Prob=Prob.*Prob_pos;
        end
    end