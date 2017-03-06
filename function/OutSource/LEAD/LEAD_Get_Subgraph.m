function Subgraph=LEAD_Get_Subgraph(nodes,classifiers)

    if(isempty(nodes))
        Subgraph=[];
        return;
    end
    
    num_nodes=length(nodes);
    
    ancestors=cell(1,num_nodes);
    
    for i=1:num_nodes
        pa=classifiers{1,nodes(i)}.parent;
        if(isempty(pa))
            ancestors{1,i}=[];
        else
            ancestors{1,i}=LEAD_Get_Subgraph(pa,classifiers);
        end
    end
    
    Subgraph=[];
    
    for i=1:num_nodes;
        Subgraph=union(Subgraph,ancestors{1,i});
    end
    
    Subgraph=union(Subgraph,nodes);
    
    Subgraph=sort(Subgraph);