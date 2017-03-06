function [assign,Dist,Clsdist,Clsind,bL,rankDist,rankInd]=assignLabels(K,target,assign,Dist,Clsdist,Clsind,bL,rankDist,rankInd)
% sort distances 

% assign step
    while(1)
      % a given target label, first, we choose the closest cluster first
      %Cls is the index of the closest cluster
      Cls=rankInd(target,1);
      %Each Clsuter has a ranking and distance of labels
      %tmprank is the rank of target in the Cls-th cluster 
      tmprank=sum(Clsdist{Cls}<rankDist(target,1));
      % if the ranking is smaller than maximum size of clusters (BL)  
      if (tmprank<=bL)
      % if this label has already been assigned before,
        if ~assign(target)==0
          
          % obtain index of the last cluster this target label was assigned to  
          tmpCls=assign(target);
          
          %delete information from tmpCls-th cluster 
          %find the ranking of this label in the clutser
          tmpind=find(Clsind{tmpCls}==target);
          Clsdist{tmpCls}(tmpind)=[];
          Clsind{tmpCls}(tmpind)=[];
        end
        % assign label to the cluster
        assign(target)=Cls;
        % add the target label to the cluster
        %NOTE: Cls dist is not sorted yet, just add the info to the end of
        %ranking
        Clsdist{Cls}=[Clsdist{Cls}, rankDist(target,1)];
        Clsind{Cls}=[Clsind{Cls},target];
        % Check the total number of labels in the cluster    
        totalCls=length(Clsind{Cls});
            %if the total number is more than the maximum size (bL)
            % we need to kick out the farest label  
            if (totalCls>bL)
                %find the farest label
                [~,tmpind]=max(Clsdist{Cls});
                %treat the farest label as new target to be assigned 
                target=Clsind{Cls}(tmpind);
                %delete info of this label from the cluster
                Clsdist{Cls}(tmpind)=[];
                Clsind{Cls}(tmpind)=[];
                %set the distance from this farest label to this cluster as
                %infinity never to be assigned again
                Dist(target,Cls)=inf;
                %update ranking of this cluster
                %NOTE: This code is not efficient because we
                %don't use an insert sorting to do these procedure
                rankDist(target,:)=circshift(rankDist(target,:),[0,-1]);
                rankInd(target,:)=circshift(rankInd(target,:),[0,-1]);
                %[rankDist(target,:),rankInd(target,:)]=sort(Dist(target,:));
            else
                break;
            end
      %if this label will not be added to the clutser (not top-BL label)   
      else
       %set the distance from this farest label to this cluster as
       %infinity never to be assigned again
       Dist(target,Cls)=Inf;
       rankDist(target,:)=circshift(rankDist(target,:),[0,-1]);
       rankInd(target,:)=circshift(rankInd(target,:),[0,-1]);
       %[rankDist(target,:),rankInd(target,:)]=sort(Dist(target,:));
       continue;
      end
      break;
    end