function[W,H]= normalize_factor(W,H)

[~,K]=size(W);

for i=1:K
    colnorm=norm(H(:,i),'fro');
    H(:,i)=H(:,i)/colnorm;
    W(:,i)=W(:,i)*colnorm;
end
