function [accuracy,ins_f,label_f,ham,zero_one] = evaluate(p,y)
    e = 0.0000001;
    m = size(y,1);
    tp = (p == y & p > e);

    %disp('p_y')
    %disp(p(1:3,1:3))
    %disp(y(1:3,1:3))
    
    %%caculate accuracy
    sum_tp = sum(tp,2);
    sum_py = sum(p > 0 | y > 0,2);
    addone  = sum_py == 0;
    sum_tp = sum_tp + addone;
    sum_py = sum_py + addone;
    accuracy = sum(sum_tp./sum_py)/m;

    %%macro_f
    label_f = 0.0;
    for i=1:size(y,2)
		tp1 = tp(:,i) > e;
		p1  = p(:,i)  > e;
		y1  = y(:,i)  > e;
		n_tp = 1.0*sum(tp1,1);
		n_p  = 1.0*sum(p1,1);
		n_y  = 1.0*sum(y1,1);
		%disp([num2str(n_tp) ' ' num2str(n_p) ' ' num2str(n_y)]);
		if n_p+n_y > 0
			label_f = label_f+ 2*n_tp/(n_p+n_y);
		end
    end
    label_f = label_f/size(y,2);
	
    %%ins_f
    ins_f = 0.0;
    for i = 1:size(y,1)
        tp1 = tp(i,:) > e;
        p1  = p(i,:)  > e;
        y1  = y(i,:)  > e;
        n_tp = 1.0*sum(tp1,2);
        n_p  = 1.0*sum(p1,2);
        n_y  = 1.0*sum(y1,2);
        %disp([num2str(n_tp) ' ' num2str(n_p) ' ' num2str(n_y)]);
        if n_p+n_y > 0
            ins_f = ins_f+ 2*n_tp/(n_p+n_y);
        end
    end
    ins_f = ins_f / size(y,1);

	%%ham
    ham = sum(sum(p ~= y)) / size(p,1) / size(p,2);

	%%zero_one
	a = sum((p-y)==0,2) == size(y,2);
	zero_one = 1 - sum(a,1)/size(y,1);

	disp(['acc:' num2str(accuracy) '|ins_f:' num2str(ins_f) '|label_f:' num2str(label_f) '|ham:' num2str(ham) '|zero_one:' num2str(zero_one)]);
	
end
