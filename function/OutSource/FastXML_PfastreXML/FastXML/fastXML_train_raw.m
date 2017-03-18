function fastXML_train_raw(trn_X_Xf, trn_X_Y, param,X_name,Y_name,model_name)

    write_text_mat(trn_X_Xf,X_name);    
    write_text_mat(trn_X_Y,Y_name);
	clear trn_X_Xf trn_X_Y;
 
    cmd = sprintf('fastXML_train %s %s %s %s', X_name, Y_name, model_name, get_arguments(param));
    cmd
    if isunix
       cmd = ['./' cmd]; 
    end
    
	system(cmd);
end

function args = get_arguments(param)
	args = ' ';
	
	if isfield(param,'num_thread')
		args = sprintf(' %s -T %d',args,param.num_thread);
	end

	if isfield(param,'start_tree')
		args = sprintf(' %s -s %d',args,param.start_tree);
	end

	if isfield(param,'num_tree')
		args = sprintf(' %s -t %d',args,param.num_tree);
	end

	if isfield(param,'bias')
		args = sprintf(' %s -b %f',args,param.bias);
	end

	if isfield(param,'log_loss_coeff')
		args = sprintf(' %s -c %f',args,param.log_loss_coeff);
	end

	if isfield(param,'max_leaf')
		args = sprintf(' %s -m %d',args,param.max_leaf);
	end

	if isfield(param,'lbl_per_leaf')
		args = sprintf(' %s -l %d',args,param.lbl_per_leaf);
	end

end
