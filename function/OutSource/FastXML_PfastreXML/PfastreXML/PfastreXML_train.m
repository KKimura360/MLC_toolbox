function PfastreXML_train(trn_X_Xf, trn_X_Y, inv_prop, param, model_folder)

    addpath('../Tools');
    
    file_trn_X_Xf = 'tmp/tmp_trn_Xf.txt';
    write_text_mat(trn_X_Xf,file_trn_X_Xf);
    
    file_trn_X_Y ='tmp/tmp_trn_Y.txt';
    write_text_mat(trn_X_Y,file_trn_X_Y);

	file_inv_prop = 'tmp/inv_prop.txt';
	csvwrite(file_inv_prop,inv_prop);

	clear trn_X_Xf trn_X_Y;

    cmd = sprintf('PfastreXML_train %s %s %s %s %s', file_trn_X_Xf, file_trn_X_Y, file_inv_prop, model_folder, get_arguments(param));
    if isunix
       cmd = ['./' cmd]; 
    end
    
	system(cmd);
end

function args = get_arguments(param)
	args = ' ';

	if isfield(param,'pfswitch')
		args = sprintf(' %s -S %d',args,param.pfswitch);
	end
	
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

	if isfield(param,'gamma')
		args = sprintf(' %s -g %f',args,param.gamma);
	end

	if isfield(param,'alpha')
		args = sprintf(' %s -a %f',args,param.alpha);
	end

end
