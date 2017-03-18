function score_mat = PfastreXML_test(tst_X_Xf, param, model_folder)

    addpath('../Tools');
    
    file_tst_X_Xf = 'tmp/tmp_tst_Xf.text';
    write_text_mat(tst_X_Xf,file_tst_X_Xf);
    
    file_score_mat = 'tmp/score_mat.txt';
   
	clear tst_X_Xf;
 
    cmd = sprintf('PfastreXML_test %s %s %s %s', file_tst_X_Xf, file_score_mat, model_folder, get_arguments(param));
    if isunix
        cmd = ['./' cmd];
    end
    
	system(cmd);

    score_mat = read_text_mat(file_score_mat);
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

	if isfield(param,'actlbl')
		args = sprintf(' %s -n %d',args,param.actlbl);
	end

end

