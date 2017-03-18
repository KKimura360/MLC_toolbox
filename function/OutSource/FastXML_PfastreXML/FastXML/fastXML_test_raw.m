function score_mat = fastXML_test_raw(tst_X_Xf, param,Xt_name,Yt_name,model_name)

    
    write_text_mat(tst_X_Xf,Xt_name);
    
       
	clear tst_X_Xf;
   
    cmd = sprintf('fastXML_test %s %s %s %s', Xt_name, Yt_name, model_name, get_arguments(param));
    cmd
    if isunix
        cmd = ['./' cmd];
    end
    
	system(cmd);

    score_mat = read_text_mat(Yt_name);

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
end

