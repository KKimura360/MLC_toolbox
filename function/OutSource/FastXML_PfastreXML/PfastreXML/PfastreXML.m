function PfastreXML( trn_X_Xf, trn_X_Y, tst_X_Xf, tst_X_Y, inv_prop, tune, tune_metric, MCNAME, param, res_dir )

	model_dir = [res_dir '/model'];
	system(['mkdir -p ' model_dir]);

	diary([res_dir '/log.txt']);
	
	if tune
		[cv_split, cv_trn_X_Xf, cv_trn_X_Y, cv_tst_X_Xf, cv_tst_X_Y] = get_std_xmlc_tune_data( trn_X_Xf, trn_X_Y );

		cv_param.log_loss_coeff = 2.^[-5:5];
		cvparam.alpha = [0:0.1:1.0];
		cv_param.max_leaf = [5 10 20 50];

		cv_res_dir = tempname;
		system(['mkdir -p ' cv_res_dir]);

		func_hdl = str2func('PfastreXML');
		func_data = { cv_trn_X_Xf, cv_trn_X_Y, cv_tst_X_Xf, cv_tst_X_Y, inv_prop, false, tune_metric, MCNAME };

		metric_hdl = str2func(tune_metric);
		metric_data = get_tune_metric_data( cv_tst_X_Y, inv_prop, tune_metric );

		param = sequential_search( func_hdl, func_data, metric_hdl, metric_data, param, cv_param, cv_res_dir );
		clear mex;
	end

	save([res_dir '/param.mat'], 'param', '-v7.3');

	PfastreXML_train( trn_X_Xf, trn_X_Y, inv_prop, param, model_dir, MCNAME );
	clear mex;

	score_mat = PfastreXML_test( tst_X_Xf, param, res_dir, MCNAME );
	clear mex;
	save([res_dir '/score_mat.mat'], 'score_mat', '-v7.3');
	%load([res_dir '/score_mat.mat']);
	
	metrics = get_all_metrics( score_mat, tst_X_Y, inv_prop );
	clear mex;
	save([res_dir '/metrics.mat'], 'metrics', '-v7.3');

	diary off;
end 
