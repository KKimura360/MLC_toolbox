#include "fastXML.h"
#include <ctime>

using namespace std;

LOGLVL loglvl = LOGLVL::PROGRESS;  // print progress reports
mutex mtx;	// used to synchronize aggregation of individual tree scores during prediction time
_bool USE_IDCG = true; // if true, optimizes for nDCG; otherwise, optimized for DCG

/* reusable general data containers */ 
#define Icount 3
thread_local VecI I[Icount];

#define Bcount 1
thread_local VecB B[Bcount];

#define Fcount 3
thread_local VecF F[Fcount];

#define Dcount 10
thread_local VecD D[Dcount];

#define IFcount 2
thread_local VecIF IF[IFcount];

thread_local VecF wt_vec;   // stores nDCG weights
thread_local VecF idcg;		// stores ideal DCG values of training points necessary for nDCG calculation

thread_local pairIF** ftptr; // ftptr and ftdata store inverted feature matrix for training instances in the current node (node being split). Instance and feature indices are shrunk to a continuous range
thread_local pairIF* ftdata;
thread_local pairIF** lblptr; // lblptr and lbldata store label matrix for training instances in the current node. Instance and label indices are shrunk to a continuous range
thread_local pairIF* lbldata;
thread_local VecI fts; // active feature indices in the current node
thread_local VecI lbls; // active label indices in the current node

thread_local mt19937 reng; // random number generator used during training 

void setup(_int num_inst, _int num_ft, _int num_lbl)
{
	/* initialize general data containers */

	_int max_size = max(max(num_inst+1,num_ft+1), num_lbl+1);

	for(_int i=0; i<Icount; i++)
		I[i].resize(max_size,0);

	for(_int i=0; i<Bcount; i++)
		B[i].resize(max_size,0);

	for(_int i=0; i<Fcount; i++)
		F[i].resize(max_size,0);

	for(_int i=0; i<Dcount; i++)
		D[i].resize(max_size,0);

	for(_int i=0; i<IFcount; i++)
		IF[i].resize(max_size,make_pair(0,0));

	wt_vec.resize(max_size,0);
	idcg.resize(max_size,0);
}

void cleanup()
{

}

void trn_setup(SMatF* trn_ft_mat, SMatF* trn_lbl_mat)
{
	_int num_trn = trn_ft_mat->nc;
	_int num_ft = trn_ft_mat->nr;
	_int num_lbl = trn_lbl_mat->nr;

	/* initialize ftptr & ftdata */
	ftptr = new pairIF*[num_ft+2];
	_int totsiz = 0;
	for(_int i=0; i<num_trn; i++)
		totsiz += trn_ft_mat->size[i];
	totsiz += num_trn;
	ftdata = new pairIF[totsiz];

	/* initialize lblptr & lbldata */
	lblptr = new pairIF*[num_trn+1];
	totsiz = 0;
	for(_int i=0; i<num_trn; i++)
		totsiz += trn_lbl_mat->size[i];
	lbldata = new pairIF[totsiz];

	/* initialize wt_vec and idcg */
	for(_int i=0; i<num_lbl; i++)
		wt_vec[i] = 1/log2(i+2);

	_int* siz = trn_lbl_mat->size;
	pairIF** data = trn_lbl_mat->data;

	for(_int i=0; i<num_trn; i++)
	{
		VecIF vec;
		for(_int j=0; j<siz[i]; j++)
			vec.push_back(make_pair(data[i][j].first,data[i][j].second));
		sort(vec.begin(),vec.end(),comp_pair_by_second_desc<_int,_float>);
				
		_float val = 0;
		for(_int j=0; j<vec.size(); j++)
			val += vec[j].second*wt_vec[j];

		if(USE_IDCG)
		{
			if(val==0)
				idcg[i] = 1.0;
			else
				idcg[i] = 1.0/val;
		}
		else
			idcg[i] = 1.0;
	}
}

/* training cleanup */
void trn_cleanup()
{
	delete [] ftptr;
	delete [] ftdata;
	delete [] lblptr;
	delete [] lbldata;
}

pairII get_pos_neg_count(VecI& pos_or_neg)
{
	pairII counts = make_pair(0,0);
	for(_int i=0; i<pos_or_neg.size(); i++)
	{
		if(pos_or_neg[i]==+1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

void active_dims(SMatF* mat, VecI& insts, VecI& dims, VecI& maps, VecI& counts)
{
	dims.clear();
	_int num_trn = mat->nc;
	_int num_dim = mat->nr;
	_int* size = mat->size;
	pairIF** data = mat->data;

	for(_int i=0; i<insts.size(); i++)
	{
		_int inst = insts[i];
		for(_int j=0; j<size[inst]; j++)
		{
			_int dim = data[inst][j].first;
			if(!counts[dim])
				dims.push_back(dim);
			counts[dim]++;
		}
	}

	sort(dims.begin(),dims.end());

	for(_int i=0; i<dims.size(); i++)
		maps[dims[i]] = i;
}

void shrink_ft_mat(SMatF* trn_ft_mat, VecI& insts, _float bias)
{
	_int num_ft = trn_ft_mat->nr;
	_int* size = trn_ft_mat->size;
	pairIF** data = trn_ft_mat->data;

	VecI& counts = I[0];
	VecI& maps = I[1];

	active_dims(trn_ft_mat, insts, fts, maps, counts);
	fts.push_back(num_ft);
	maps[num_ft] = fts.size()-1;
	counts[num_ft] = insts.size();

	ftptr[0] = ftdata;
	for(_int i=0; i<fts.size(); i++)
	{
		_int ft = fts[i];
		ftptr[i+1] = ftptr[i]+counts[ft];
		counts[ft] = 0;
	}

	for(_int i=0; i<insts.size(); i++)
	{	
		_int inst = insts[i];
		for(_int j=0; j<size[inst]; j++)
		{
			_int ft = data[inst][j].first;
			_int mapft = maps[ft];
			_float val = data[inst][j].second;

			ftptr[mapft][counts[ft]++] = make_pair(i,val);
		}
		_int mapft = fts.size()-1;
		ftptr[mapft][counts[num_ft]++] = make_pair(i,bias);
	}

	for(_int i=0; i<fts.size(); i++)
	{
		counts[fts[i]] = 0;
		maps[fts[i]] = 0;
	}
}

void shrink_lbl_mat(SMatF* trn_lbl_mat, VecI& insts)
{
	_int* size = trn_lbl_mat->size;
	pairIF** data = trn_lbl_mat->data;

	VecI& counts = I[0];
	VecI& maps = I[1];
	active_dims(trn_lbl_mat, insts, lbls, maps, counts);
	
	lblptr[0] = lbldata;
	for(_int i=0; i<insts.size(); i++)
		lblptr[i+1] = lblptr[i] + size[insts[i]];

	for(_int i=0; i<insts.size(); i++)
	{	
		_int inst = insts[i];
		for(_int j=0; j<size[inst]; j++)
		{
			_int lbl = maps[data[inst][j].first];
			_float val = data[inst][j].second;
			lblptr[i][j] = make_pair(lbl,val);
		}
	}

	for(_int i=0; i<lbls.size(); i++)
	{
		counts[lbls[i]] = 0;
		maps[lbls[i]] = 0;
	}
}

void test_svm(VecI& insts, SMatF* ft_mat, VecIF& w, _float bias, VecI& pos_or_neg)
{
	pos_or_neg.resize(insts.size());

	VecF& dense_w = F[0];
	_int num_ft = ft_mat->nr;

	for(_int i=0; i<w.size(); i++)
		dense_w[w[i].first] = w[i].second;

	_int* siz = ft_mat->size;
	pairIF** data = ft_mat->data;

	for(_int i=0; i<insts.size(); i++)
	{
		_int inst = insts[i];
		_float prod = bias*dense_w[num_ft];

		for(_int j=0; j<siz[inst]; j++)
		{
			_int ft = data[inst][j].first;
			_float val = data[inst][j].second;
			prod += val*dense_w[ft];
		}

		if(prod>=0)
			pos_or_neg[i] = 1;
		else
			pos_or_neg[i] = -1;
	}	

	for(_int i=0; i<w.size(); i++)
		dense_w[w[i].first] = 0;
}

#define GETI(i) (y[i]+1)
typedef signed char schar;
bool optimize_log_loss(VecI& insts, SMatF* trn_ft_mat, _float log_loss_coeff, _float bias, VecI& y, VecIF& sparse_w)
{
	_int node_trn = insts.size();
	_int node_ft = fts.size();

	pairII num_pos_neg = get_pos_neg_count(y);
	_float frac_pos = (_float)num_pos_neg.first/(num_pos_neg.first+num_pos_neg.second);
	_float frac_neg = (_float)num_pos_neg.second/(num_pos_neg.first+num_pos_neg.second);

	_double Cp = log_loss_coeff/frac_pos;
	_double Cn = log_loss_coeff/frac_neg;  // unequal Cp,Cn improves the balancing in some data sets

	VecD& w = D[0];
	_double eps = 0.01;

	_int l = node_trn;
	_int w_size = node_ft;
	_int newton_iter=0, iter=0;
	_int max_newton_iter = 10;
	_int max_iter = 10;
	_int max_num_linesearch = 20;
	_int active_size;
	_int QP_active_size;

	_double nu = 1e-12;
	_double inner_eps = 1;
	_double sigma = 0.01;
	_double w_norm, w_norm_new;
	_double z, G, H;
	_double Gnorm1_init;
	_double Gmax_old = INF;
	_double Gmax_new, Gnorm1_new;
	_double QP_Gmax_old = INF;
	_double QP_Gmax_new, QP_Gnorm1_new;
	_double delta, negsum_xTd, cond;

	VecI& index = I[0];
	VecD& Hdiag = D[1];
	VecD& Grad = D[2];
	VecD& wpd = D[3];
	VecD& xjneg_sum = D[4];
	VecD& xTd = D[5];
	VecD& exp_wTx = D[6];
	VecD& exp_wTx_new = D[7];
	VecD& tau = D[8];
	VecD& D1 = D[9];

	_double C[3] = {Cn,0,Cp};
	
	w_norm = 0;
	for(_int i=0; i<w_size; i++)
	{
		index[i] = i;

		for(pairIF* dat = ftptr[i]; dat < ftptr[i+1]; dat++)
		{
			_int inst = dat->first;
			_float val = dat->second;

			if(y[inst] == -1)
				xjneg_sum[i] += C[GETI(inst)]*val;
		}
	}

	for(_int i=0; i<l; i++)
	{
		exp_wTx[i] = exp(exp_wTx[i]);
		_double tau_tmp = 1/(1+exp_wTx[i]);
		tau[i] = C[GETI(i)]*tau_tmp;
		D1[i] = C[GETI(i)]*exp_wTx[i]*SQ(tau_tmp);
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(_int s=0; s<active_size; s++)
		{
			_int i = index[s];
			Hdiag[i] = nu;

			_double tmp = 0;
			pairIF* dat = ftptr[i];
		
			for(pairIF* dat = ftptr[i]; dat < ftptr[i+1]; dat++)
			{
				_int inst = dat->first;
				_float val = dat->second;

				Hdiag[i] += SQ(val)*D1[inst];
				tmp += val*tau[inst];
			}

			Grad[i] = -tmp + xjneg_sum[i];

			_double Gp = Grad[i]+1;
			_double Gn = Grad[i]-1;
			_double violation = 0;

			if(w[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(_int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(_int i=0; i<QP_active_size; i++)
			{
				_llint r = reng();
				_int j = i+r%(QP_active_size-i);
				swap(index[j], index[i]);
			}

			for(_int s=0; s<QP_active_size; s++)
			{
				_int i = index[s];
				H = Hdiag[i];

				G = Grad[i] + (wpd[i]-w[i])*nu;
				for(pairIF* dat = ftptr[i]; dat < ftptr[i+1]; dat++)
				{
					_int inst = dat->first;
					_float val = dat->second;
					G += val*D1[inst]*xTd[inst];
				}

				_double Gp = G+1;
				_double Gn = G-1;
				_double violation = 0;
				if(wpd[i] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[i] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[i])
					z = -Gp/H;
				else if(Gn > H*wpd[i])
					z = -Gn/H;
				else
					z = -wpd[i];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[i] += z;

				for(pairIF* dat = ftptr[i]; dat < ftptr[i+1]; dat++)
				{
					_int inst = dat->first;
					_float val = dat->second;
					xTd[inst] += val*z;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		delta = 0;
		w_norm_new = 0;
		for(_int i=0; i<w_size; i++)
		{
			delta += Grad[i]*(wpd[i]-w[i]);
			if(wpd[i] != 0)
				w_norm_new += fabs(wpd[i]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(_int i=0; i<l; i++)
		{
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];
		}

		_int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			_double cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(_int i=0; i<l; i++)
			{
				_double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(_int i=0; i<w_size; i++)
					w[i] = wpd[i];

				for(_int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					_double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D1[i] = C[GETI(i)]*exp_wTx[i]*SQ(tau_tmp);
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(_int i=0; i<w_size; i++)
				{
					wpd[i] = (w[i]+wpd[i])*0.5;

					if(wpd[i] != 0)
						w_norm_new += fabs(wpd[i]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(_int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(_int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(_int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;

				for(pairIF* dat = ftptr[i]; dat < ftptr[i+1]; dat++)
				{
					_int inst = dat->first;
					_float val = dat->second;
					exp_wTx[inst] += w[i]*val;
				}
			}

			for(_int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;
	}

	_float th = 1e-16;
	for(_int i=0; i<w_size; i++)
	{
		if(fabs(w[i])>th)
			sparse_w.push_back(make_pair(fts[i],w[i]));
	}

	for(_int i=0; i<fts.size(); i++)
	{
		index[i] = 0;
		w[i] = 0;
		Hdiag[i] = 0;
		Grad[i] = 0;
		wpd[i] = 0;
		xjneg_sum[i] = 0;
	}

	for(_int i=0; i<insts.size(); i++)
	{
		xTd[i] = 0;
		exp_wTx[i] = 0;
		exp_wTx_new[i] = 0;
		tau[i] = 0;
		D1[i] = 0;
	}

	test_svm(insts, trn_ft_mat, sparse_w, bias, y);

	num_pos_neg = get_pos_neg_count(y);

	if(num_pos_neg.first==0 || num_pos_neg.second==0)
	{
		sparse_w.clear();
		return false;
	}

	return true;
}

void calc_leaf_prob(Node* node, _int lbl_per_leaf)
{
	VecI& insts = node->insts;
	VecIF& leaf_dist = node->leaf_dist;
	leaf_dist.resize(lbls.size());

	for(_int i=0; i<lbls.size(); i++)
		leaf_dist[i] = make_pair(lbls[i],0);

	for(_int i=0; i<insts.size(); i++)
	{
		for(pairIF* dat = lblptr[i]; dat < lblptr[i+1]; dat++)
		{
			_int lbl = dat->first;
			_float val = dat->second;
			leaf_dist[lbl].second += val;
		}
	}	

	for(_int i=0; i<lbls.size(); i++)
		leaf_dist[i].second /= insts.size();

	sort(leaf_dist.begin(), leaf_dist.end(), comp_pair_by_second_desc<_int,_float>);
	if(leaf_dist.size()>lbl_per_leaf)
		leaf_dist.resize(lbl_per_leaf);
	sort(leaf_dist.begin(), leaf_dist.end(), comp_pair_by_first<_int,_float>);
}

bool optimize_ndcg(VecI& insts, VecI& pos_or_neg)
{
	_int num_trn = insts.size();
	pairII num_pos_neg = get_pos_neg_count(pos_or_neg);

	_float eps = 1e-6;

	VecIF& pos_sum = IF[0];
	VecIF& neg_sum = IF[1];
	VecF& diff_vec = F[0];
	VecF& idcg_vec = F[1];

	for(_int i=0; i<insts.size(); i++)
		idcg_vec[i] = idcg[insts[i]];

	_float ndcg=-1, new_ndcg=0;

	while(true)
	{
		for(_int i=0; i<lbls.size(); i++)
		{
			pos_sum[i] = make_pair(i,0);
			neg_sum[i] = make_pair(i,0);
			diff_vec[i] = 0;
		}

		for(_int i=0; i<insts.size(); i++)
		{
			for(pairIF* dat = lblptr[i]; dat < lblptr[i+1]; dat++)
			{
				_int lbl = dat->first;
				_float val = dat->second * idcg_vec[i];

				if(pos_or_neg[i]==+1)
					pos_sum[lbl].second += val;
				else
					neg_sum[lbl].second += val;
			}
		}

		new_ndcg = 0;
		for(_int s=-1; s<=1; s+=2)
		{
			VecIF& sum = s==-1 ? neg_sum : pos_sum;
			sort(sum.begin(), sum.begin()+lbls.size(), comp_pair_by_second_desc<_int,_float>);

			for(_int i=0; i<lbls.size(); i++)
			{
				_int lbl = sum[i].first;
				_float val = sum[i].second;
				diff_vec[lbl] += s*wt_vec[i];
				new_ndcg += wt_vec[i]*val;
			}
		}

		new_ndcg /= num_trn;

		for(_int i=0; i<insts.size(); i++)
		{
			_float gain_diff = 0;
			for(pairIF* dat = lblptr[i]; dat < lblptr[i+1]; dat++)
			{
				_int lbl = dat->first;
				_float val = dat->second * idcg_vec[i];
				gain_diff += val*diff_vec[lbl];
			}

			if(gain_diff>0)
				pos_or_neg[i] = +1;
			else if(gain_diff<0)
				pos_or_neg[i] = -1;
		}
		
		if(new_ndcg-ndcg<eps)
			break;
		else
			ndcg = new_ndcg;
	}

	num_pos_neg = get_pos_neg_count(pos_or_neg);

	for(_int i=0; i<lbls.size(); i++)
	{
		pos_sum[i] = make_pair(0,0);
		neg_sum[i] = make_pair(0,0);
		diff_vec[i] = 0;
	}

	for(_int i=0; i<insts.size(); i++)
		idcg_vec[i] = 0;

	if(num_pos_neg.first==0 || num_pos_neg.second==0)
		return false;

	return true;
}

bool split_node(Node* node, SMatF* trn_ft_mat, SMatF* trn_lbl_mat, _float log_loss_coeff, _float bias, VecI& pos_or_neg)
{
	VecI& insts = node->insts;
	pos_or_neg.resize(insts.size());
 
	for(_int i=0; i<insts.size(); i++)
	{
		_llint r = reng();

		if(r%2)
			pos_or_neg[i] = 1;
		else
			pos_or_neg[i] = -1;
	}

	// one run of ndcg optimization
	bool success;
	success = optimize_ndcg(insts, pos_or_neg);
	if(!success)
		return false;

	// one run of log-loss optimization
	success = optimize_log_loss(insts, trn_ft_mat, log_loss_coeff, bias, pos_or_neg, node->w);
	if(!success)
		return false;

	return true;
}

Tree* train_tree(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param param, _int tree_no)
{
	reng.seed(tree_no);

	_int num_trn = trn_ft_mat->nc;
	_int num_ft = trn_ft_mat->nr;
	_int num_lbl = trn_lbl_mat->nr;

	setup(num_trn,num_ft,num_lbl);
	trn_setup(trn_ft_mat,trn_lbl_mat);

	Tree* tree = new Tree;
	vector<Node*>& nodes = tree->nodes;

	VecI insts;
	for(_int i=0; i<num_trn; i++)
		insts.push_back(i);
	Node* root = new Node(insts,0,param.max_leaf);
	nodes.push_back(root);

	VecI pos_or_neg;

	for(_int i=0; i<nodes.size(); i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
		{
			if(i%1000==0)
				cout<<"\tnode "<<i<<endl;
		}		


		Node* node = nodes[i];
		shrink_ft_mat(trn_ft_mat, node->insts, param.bias);
		shrink_lbl_mat(trn_lbl_mat, node->insts);

		if(node->is_leaf)
		{
			node->pos_child = -1;
			node->neg_child = -1;
			calc_leaf_prob(node, param.lbl_per_leaf);
		}
		else
		{
			bool success = split_node(node, trn_ft_mat, trn_lbl_mat, param.log_loss_coeff, param.bias, pos_or_neg);

			if(success)
			{
				VecI& insts = node->insts;
				VecI pos_insts, neg_insts;
				for(_int j=0; j<insts.size(); j++)
				{
					_int inst = insts[j];
					if(pos_or_neg[j]==+1)
						pos_insts.push_back(inst);
					else
						neg_insts.push_back(inst);
				}
	
				Node* pos_node = new Node(pos_insts, node->depth+1, param.max_leaf);
				nodes.push_back(pos_node);
				node->pos_child = nodes.size()-1;

				Node* neg_node = new Node(neg_insts, node->depth+1, param.max_leaf);
				nodes.push_back(neg_node);
				node->neg_child = nodes.size()-1;
			}
			else
			{
				node->is_leaf = true;
				i--;
			}
		}
	}

	cleanup();
	trn_cleanup();

	return tree;
}

void train_trees_thread(SMatF* trn_ft_mat, SMatF* trn_lbl_mat,Param param, _int s, _int t, string model_folder)
{
	for(_int i=s; i<s+t; i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" training started"<<endl;
		
		Tree* tree = train_tree(trn_ft_mat, trn_lbl_mat, param, i);
		tree->write(model_folder+"/"+to_string(i)+".tree");
		delete tree;

		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" training completed"<<endl;
	}
}

void train_trees(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param param, string model_folder)
{
	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;

	_int s = param.start_tree;
	for(_int i=0; i<param.num_thread; i++)
	{
		if(s < param.start_tree+param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree+param.num_tree-s);
			threads.push_back(thread(train_trees_thread, trn_ft_mat, trn_lbl_mat, param, s, t, model_folder));
			s += t;
		}
	}
	
	for(_int i=0; i<threads.size(); i++)
		threads[i].join();
}

SMatF* test_tree(SMatF* tst_ft_mat, Tree* tree, Param param)
{
	_int num_tst = tst_ft_mat->nc;
	_int num_ft = param.num_ft;
	_int num_lbl = param.num_lbl;

	setup(num_tst,num_ft,num_lbl);

	vector<Node*>& nodes = tree->nodes;
	Node* node = nodes[0];
	node->insts.clear();

	for(_int i=0; i<num_tst; i++)
		node->insts.push_back(i);

	SMatF* tst_score_mat = new SMatF(num_lbl,num_tst);
	VecI pos_or_neg;

	for(_int i=0; i<nodes.size(); i++)
	{
		Node* node = nodes[i];
	
		if(!node->is_leaf)
		{
			VecI& insts = node->insts;
			test_svm(insts, tst_ft_mat, node->w, param.bias, pos_or_neg);
			Node* pos_node = nodes[node->pos_child];
			pos_node->insts.clear();
			Node* neg_node = nodes[node->neg_child];
			neg_node->insts.clear();

			for(_int j=0; j<insts.size(); j++)
			{
				if(pos_or_neg[j]==+1)
					pos_node->insts.push_back(insts[j]);
				else
					neg_node->insts.push_back(insts[j]);
			}
		}
		else
		{
			VecI& insts = node->insts;
			VecIF& leaf_dist = node->leaf_dist;
			_int* size = tst_score_mat->size;
			pairIF** data = tst_score_mat->data;

			for(_int j=0; j<insts.size(); j++)
			{
				_int inst = insts[j];
				size[inst] = leaf_dist.size();
				data[inst] = new pairIF[leaf_dist.size()];

				for(_int k=0; k<leaf_dist.size(); k++)
					data[inst][k] = leaf_dist[k];
			}
		}
	}

	cleanup();
	return tst_score_mat;
}

_float tree_ram_sum;
void test_trees_thread(SMatF* tst_ft_mat, SMatF* score_mat, Param param, _int s, _int t, string model_folder )
{
	for(_int i=s; i<s+t; i++)
	{
		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" testing started"<<endl;

		string tree_name = model_folder+"/"+to_string(i)+".tree";
		check_valid_filename(tree_name);

		Tree* tree = new Tree(tree_name);
		SMatF* tree_score_mat = test_tree(tst_ft_mat, tree, param);

		{
			lock_guard<mutex> lock(mtx);
			score_mat->add(tree_score_mat);
			tree_ram_sum += tree->get_ram();
		}

		delete tree;
		delete tree_score_mat;

		if(loglvl == LOGLVL::PROGRESS)
			cout<<"tree "<<i<<" testing completed"<<endl;
	}
}

SMatF* test_trees(SMatF* tst_ft_mat, Param param, string model_folder, _float& ram)
{
	SMatF* score_mat = new SMatF(param.num_lbl, tst_ft_mat->nc);
	tree_ram_sum = 0;

	_int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
	vector<thread> threads;

	_int s = param.start_tree;
	for(_int i=0; i<param.num_thread; i++)
	{
		if(s < param.start_tree+param.num_tree)
		{
			_int t = min(tree_per_thread, param.start_tree+param.num_tree-s);
			threads.push_back(thread(test_trees_thread, tst_ft_mat, ref(score_mat), param, s, t, model_folder ));
			s += t;
		}
	}
	
	for(_int i=0; i<threads.size(); i++)
		threads[i].join();

	ram = tree_ram_sum;

	for(_int i=0; i<score_mat->nc; i++)
		for(_int j=0; j<score_mat->size[i]; j++)
			score_mat->data[i][j].second /= param.num_tree;

	return score_mat;
}

