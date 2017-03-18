#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <functional>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <ctime>

#include "config.h"
#include "utils.h"
#include "mat.h"

using namespace std;

extern LOGLVL loglvl;
extern mutex mtx;
extern _bool USE_IDCG;

class Node
{
public:
	_bool is_leaf;
	_int pos_child;
	_int neg_child;
	_int depth;
	VecI insts;
	VecIF w;
	VecIF leaf_dist;


	Node()
	{
		is_leaf = false;
		depth = 0;
		pos_child = neg_child = -1;
	}

	Node(VecI insts, _int depth, _int max_leaf)
	{
		this->insts = insts;
		this->depth = depth;
		this->pos_child = -1;
		this->neg_child = -1;

		if(insts.size()<=max_leaf)
			this->is_leaf = true;
		else
			this->is_leaf = false;
	}

	~Node()
	{
	}

	_float get_ram()
	{
		_float ram = 1 + 4*3;
		ram += insts.size()*4;
		ram += w.size()*8;
		ram += leaf_dist.size()*8;
	
		return ram;
	}

	friend ostream& operator<<(ostream& fout, const Node& node)
	{
		fout<<(node.is_leaf?1:0)<<endl;

		fout<<node.pos_child<<" "<<node.neg_child<<endl;
		fout<<node.depth<<endl;

		fout<<node.insts.size();
		for(_int i=0; i<node.insts.size(); i++)
			fout<<" "<<node.insts[i];
		fout<<endl;

		if(node.is_leaf)
		{
			fout<<node.leaf_dist.size();
			for(_int i=0; i<node.leaf_dist.size(); i++)
			{
				fout<<" "<<node.leaf_dist[i].first<<":"<<node.leaf_dist[i].second;
			}
			fout<<endl;
		}
		else
		{
			fout<<node.w.size();
			for(_int i=0; i<node.w.size(); i++)
			{
				fout<<" "<<node.w[i].first<<":"<<node.w[i].second;
			}
			fout<<endl;
		}
		return fout;
	}

	friend istream& operator>>(istream& fin, Node& node)
	{
		fin>>node.is_leaf;
		fin>>node.pos_child>>node.neg_child>>node.depth;

		_int siz;
		_int ind;
		_float val;
		char c;

		node.insts.clear();
		fin>>siz;
		for(_int i=0; i<siz; i++)
		{
			fin>>ind;	
			node.insts.push_back(ind);
		}

		if(node.is_leaf)
		{
			node.leaf_dist.clear();
			fin>>siz;
			for(_int i=0; i<siz; i++)
			{
				fin>>ind>>c>>val;
				node.leaf_dist.push_back(make_pair(ind,val));
			}
		}
		else
		{
			node.w.clear();
			fin>>siz;
			for(_int i=0; i<siz; i++)
			{
				fin>>ind>>c>>val;
				node.w.push_back(make_pair(ind,val));
			}	
		}
		return fin;
	}
};

class Tree
{
public:
	vector<Node*> nodes;

	Tree()
	{
		
	}

	Tree(string fname)
	{
		ifstream fin;
		fin.open(fname);

		_int num_nodes;
		fin>>num_nodes;

		for(_int i=0; i<num_nodes; i++)
		{
			Node* node = new Node;
			fin>>(*node);
			nodes.push_back(node);
		}
		
		fin.close();
	}

	~Tree()
	{
		for(_int i=0; i<nodes.size(); i++)
			delete nodes[i];
	}

	_float get_ram()
	{
		_float ram = 0;
		for(_int i=0; i<nodes.size(); i++)
			ram += nodes[i]->get_ram();

		return ram;
	}

	void write(string fname)
	{
		ofstream fout;
		fout.open(fname);

		fout<<nodes.size()<<endl;

		for(_int i=0; i<nodes.size(); i++)
		{
			Node* node = nodes[i];
			fout<<(*node);
		}

		fout.close();
	}
};

class Param
{
public:
	_int num_ft;
	_int num_lbl;
	_float log_loss_coeff;
	_int max_leaf;
	_int lbl_per_leaf;
	_float bias;
	_int num_thread;
	_int start_tree;
	_int num_tree;
	_bool quiet;

	Param()
	{
		num_ft = 0;
		num_lbl = 0;
		log_loss_coeff = 1.0;
		max_leaf = 10;
		lbl_per_leaf = 100;
		bias = 1.0;
		num_thread = 1;
		start_tree = 0;
		num_tree = 50;
		quiet = false;
	}

	Param(string fname)
	{
		check_valid_filename(fname,true);
		ifstream fin;
		fin.open(fname);

		fin>>num_ft;
		fin>>num_lbl;
		fin>>log_loss_coeff;
		fin>>max_leaf;
		fin>>lbl_per_leaf;
		fin>>bias;
		fin>>num_thread;
		fin>>start_tree;
		fin>>num_tree;
		fin>>quiet;

		fin.close();
	}

	void write(string fname)
	{
		check_valid_filename(fname,false);
		ofstream fout;
		fout.open(fname);

		fout<<num_ft<<endl;
		fout<<num_lbl<<endl;
		fout<<log_loss_coeff<<endl;
		fout<<max_leaf<<endl;
		fout<<lbl_per_leaf<<endl;
		fout<<bias<<endl;
		fout<<num_thread<<endl;
		fout<<start_tree<<endl;
		fout<<num_tree<<endl;
		fout<<quiet<<endl;

		fout.close();
	}
};

Tree* train_tree(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param param, _int tree_no);
void train_trees(SMatF* trn_ft_mat, SMatF* trn_lbl_mat, Param param, string model_folder);

SMatF* test_tree(SMatF* tst_ft_mat, Tree* tree, Param param);
SMatF* test_trees(SMatF* tst_ft_mat, Param param, string model_folder, _float& ram);

