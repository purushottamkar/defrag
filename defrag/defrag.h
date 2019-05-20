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
#include "timer.h"
#include "group.h"
#include "svm.h"

using namespace std;

enum _Classifier_Kind { L2R_L2LOSS_SVC=0, L2R_LR };

class Node
{
public:
	_bool is_leaf;
	_int pos_child;
	_int neg_child;
	_int depth;
	VecI Y;
	SMatF* w;
	VecIF X;

	Node()
	{
		is_leaf = false;
		pos_child = neg_child = -1;
		depth = 0;
		w = NULL;
	}

	Node( VecI Y, _int depth, _int max_depth )
	{
		this->Y = Y;
		this->depth = depth;
		this->pos_child = -1;
		this->neg_child = -1;
		this->is_leaf = (depth >= max_depth-1);
		this->w = NULL;
	}

	~Node()
	{
		delete w;
	}

	_float get_ram()
	{
		_float ram = 0;
		ram += sizeof( Node );
		if( this->is_leaf )  // Label values in internal nodes are not essential for model and hence not included in model size measurements
			ram += sizeof( _int ) * Y.size();
		ram += w->get_ram();
		return ram;
	}

	friend ostream& operator<<( ostream& fout, const Node& node )
	{
		fout << node.is_leaf << "\n";
		fout << node.pos_child << " " << node.neg_child << "\n";
		fout << node.depth << "\n";

		fout << node.Y.size();
		for( _int i=0; i<node.Y.size(); i++ )
			fout << " " << node.Y[i];
		fout << "\n";

		fout << (*node.w);

		return fout;
	}

 	friend istream& operator>>( istream& fin, Node& node )
	{
		fin >> node.is_leaf;
		fin >> node.pos_child >> node.neg_child;
		fin >> node.depth;

		_int Y_size;
		fin >> Y_size;
		node.Y.resize( Y_size );

		for( _int i=0; i<Y_size; i++ )
			fin >> node.Y[i];

		node.w = new SMatF;
		fin >> (*node.w);

		return fin;
	} 
};

class Tree
{
public:
	_int num_Xf;
	_int num_Y;
	vector<Node*> nodes;

	Tree()
	{
		
	}

	Tree( string model_dir, _int tree_no )
	{
		ifstream fin;
		fin.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		fin >> num_Xf;
		fin >> num_Y;
		_int num_node;
		fin >> num_node;

		for( _int i=0; i<num_node; i++ )
		{
			Node* node = new Node;
			nodes.push_back( node );
		}

		for( _int i=0; i<num_node; i++ )
			fin >> (*nodes[i]);

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
		ram += sizeof( Tree );
		for(_int i=0; i<nodes.size(); i++)
			ram += nodes[i]->get_ram();
		return ram;
	}

	void write( string model_dir, _int tree_no )
	{
		ofstream fout;
		fout.open( model_dir + "/" + to_string( tree_no ) + ".tree" );

		fout << num_Xf << "\n";
		fout << num_Y << "\n";
		_int num_node = nodes.size();
		fout << num_node << "\n";

		for( _int i=0; i<num_node; i++ )
			fout << (*nodes[i]);

		fout.close();
	}
};

class Param
{
public:
	_int num_trn;
	_int num_Xf;
	_int num_Y;
	_int num_thread;
	_int start_tree;
	_int num_tree;
	_float classifier_cost;
	_int max_leaf;
	_float bias_feat;
	_float classifier_threshold;
	_float centroid_threshold;
	_float clustering_eps;
	_int classifier_maxitr;
	_Classifier_Kind classifier_kind;
	_bool quiet;
	_int beam_width;

	string logfile;
	string model_dir;
	string grp_file;
	string agg_file;

	_int avg;

	_int feature_representation;	// XT XTY XT-XTY

	_int cluster_label_sample;		// Number of labels to use for clustering.
	_int cluster_data_sample;		// Number of data points to use for clustering.

	_float cluster_maxleaf;			// Max number of features in the leaf node.
	
	
	Param()
	{
		feature_representation = 1;
		cluster_label_sample = 5;
		cluster_data_sample = 20;
		cluster_maxleaf = 8;

		avg = 0;
	}
};

void defrag_clustering( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _float& cluster_time );
void defrag_agglomeration( SMatF* tst_X_Xf, Param& param, _float& agglomeration_time );
