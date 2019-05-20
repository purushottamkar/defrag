#include "group.h"
#include "defrag.h"

using namespace std;

LOGLVL loglvl;
mutex mtx;
_bool USE_IDCG;

thread_local mt19937 reng; // random number generator used during training 
thread_local VecF discounts;
thread_local VecF csum_discounts;
thread_local VecF dense_w;
thread_local VecI countmap;

void setup_thread_locals_pbl( _int num_X, _int num_Xf, _int num_Y )
{
	countmap.resize( num_Xf + num_Y + num_X , 0 );
}

_int get_rand_num( _int siz )
{
	_llint r = reng();
	_int ans = r % siz;
	return ans;
}

Node* init_root( _int num_Y, _int max_depth )
{
	VecI lbls;
	for( _int i=0; i<num_Y; i++ )
		lbls.push_back(i);
	Node* root = new Node( lbls, 0, max_depth );
	return root;
}

pairII get_pos_neg_count_pbl( VecI& pos_or_neg )
{
	pairII counts = make_pair(0,0);
	for( _int i=0; i<pos_or_neg.size(); i++ )
	{
		if(pos_or_neg[i]==+1)
			counts.first++;
		else
			counts.second++;
	}
	return counts;
}

void reset_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
		dvec[ svec[i].first ] = 0;
}

void set_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
		dvec[ svec[i].first ] = svec[i].second;
}

void init_2d_float( _int dim1, _int dim2, _float**& mat )
{
	mat = new _float*[ dim1 ];
	for( _int i=0; i<dim1; i++ )
		mat[i] = new _float[ dim2 ]; 
}

void delete_2d_float( _int dim1, _int dim2, _float**& mat )
{
	for( _int i=0; i<dim1; i++ )
		delete [] mat[i];
	delete [] mat;
	mat = NULL;
}

void reset_2d_float( _int dim1, _int dim2, _float**& mat )
{
	for( _int i=0; i<dim1; i++ )
		for( _int j=0; j<dim2; j++ )
			mat[i][j] = 0;
}

_float mult_d_s_vec( _float* dvec, pairIF* svec, _int siz )
{
	_float prod = 0;
	for( _int i=0; i<siz; i++ )
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		prod += dvec[ id ] * val;
	}
	return prod;
}

void add_s_to_d_vec( pairIF* svec, _int siz, _float* dvec )
{
	for( _int i=0; i<siz; i++ )
	{
		_int id = svec[i].first;
		_float val = svec[i].second;
		dvec[ id ] += val;
	}
}

_float get_norm_d_vec( _float* dvec, _int siz )
{
	_float norm = 0;
	for( _int i=0; i<siz; i++ )
		norm += SQ( dvec[i] );
	norm = sqrt( norm );
	return norm;
}

void div_d_vec_by_scalar( _float* dvec, _int siz, _float s )
{
	for( _int i=0; i<siz; i++)
		dvec[i] /= s;
}

void normalize_d_vec( _float* dvec, _int siz )
{
	_float norm = get_norm_d_vec( dvec, siz );
	if( norm>0 )
		div_d_vec_by_scalar( dvec, siz, norm );
}

void balanced_kmeans( SMatF* mat, _float acc, VecI& partition )
{
	_int nc = mat->nc;
	_int nr = mat->nr;

	_int c[2] = {-1,-1};
	c[0] = get_rand_num( nc );
	c[1] = c[0];
	while( c[1] == c[0] )
		c[1] = get_rand_num( nc );

	_float** centers;
	init_2d_float( 2, nr, centers );
	reset_2d_float( 2, nr, centers );
	for( _int i=0; i<2; i++ )
		set_d_with_s( mat->data[c[i]], mat->size[c[i]], centers[i] );

	_float** cosines;
	init_2d_float( 2, nc, cosines );
	
	pairIF* dcosines = new pairIF[ nc ];

	partition.resize( nc );

	_float old_cos = -10000;
	_float new_cos = -1;

	while( new_cos - old_cos >= acc )
	{

		for( _int i=0; i<2; i++ )
		{
			for( _int j=0; j<nc; j++ )
				cosines[i][j] = mult_d_s_vec( centers[i], mat->data[j], mat->size[j] );
		}

		for( _int i=0; i<nc; i++ )
		{
			dcosines[i].first = i;
			dcosines[i].second = cosines[0][i] - cosines[1][i];
		}
		
		sort( dcosines, dcosines+nc, comp_pair_by_second_desc<_int,_float> );

		old_cos = new_cos;
		new_cos = 0;
		for( _int i=0; i<nc; i++ )
		{
			_int id = dcosines[i].first;
			_int part = (_int)(i < nc/2);
			partition[ id ] = 1 - part;
			new_cos += cosines[ partition[id] ][ id ];
		}
		new_cos /= nc;

		reset_2d_float( 2, nr, centers );

		for( _int i=0; i<nc; i++ )
		{
			_int p = partition[ i ];
			add_s_to_d_vec( mat->data[i], mat->size[i], centers[ p ] );
		}

		for( _int i=0; i<2; i++ )
			normalize_d_vec( centers[i], nr );
	}

	delete_2d_float( 2, nr, centers );
	delete_2d_float( 2, nc, cosines );
	delete [] dcosines;
}

void shrink_data_matrices( SMatF* trn_X_Xf, SMatF* trn_Y_X, SMatF* cent_mat, VecI& n_Y, Param& param, SMatF*& n_trn_X_Xf, SMatF*& n_trn_Y_X, SMatF*& n_cent_mat, VecI& n_X, VecI& n_Xf, VecI& n_cXf )
{
	trn_Y_X->shrink_mat( n_Y, n_trn_Y_X, n_X, countmap, false );
	trn_X_Xf->shrink_mat( n_X, n_trn_X_Xf, n_Xf, countmap, false );
	cent_mat->shrink_mat( n_Y, n_cent_mat, n_cXf, countmap, false );
}

Group* cluster_features_parabel( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _int id)
{
	reng.seed(id);

	param.num_trn = trn_X_Xf->nc;
	
	SMatF* trn_Y_X = trn_X_Y->transpose();
	SMatF* cent_mat;

	_int feature_representation = param.feature_representation;

	if(feature_representation == 1)
	{
		cent_mat = trn_Y_X->copy();
	}
	if(feature_representation == 2)
	{
		cent_mat = trn_X_Xf->prod( trn_Y_X );
	}

	cent_mat->unit_normalize_columns(); 
	cent_mat->threshold( param.centroid_threshold );

	_int num_X = trn_X_Xf->nc;
	_int num_Xf = trn_X_Xf->nr;
	_int num_Y = trn_Y_X->nc;
	setup_thread_locals_pbl( num_X, num_Xf, num_Y );

	_int max_depth = ceil( log2( num_Y/param.cluster_maxleaf ) ) + 1;
	
	Tree* tree = new Tree;
	vector<Node*>& nodes = tree->nodes;

	Node* root = init_root( num_Y, max_depth );
	nodes.push_back( root );

	_int num_leaf = 0;

	for(_int i=0; i<nodes.size(); i++)
	{
		if( i%1000==0 )
			cout << "node " << i << endl;

		if( nodes[i]->is_leaf )
		{
			continue;
		}


		Node* node = nodes[i];
		VecI& n_Y = node->Y;
		SMatF* n_trn_X_Xf;
		SMatF* n_trn_Y_X;
		SMatF* n_cent_mat;
		VecI n_X;
		VecI n_Xf;
		VecI n_cXf;

		shrink_data_matrices( trn_X_Xf, trn_Y_X, cent_mat, n_Y, param, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, n_X, n_Xf, n_cXf );		

		VecI partition;

		balanced_kmeans( n_cent_mat, param.clustering_eps, partition );

		VecI pos_Y, neg_Y;
		for( _int j=0; j<n_Y.size(); j++ )
			if( partition[j] )
				pos_Y.push_back( n_Y[ j ] );
			else
				neg_Y.push_back( n_Y[ j ] );

		if(pos_Y.size() > 0){
			Node* pos_node = new Node( pos_Y, node->depth+1, max_depth );
			nodes.push_back( pos_node );
			node->pos_child = nodes.size()-1;

			if(pos_Y.size() <= param.cluster_maxleaf || neg_Y.size() == 0){
				pos_node->is_leaf = 1;
			}
		}

		if(neg_Y.size() > 0){
			Node* neg_node = new Node( neg_Y, node->depth+1, max_depth );
			nodes.push_back(neg_node);
			node->neg_child = nodes.size()-1;

			if(neg_Y.size() <= param.cluster_maxleaf || pos_Y.size() == 0){
				neg_node->is_leaf = 1;
			}
		}

		delete n_trn_X_Xf;
		delete n_trn_Y_X;
		delete n_cent_mat;
	}

	tree->num_Xf = num_Xf;
	tree->num_Y = num_Y;

	_float max_leaf_depth = 0;
	_float min_leaf_depth = 10000000;

	for(int n=0; n<nodes.size(); n++) {
		if(nodes[n]->is_leaf){
			num_leaf++;
		}
	}

	Group *groups = new Group(num_leaf);

	int n = 0;
	for(int j=0; j<nodes.size(); j++)
	{
		Node* node = nodes[j];
		if( node->is_leaf )
		{
			groups->size[n] = node->Y.size();
			groups->data[n] = new int[groups->size[n]]();

			for( _int j=0; j<groups->size[n]; j++ ){
				groups->data[n][j] = node->Y[j];
			}

			max_leaf_depth = max(max_leaf_depth, (_float)(node->depth));
			min_leaf_depth = min(min_leaf_depth, (_float)(node->depth));

			n++;
		}
	}

	delete trn_Y_X;
	delete cent_mat;
	delete tree;

	return groups;
}

Group* get_cluster_groups(SMatF* trn_X_Xf, SMatF* trn_X_Y, Param param, _float *time, _int group_id)
{
	Timer timer;
	timer.tic();

	SMatF *trn_X_Xf_sample;
	SMatF *trn_X_Y_dsample, *trn_X_Y_lsample;
	SMatF *XT, *XTY;
	
	Group *groups;

	_int feature_representation = param.feature_representation;

	// SAMPLING OF DATA AND LABELS,FINALLY WE WILL USE
	// trn_X_Xf_sample
	// trn_X_Y_lsample

	_int cds = param.cluster_data_sample;
	_int cls = param.cluster_label_sample;

	if(param.cluster_data_sample < 100)
	{
		vector<_int> select_col = trn_X_Xf->sample_lengthy_documents(param.cluster_data_sample);
		
		trn_X_Xf_sample = trn_X_Xf->sample_users(select_col);
		trn_X_Y_dsample = trn_X_Y->sample_users(select_col);
	}
	else
	{
		trn_X_Xf_sample = trn_X_Xf->copy();
		trn_X_Y_dsample = trn_X_Y->copy();
	}

	if(param.cluster_label_sample < 100)
	{
		trn_X_Y_lsample = trn_X_Y_dsample->sample_top_labels(param.cluster_label_sample);
	}
	else{
		trn_X_Y_lsample = trn_X_Y_dsample->copy();
	}

	// no use of this now.
	delete trn_X_Y_dsample;

	// KMeans based clustering
	groups =  cluster_features_parabel( trn_X_Y_lsample, trn_X_Xf_sample, param, group_id);

	*time += timer.toc();

	delete trn_X_Xf_sample;
	delete trn_X_Y_lsample;
	
	return groups;
}

void defrag_clustering( SMatF* trn_X_Xf, SMatF* trn_X_Y, Param& param, _float& cluster_time )
{
	_float* c_time = new _float;
	*c_time = 0;	

	Group *group;
	cout << "Clustering started" << endl;
	
	group = get_cluster_groups(trn_X_Xf, trn_X_Y, param, c_time, 1);
	cluster_time += *c_time;
	
	cout << "Clustering end" << endl;

	group->dump_groups(param.grp_file);
}

void defrag_agglomeration( SMatF* tst_X_Xf, Param& param, _float& agglomeration_time )
{
	Timer timer;
	timer.tic();

	Group *group = new Group();
	group->read_groups(param.grp_file);

	SMatF *tst_X_Xf_D = new SMatF();

	tst_X_Xf_D = tst_X_Xf->make_dense(group);
	
	if(param.avg)
	{
		SMatF* aone = tst_X_Xf->all_ones();
		SMatF *aone_dense = aone->make_dense(group);
		tst_X_Xf_D->elem_div(aone_dense);
	}
	
	agglomeration_time += timer.toc();
	tst_X_Xf_D->write(param.agg_file);
}