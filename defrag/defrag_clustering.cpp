#include <iostream>
#include <fstream>
#include <string>

#include "defrag.h"

using namespace std;

void help()
{
	cout<<"-------------------------------------------"<<endl;
	cerr<<"Sample Usage :"<<endl;
	cerr<<"./defrag [input feature file name] [input label file name] [output group file name] -cc 1 -cls 5 -cds 20 -cml 8"<<endl;
	cerr<<endl;
	cerr<<"-fr  = param.feature_representation  : Use feture repersentation X or XY, default 1 (X)."<<endl;
	cerr<<"-cml = param.cluster_maxleaf         : Maximum number of features in a leaf node of DEFRAG tree, default 8."<<endl;
	cerr<<"-cls = param.cluster_label_sample    : Percentage of labels used for clustering, default 5."<<endl;
	cerr<<"-cds = param.cluster_data_sample     : Percentage of data points used for clustering, default 20."<<endl;
	cerr<<"Please refer to README.txt for mode details."<<endl;
	cout<<"-------------------------------------------"<<endl;
	exit(1);
}

Param parse_param(_int argc, char* argv[])
{
	Param param;
	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt 	= string(argv[i]);
		sval 	= string(argv[i+1]);
		val 	= stof(sval);

		if( opt == "-fr" )
			param.feature_representation = (_int)(val);
		else if( opt == "-cls" )
			param.cluster_label_sample = (_int)(val);
		else if( opt == "-cds" )
			param.cluster_data_sample = (_int)(val);
		else if( opt == "-cml" )
			param.cluster_maxleaf = (_int)(val);
	}
	
	return param;
}

int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string ft_file = string( argv[1] );
	SMatF* trn_X_Xf = new SMatF( ft_file );

	string lbl_file = string( argv[2] );
	SMatF* trn_X_Y = new SMatF(lbl_file);

	Param param = parse_param( argc-4, argv+4 );

	param.grp_file = string( argv[3] );
	param.num_Xf = trn_X_Xf->nr;
	param.num_Y = trn_X_Y->nr;

	_float cluster_time = 0;
	defrag_clustering( trn_X_Xf, trn_X_Y, param, cluster_time);
	
	cout << "DEFRAG clustering time: " + to_string(cluster_time) + " seconds." << endl;
}
