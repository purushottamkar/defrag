#include <iostream>
#include <fstream>
#include <string>

#include "defrag.h"

using namespace std;

void help()
{
	cout<<"-------------------------------------------"<<endl;
	cout<<"Sample Usage :"<<endl;
	cout<<"./defrag_agglomeration [feature file name] [cluster file name] [agglomerated feature file name] -avg 0"<<endl;
	cout<<endl;
	cout<<"-avg = param.avg	: Average out non-zero entries while agglomeration, default 0"<<endl;
	cout<<"Please refer to README.txt for more details"<<endl;
	cout<<"-------------------------------------------"<<endl;
	exit(1);
}

Param parse_param(int argc, char* argv[])
{
	Param param;

	string opt;
	string sval;
	_float val;

	for(_int i=0; i<argc; i+=2)
	{
		opt = string(argv[i]);
		sval = string(argv[i+1]);
		val = stof(sval);

		if( opt == "-avg")
			param.avg = (_int)(val);
	}

	return param;
}


int main(int argc, char* argv[])
{
	std::ios_base::sync_with_stdio(false);

	if(argc < 4)
		help();

	string ft_file  = string(argv[1]);
	string grp_file = string(argv[2]);
	string agg_file = string(argv[3]);

	Param param = parse_param(argc-4, argv+4);

	param.grp_file = grp_file;
	param.agg_file = agg_file;

	SMatF* tst_X_Xf = new SMatF(ft_file);
		
	_float agglomeration_time = 0;
	defrag_agglomeration( tst_X_Xf, param, agglomeration_time );
	cout << "DEFRAG agglomeration time: " + to_string(agglomeration_time) + " seconds."<< endl;
}
