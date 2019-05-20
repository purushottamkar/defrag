#ifndef GROUP_H
#define GROUP_H

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>
#include <fstream>
#include <map>
#include <random>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cassert>

#include "config.h"
#include "utils.h"
#include "timer.h"

using namespace std;

class Group 
{
public:

	int num_groups;
	int *size;
	int **data;

	Group()
	{

	}

	Group(int ng)
	{
		num_groups = ng;
		size = new int[num_groups]();
		data = new int*[num_groups]();		
	}

	void dump_groups(string filename)
	{
		ofstream fout;
		fout.open(filename);

		fout << num_groups << endl;
		for(_int i=0; i<num_groups; i++){
			fout << size[i] << " ";
			for(_int j=0; j<size[i]; j++){
				fout << data[i][j] << " ";
			}
			fout << endl;
		}
		fout.close();
	}

	void read_groups(string filename)
	{
		ifstream fin;
		fin.open(filename);

		fin >> num_groups;

		assert(num_groups > 0);
		size = new int[num_groups]();
		data = new int*[num_groups]();	
		
		for(int i=0; i<num_groups; i++)
		{
			fin >> size[i];
			data[i] = new int[size[i]]();

			for(int j=0; j<size[i]; j++){
				fin >> data[i][j];
			}
		}
	}

	float get_ram()
	{
		float r = 0;
		for(int i=0; i<num_groups; i++){
			r += size[i]*4;
		}
		return r;
	}

	~Group()
	{
		for(int i=0; i<num_groups; i++){
			delete [] data[i];
		}
		delete [] size;
	}
};

#endif