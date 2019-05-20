''' Sample script to run DEFRAG code.

The file trainX assumes features in the format required by 
Parabel code. 

Author: Ankit Jalan
'''

import os

sandbox = "sandbox"
dataset = "bibtex"

train_feat = sandbox + "/" + dataset + "/trainX.txt"
train_lab  = sandbox + "/" + dataset + "/trainY.txt"

test_feat  = sandbox + "/" + dataset + "/testX.txt"

train_feat_defrag = sandbox + "/" + dataset + "/trainX_defrag.txt"
test_feat_defrag  = sandbox + "/" + dataset + "/testX_defrag.txt"

grp_file   = sandbox + "/" + dataset + "/group.txt"


# Clusterting
run_command = "./defrag/defrag_clustering " + train_feat + " " + train_lab + " " + grp_file + " -cml 8 -cds 20 -cls 5"
os.system(run_command)

# Agglomeration
run_command = "./defrag/defrag_agglomeration " + train_feat + " " + grp_file + " " + train_feat_defrag + " avg 0"
os.system(run_command)
run_command = "./defrag/defrag_agglomeration " + test_feat + " " + grp_file + " " + test_feat_defrag + " avg 0"
os.system(run_command)