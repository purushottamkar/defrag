# DEFRAG

## Introduction
Accelerating Extreme Classification via Adaptive Feature Agglomeration

# Running DEFRAG
DEFARG is executed in two steps:
- defrag_clustering: This computes a grouping of features.
- defrag_agglomeration: This agglomerates the features based on groupings obtained from previous step.

Please refer to sample_run.py for more information on how to use DEFRAG.

Feature and label files should be formatted as expected by Parabel.

# Parameters
Following parameters can be tuned in DEFRAG

## defrag_clustering
fr  = param.feature_representation  : Use feture repersentation X or XY, default 1 (X).<br/>
cml = param.cluster_maxleaf         : Maximum number of features in a leaf node of DEFRAG tree, default 8.<br/>
cls = param.cluster_label_sample    : Percentage of labels used for clustering, default 5.<br/>
cds = param.cluster_data_sample     : Percentage of data points used for clustering, default 20.<br/>

## defrag_agglomeraton
avg = param.avg	: Average out non-zero entries while agglomeration, default 0"<<endl;
	
# Acknowledgement

The code is adapted and subsequently modified from the source code provided by the authors of [Parabel: Partitioned Label Trees for Extreme Classification with Application to Dynamic Search Advertising](https://dl.acm.org/citation.cfm?id=3185998). 
