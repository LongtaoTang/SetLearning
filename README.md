# SetLearning

This repository is the source code of "Sequence-to-Set generative model" Submitted to 36th Conference on Neural Information Processing Systems (NeurIPS 2022).

You need python to run it. Pytorch, networkx, numpy are needed.

## utility: 
  util.py: contains some useful function.
  metric.py: contains a function to calculate L1-distance.
  
## SizeBias:
  plan1.py: output the sizebias array by plan1.
  
## Baseline:
  baseline method which use the histogram of training data. 
  
## DCM:
  method form paper "A Discrete Choice Model for Subset Selection"

## RW: 
method form paper "Representation Learning for Predicting Customer Orders", the file randomwalk_fast_neighborhood_v3.py is copy form thier source code.

## SubsetLearningFramework:
SubsetLearningFramework.py: contain the Sequential2Set interface.
You can extend this class for you own Sequential2Set model.  

GRUv2.py: class of GRU2Set model. 

SetEmbeddingv2.py: class of SetNN.

SparseGragh: biuld the item graph. 

## tasks:

Dataset are store here. trained model and predicting data also store in here. (We delete the predicting data since thier are too large, while the trained model are remained)

task1 is the Tmall task. 

task3 is the HKTVmall task. To apply the data: https://opendatabank.hktvmall.com/portal/register, If you get the access right, email the author of this paper, we will give you the training and testing set we used in the experiment.
