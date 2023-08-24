# HRGCN
Code Repository for Paper "HRGCN: Heterogeneous Graph-level Anomaly Detection with Hierarchical Relation-augmented Graph Neural Networks"

## Abstract
> This work considers the problem of heterogeneous graph-level anomaly detection. Heterogeneous graphs are commonly used to represent behaviours between different types of entities in complex industrial systems for capturing as much information about the system operations as possible. Detecting anomalous heterogeneous graphs from a large set of system behaviour graphs is crucial for many real-world applications like online web/mobile service and cloud access control. To address the problem, we propose HRGCN, an unsupervised deep heterogeneous graph neural network, to model complex heterogeneous relations between different entities in the system for effectively identifying these anomalous behaviour graphs. HRGCN trains a hierarchical relation-augmented Heterogeneous Graph Neural Network (HetGNN), which learns better graph representations by modelling the interactions among all the system entities and considering both source-to-destination entity (node) types and their relation (edge) types. Extensive evaluation on two real-world application datasets shows that HRGCN outperforms state-of-the-art competing anomaly detection approaches. We further present a real-world industrial case study to justify the effectiveness of HRGCN in detecting anomalous (e.g., congested) network devices in a mobile communication service.

## Install
Dependency Libs:
- gcc>=7.2
- cuda>=10.2
- torch>=1.9.1
- torch-geometric

## Data

## Train and Evaluate
### Run the `FlowGraph` Dataset
```shell
# FlowGraph
python main.py \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--ignore_weight False \
--lr 0.01 \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 2 \
--edge_addition_pct 0.39185763245124894 \
--swap_node_pct 0.5266846615473234 \
--loss_weight 0.2129864286429184 \
--model_path ../model/model_save_streamspot \
--data_path ../data//ProcessedData_streamspot
```

### Run the `TraceLog` Dataset
```shell
# TraceLog
python main.py \
--num_node_types 8 \
--num_edge_types 4 \
--num_train 65000 \
--source_types 0,1,2,3,4,5,6,7 \
--sampling_size 160 \
--batch_s 32 \
--mini_batch_s 8 \
--eval_size 10 \
--lr 0.0001 \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--edge_mutate_prob 0.8420627973829723 \
--edge_addition_pct 0.12868699273268602 \
--swap_node_pct 0.10941908541074977 \
--swap_edge_pct 0.17953551869297305 \
--loss_weight 0.0009732460622703387 \
--model_path ../model/model_save_tralog \
--data_path ../data//ProcessedData_HetGCN \
```

## Citation
