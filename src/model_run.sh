python main.py \
--sagemaker False \
--num_node_types 8 \
--num_edge_types 4 \
--num_train 65000 \
--source_types 0,1,2,3,4,5,6,7 \
--sampling_size 160 \
--batch_s 32 \
--mini_batch_s 8 \
--eval_size 10 \
--unzip False \
--s3_stage True \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--augmentation_method all \
--edge_ratio_percentile 0.95 \
--edge_mutate_prob 0.1 \
--add_method rare \
--edge_addition_pct 0.1 \
--replace_edges True \
--swap_node_pct 0.1 \
--swap_edge_pct 0.1 \
--main_loss semi-svdd \
--weighted_loss deviation \
--loss_weight 0 \
--eval_method both \
--model_path ../model_save_tralog_gcn11_all \
--data_path ../ProcessedData_HetGCN \
--job_prefix test

sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html


# StreamSpot Data
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 64 \
--hidden_channels 64 \
--num_hidden_conv_layers 3 \
--embed_activation sigmoid \
--augmentation_method all \
--add_method rare \
--edge_addition_pct 0.1 \
--replace_edges True \
--swap_node_pct 0.1 \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method both \
--model_path ../model_save_streamspot_gcn11 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspot


# DeepTraLog Baseline
python main.py \
--sagemaker False \
--num_train 65000 \
--sampling_size 320 \
--batch_s 160 \
--mini_batch_s 160 \
--eval_size 10 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--model_version 12 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 3 \
--model_path ../model_save_tralog_gcn12 \
--data_path ../ProcessedData_HetGCN

# StreamSpot Baseline
python main.py \
--sagemaker False \
--num_train 375 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 25 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--model_version 9 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 3 \
--model_path ../model_save_streamspot_gcn12 \
--data_path ../ProcessedData_streamspot




# Testing
# 11
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--random_seed 36 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 3 \
--embed_activation sigmoid \
--tolerance 5 \
--augmentation_method all \
--add_method rare \
--edge_addition_pct 0.1 \
--replace_edges True \
--swap_node_pct 0.1 \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method svdd \
--model_path ../model_save_streamspot_gcn11 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspotv11

# 8
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--trainer_version 0 \
--model_version 8 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 3 \
--embed_activation sigmoid \
--augmentation_method all \
--add_method rare \
--edge_addition_pct 0.1 \
--replace_edges True \
--swap_node_pct 0.1 \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method both \
--model_path ../model_save_streamspot_gcn8 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspotv8


# 7
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--trainer_version 0 \
--model_version 7 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 3 \
--embed_activation sigmoid \
--augmentation_method all \
--add_method rare \
--edge_addition_pct 0.1 \
--replace_edges True \
--swap_node_pct 0.1 \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method both \
--model_path ../test_save_streamspot_gcn7 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspotv7


# Ablation - TraLog
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_edge_types 4 \
--source_types 0,1,2,3,4,5,6,7 \
--num_train 65000 \
--sampling_size 160 \
--batch_s 32 \
--mini_batch_s 8 \
--eval_size 10 \
--unzip False \
--s3_stage True \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--augmentation_method all \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--ablation no-node-relation \
--eval_method svdd \
--model_path ../model_save_tralog_gcn11_all \
--data_path ../ProcessedData_HetGCN \
--job_prefix test


# Ablation - StreamSpot
python main.py \
--sagemaker False \
--num_node_types 8 \
--source_types 0,1 \
--num_train 375 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--random_seed 36 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 2 \
--embed_activation sigmoid \
--tolerance 5 \
--augmentation_method all \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--ablation no-edge-relation \
--eval_method svdd \
--model_path ../model_save_streamspot_gcn11 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspotv11



# h2 - StreamSpot Hyperparameter Analysis
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_train 375 \
--source_types 0,1 \
--sampling_size 375 \
--batch_s 25 \
--mini_batch_s 25 \
--eval_size 375 \
--unzip False \
--s3_stage False \
--split_data False \
--ignore_weight False \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 100 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 26 \
--out_embed_s 32 \
--hidden_channels 32 \
--num_hidden_conv_layers 2 \
--embed_activation sigmoid \
--tolerance 5 \
--augmentation_method all \
--add_method rare \
--edge_addition_pct 0.39185763245124894 \
--replace_edges True \
--swap_node_pct 0.5266846615473234 \
--main_loss svdd \
--weighted_loss deviation \
--loss_weight 0.2129864286429184 \
--eval_method both \
--model_path ../model_save_streamspot_gcn11 \
--data_path ../ProcessedData_streamspot \
--job_prefix test_streamspotv11


# h1 - Tralog
python main.py \
--sagemaker False \
--num_node_types 8 \
--num_edge_types 4 \
--num_train 65000 \
--source_types 0,1,2,3,4,5,6,7 \
--sampling_size 160 \
--batch_s 32 \
--mini_batch_s 8 \
--eval_size 10 \
--unzip False \
--s3_stage True \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.0001 \
--train_iter_n 200 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 7 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 1 \
--augmentation_method all \
--edge_ratio_percentile 0.95 \
--edge_mutate_prob 0.8420627973829723 \
--add_method rare \
--edge_addition_pct 0.12868699273268602 \
--replace_edges True \
--swap_node_pct 0.10941908541074977 \
--swap_edge_pct 0.17953551869297305 \
--main_loss svdd \
--weighted_loss bce \
--loss_weight 0.0009732460622703387 \
--eval_method both \
--model_path ../model_save_tralog_gcn11_all \
--data_path ../ProcessedData_HetGCN \
--job_prefix test