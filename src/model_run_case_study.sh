sudo update-alternatives --config gcc
sudo ln -sfT /usr/local/cuda-10.2/ /usr/local/cuda
pip install torch==1.9.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.1+cu102.html

# Case Study
python main.py \
--sagemaker False \
--s3_stage False \
--unzip False \
--num_node_types 2 \
--source_types 0 \
--num_train 12619 \
--sampling_size 6400 \
--batch_s 256 \
--mini_batch_s 256 \
--eval_size 5000 \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 51 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 4 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 2 \
--embed_activation relu \
--tolerance 5 \
--augmentation_method all \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method svdd \
--model_path ../model_save_case_study \
--data_path ../tpg_case_study_data_small \
--job_prefix test_case_study

# Case Study - GGNN (DeepTraLog)
python main.py \
--sagemaker False \
--s3_stage False \
--unzip False \
--num_node_types 2 \
--source_types 0 \
--num_train 12619 \
--sampling_size 6400 \
--batch_s 256 \
--mini_batch_s 256 \
--eval_size 5000 \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.001 \
--train_iter_n 51 \
--trainer_version 0 \
--model_version 12 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 4 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 2 \
--embed_activation relu \
--tolerance 5 \
--model_path ../model_save_case_study_tra \
--data_path ../tpg_case_study_data_small \
--job_prefix test_case_study

# Case Study - HetGNN
python main.py \
--sagemaker False \
--s3_stage False \
--unzip False \
--num_node_types 2 \
--source_types 0 \
--num_train 12619 \
--sampling_size 6400 \
--batch_s 256 \
--mini_batch_s 256 \
--eval_size 5000 \
--split_data False \
--ignore_weight True \
--test_set True \
--save_model_freq 2 \
--lr 0.01 \
--train_iter_n 51 \
--trainer_version 2 \
--model_version 11 \
--model_sub_version 0 \
--dataset_id 0 \
--input_type batch \
--feature_size 4 \
--out_embed_s 300 \
--hidden_channels 300 \
--num_hidden_conv_layers 2 \
--embed_activation relu \
--tolerance 5 \
--augmentation_method all \
--main_loss svdd \
--weighted_loss ignore \
--loss_weight 0 \
--eval_method svdd \
--ablation no-edge-node-relation \
--model_path ../model_save_case_study_hetgnn \
--data_path ../tpg_case_study_data_small \
--job_prefix test_case_study