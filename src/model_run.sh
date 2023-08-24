# streamspot
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
--model_path ../model/model_save_streamspot_gcn11 \
--data_path ../data//ProcessedData_streamspot

# Tralog
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
--model_path ../model/model_save_tralog_gcn11_all \
--data_path ../data//ProcessedData_HetGCN \