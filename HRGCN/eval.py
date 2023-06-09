from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from data_loader import HetGCNEventGraphDataset


def evaluate_node(model, data_root_dir, ignore_weight, include_edge_type, device):
    valid_dataset = HetGCNEventGraphDataset(
        node_feature_csv=f"{data_root_dir}/valid_node_feature_norm.csv",
        edge_index_csv=f"{data_root_dir}/valid_edge_index.csv",
        node_type_txt=f"{data_root_dir}/valid_node_types.txt",
        edge_ratio_csv=None,
        ignore_weight=ignore_weight,
        include_edge_type=include_edge_type,
        edge_ratio_percentile=None,
        # n_known_abnormal=n_known_abnormal,
        # trace_info_csv=f"{self.data_root_dir}/trace_info.csv",
    )

    dataset_size = valid_dataset.size()

    roc_list = []
    ap_list = []

    model.eval()
    # set validation dataset
    model.dataset = valid_dataset
    for i in tqdm(range(dataset_size)):
        # 1. get node embeddings
        _, _, _node_embed = model([i], train=False, return_node_embed=True)
        _node_embed = _node_embed[0]

        svdd_score = (
            torch.mean(torch.square(_node_embed - model.svdd_center), 1)
            .cpu()
            .detach()
            .numpy()
        )

        _node_labels = valid_dataset.node_feature_df[
            valid_dataset.node_feature_df.trace_id == i
        ]["y"].values

        # 2. filter on0/1 nodes
        _mask = _node_labels <= 1
        node_labels = _node_labels[_mask]
        node_scores = svdd_score[_mask]

        # TODO: Offsetting to have positive number when the label has no positive value
        node_labels = np.append(node_labels, [1])
        node_scores = np.append(node_scores, [1.0])

        # Calc svdd score node level

        # 3. Evaluate on AUC and AP between these

        fpr, tpr, roc_thresholds = roc_curve(node_labels, node_scores)
        precision, recall, pr_thresholds = precision_recall_curve(
            node_labels, node_scores
        )
        roc_auc = auc(fpr, tpr)
        ap = auc(recall, precision)

        roc_list.append(roc_auc)
        ap_list.append(ap)

    avg_auc = sum(roc_list) / len(roc_list)
    avg_ap = sum(ap_list) / len(ap_list)

    print(f"\tAverage AUC:{avg_auc}; Average AP:{avg_ap};")
    print(f"\tMax AUC:{max(roc_list)}; Max AP:{max(ap_list)};")
    print(f"\tMin AUC:{min(roc_list)}; Min AP:{min(ap_list)};")
    return avg_auc, avg_ap, -1, -1
