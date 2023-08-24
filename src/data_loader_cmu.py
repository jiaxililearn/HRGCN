import os
import json
from re import L
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CMUGraphDataset(Dataset):
    def __init__(
        self, data_root_path=None, preprocessing=False, transform=None, **kwargs
    ):
        """
        node_feature_csv: path to the node feature csv file
        het_neigh_root: path to the het neighbour list root dir
        """
        super(CMUGraphDataset, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transform = transform

        self.n_graphs = 600
        self.max_num_edge_embeddings = 12
        self.incoming_node_embedding_size = 26
        self.topk = 12
        self.preprocessing = preprocessing

        self.data_root_path = data_root_path
        if data_root_path is None:
            self.data_root_path = "../custom_data_simple"

        self.relation_types_f = [
            "a_a_list",
            "a_b_list",
            "a_c_list",
            "a_d_list",
            "a_e_list",
            "a_f_list",
            "a_g_list",
            "a_h_list",
            "b_a_list",
            "b_b_list",
            "b_c_list",
            "b_d_list",
            "b_e_list",
            "b_h_list",
        ]

        if preprocessing:
            self.node_features = pd.read_csv(
                f"{self.data_root_path}/incoming_edge_embedding.csv"
            )

            # load up feature matrix based on the relation types
            for relation_id, relation_f in enumerate(self.relation_types_f):
                print(f"Reading Relation Type File: {relation_f}")
                fname = relation_f.split(".")[0]

                graph_edge_embedding = np.zeros(
                    (
                        self.n_graphs,
                        self.max_num_edge_embeddings,
                        self.incoming_node_embedding_size,
                    )
                )

                with open(f"{self.data_root_path}/{relation_f}.txt", "r") as fin:
                    cnt = 0
                    line = fin.readline()
                    current_gid = -1
                    current_src_id = -1
                    i = -1  # aggregate top k src-neighs in the neigh for a graph
                    while line:
                        part_ = line.strip().split(":")
                        gid = int(part_[0])
                        src_id = int(part_[1])
                        neigh_list = [int(i) for i in part_[2].split(",")]

                        # reset counters when a new graph reached
                        if current_gid != gid:
                            print(f"read graph {gid}")
                            i = -1
                            current_src_id = -1
                            current_gid = gid
                            g_node_feature = self.node_features[
                                self.node_features["graph-id"] == gid
                            ]

                        if current_src_id != src_id:
                            i += 1
                            current_src_id = src_id

                        if i >= self.topk:
                            print(
                                f"Skip src-neigh list since limit reached k for graph {gid} node {src_id}: {self.topk}"
                            )
                            line = fin.readline()
                            continue

                        for dst_id in neigh_list:
                            graph_edge_embedding[gid][i] += g_node_feature[
                                g_node_feature["destination-id"] == dst_id
                            ].values[:, 2:][0]

                            cnt += 1
                            if cnt % 10000 == 0:
                                print(f"\tProcessed {cnt} Nodes")

                        line = fin.readline()
                print("Saving to file")
                torch.save(
                    graph_edge_embedding, f"{self.data_root_path}/processed/{fname}.pt"
                )
        else:
            # TODO: read from existing
            self.feature_list = torch.zeros(
                len(self.relation_types_f),
                self.n_graphs,
                self.max_num_edge_embeddings,
                self.incoming_node_embedding_size,
                device=self.device,
            )

            for i, relation_type in enumerate(self.relation_types_f):
                print(f"reading features from {relation_type}")
                features_fpath = f"{self.data_root_path}/processed/{relation_type}.pt"
                features_ = torch.load(features_fpath)
                features_ = torch.from_numpy(features_).float().to(self.device)

                self.feature_list[i] = features_
        print("done")

    # def read_graph(self, gid):
    #     """
    #     read a graph from disk
    #     """
    #     f_path = f'{self.het_neigh_root}/g{gid}.json'
    #     with open(f_path, 'r') as fin:
    #         g_het = json.load(fin)
    #     return g_het

    def __getitem__(self, gid):
        """
        get graph data on graph id
        return: (node_feature, graph_het_feature)
                node_feature: (n_node, n_feature)
                graph_het_feature: (n_neigh, n_node, topk, n_feature)
        """
        return self.feature_list[:, gid, :]


if __name__ == "__main__":
    dataset = CMUGraphDataset()

    print(dataset[0])
    print(dataset[[0, 1]].shape)
    print(dataset[[0]].shape)
