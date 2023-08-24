# from tqdm import tqdm
import numpy as np
from enum import unique
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from HRGCNConv import HRGCNConv

# from graph_augmentation import GraphAugmentator


class HRGCN(nn.Module):
    def __init__(
        self,
        model_path=None,
        dataset=None,
        source_types=None,
        feature_size=7,
        out_embed_s=32,
        random_seed=32,
        num_node_types=7,
        hidden_channels=16,
        num_hidden_conv_layers=1,
        model_sub_version=0,
        num_edge_types=1,
        augment_func=None,
        main_loss=None,
        embed_activation="relu",
        # batch_n_known_abnormal=None,
        weighted_loss=None,
        loss_weight=0.5,
        eval_method="both",
        ablation=None,
        # edge_perturbation_p=0.002,
        **kwargs,
    ):
        """
        Het GCN based on MessagePassing
            + segragation of the source neighbour type
            + relational edge type

        Adding Graph Augmentation Methods
            + Edge Perturbation (add/removing edges comply with Het characteristics)
        """
        super().__init__()
        torch.manual_seed(random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svdd_center = None
        self.model_path = model_path
        self.dataset = dataset
        self.source_types = source_types
        self.model_sub_version = model_sub_version

        # self.edge_perturbation_p = edge_perturbation_p

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s

        self.num_node_types = num_node_types

        # self.subgraph_ratio = subgraph_ratio
        # self.insertion_iteration = insertion_iteration
        # self.augmentation_method = augmentation_method
        # self.augmentation_func = augmentations()[augmentation_method]
        self.augment_func = augment_func

        self.weighted_loss = weighted_loss
        self.main_loss = main_loss
        self.loss_weight = loss_weight
        self.eval_method = eval_method
        # self.batch_n_known_abnormal = batch_n_known_abnormal

        self.eps = 1e-6
        self.eta = 1.0  # TODO: hyperparameter

        # node feature content encoder
        if model_sub_version == 0:
            self.het_node_conv = HRGCNConv(
                self.embed_d,
                self.out_embed_d,
                self.num_node_types,
                hidden_channels=hidden_channels,
                num_hidden_conv_layers=num_hidden_conv_layers,
                num_src_types=len(source_types),
                num_edge_types=num_edge_types,
                ablation=ablation,
            )

        else:
            pass

        print(f"num_hidden_conv_layers: {num_hidden_conv_layers}")

        self.final_fc = nn.Sequential(
            nn.Linear(self.out_embed_d, 1, bias=True), nn.Sigmoid()
        )

        # Others
        if embed_activation == "relu":
            self.embed_act = nn.LeakyReLU()
        elif embed_activation == "sigmoid":
            self.embed_act = nn.Sigmoid()

        # loss
        if self.weighted_loss == "bce":
            print("using bce loss")
            self.wloss = torch.nn.BCELoss()
        elif self.weighted_loss == "deviation":
            print("using deviation loss")
            self.wloss = self.deviation_loss
        elif self.weighted_loss == "ignore":
            print("using no weighted loss")
            self.wloss = None

    def init_weights(self):
        """
        init weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, gid_batch, train=True):
        """
        forward propagate based on gid batch
        """
        batch_data = [self.dataset[i] for i in gid_batch]
        # print(f'batch_data: {batch_data}')
        if train:
            # print(f"{self.augment_func.__name__} for the batch ..")
            # het_edge_perturbation(args)
            if self.wloss is not None:
                synthetic_data, synthetic_method = self.augment_func(batch_data)
            else:
                synthetic_data, synthetic_method = [], []

            # if self.main_loss == "semi-svdd":
            #     abnormal_data = np.random.choice(
            #         self.dataset.known_attack_gid_list,
            #         # self.batch_n_known_abnormal,
            #         False,
            #     )
            #     abnormal_data = [self.dataset[i] for i in abnormal_data]
            # else:  # ie. svdd
            #     abnormal_data = []

            combined_data = batch_data + synthetic_data  # + abnormal_data

            if self.main_loss == "semi-svdd":
                combined_labels = (
                    torch.tensor([1 for _ in batch_data] + [-1 for _ in synthetic_data])
                    .to(self.device)
                    .view(-1, 1)
                )
            else:
                combined_labels = (
                    torch.tensor([0 for _ in batch_data] + [1 for _ in synthetic_data])
                    .to(self.device)
                    .view(-1, 1)
                )

            # ga_methods. 0 - known normal graph, -1 - known abnormal graph, others - ga
            combined_methods = (
                torch.tensor([0 for _ in batch_data] + synthetic_method)
                .to(self.device)
                .view(-1, 1)
            )
        else:
            combined_data = batch_data
            combined_labels = (
                torch.tensor([0 for _ in batch_data]).to(self.device).view(-1, 1)
            )
            combined_methods = (
                torch.tensor([0 for _ in batch_data]).to(self.device).view(-1, 1)
            )

        # print(f'x_node_feature shape: {x_node_feature.shape}')
        # print(f'x_edge_index shape: {x_edge_index.shape}')
        _out = torch.zeros(len(combined_data), 1, device=self.device)
        if self.main_loss == "semi-svdd":
            _out_h = torch.zeros(
                len(combined_data), self.out_embed_d, device=self.device
            )
        else:
            _out_h = torch.zeros(len(gid_batch), self.out_embed_d, device=self.device)
        # print(f'size: {len(combined_data)}')
        for i, (g_data, g_label) in enumerate(zip(combined_data, combined_labels)):
            # print(f'trace_id: {gid_batch[i]}')
            # print(f'g_data: {g_data}')
            # print(f'self.dataset[i]: {self.dataset[806]}')

            h = self.het_node_conv(g_data, source_types=self.source_types)
            h = self.embed_act(h)

            # print(f'h: {h}')
            if self.main_loss == "semi-svdd":
                _out_h[i] = h
            else:
                if g_label == 0:
                    _out_h[i] = h

            h = self.final_fc(h)
            _out[i] = h
        # print(f'combined_labels: {combined_labels.shape}')
        # print(f'_out: {_out.shape}')
        # print(f'combined_labels: {combined_labels}')

        return _out, (combined_labels, combined_methods), _out_h

    # def graph_node_pooling(self, graph_node_het_embedding):
    #     """
    #     average all the node het embedding
    #     """
    #     if graph_node_het_embedding.shape[0] == 1:
    #         return graph_node_het_embedding
    #     return torch.mean(graph_node_het_embedding, 0)

    def set_svdd_center(self, center):
        """
        set svdd center
        """
        self.svdd_center = center

    def load_svdd_center(self):
        """
        load existing svdd center
        """
        self.set_svdd_center(
            torch.load(
                f"{self.model_path}/HetGNN_SVDD_Center.pt", map_location=self.device
            )
        )

    def load_checkpoint(self, checkpoint):
        """
        load model checkpoint
        """
        checkpoint_model_path = f"{self.model_path}/HetGNN_{checkpoint}.pt"
        self.load_state_dict(
            torch.load(checkpoint_model_path, map_location=self.device)
        )

    def predict_score(self, g_data, verbose=False):
        """
        calc dist given graph features
        """

        with torch.no_grad():
            bce_scores, _, embed = self(g_data, train=False)
            svdd_score = torch.mean(torch.square(embed - self.svdd_center), 1)
            if self.eval_method == "svdd" or self.loss_weight == 0.0:
                scores = svdd_score
            elif self.eval_method == "bce":
                scores = bce_scores
            elif self.eval_method == "both":
                scores = bce_scores.view(
                    -1,
                ) * svdd_score.view(
                    -1,
                )
        if verbose:
            return svdd_score, bce_scores
        return scores, bce_scores

    def svdd_cross_entropy_loss(
        self, embed_batch, outputs, labels, ga_methods, l2_lambda=0.001, fix_center=True
    ):
        """
        Compute combination of SVDD Loss and cross entropy loss on batch

        """

        _batch_out = embed_batch
        _batch_out_resahpe = _batch_out.view(
            _batch_out.size()[0] * _batch_out.size()[1], self.out_embed_d
        )

        if fix_center:
            if self.svdd_center is None:
                with torch.no_grad():
                    print("Set initial center ..")
                    hypersphere_center = torch.mean(_batch_out_resahpe, 0)
                    self.set_svdd_center(hypersphere_center)
                    torch.save(
                        hypersphere_center, f"{self.model_path}/HetGNN_SVDD_Center.pt"
                    )
            else:
                hypersphere_center = self.svdd_center
                #  with torch.no_grad():
                #     hypersphere_center = (model.svdd_center + torch.mean(_batch_out_resahpe, 0)) / 2
                #     model.set_svdd_center(hypersphere_center)
        else:
            with torch.no_grad():
                print("compute batch center ..")
                hypersphere_center = torch.mean(_batch_out_resahpe, 0)

        dist = torch.sum(torch.square(_batch_out_resahpe - hypersphere_center), 1)

        if self.main_loss == "semi-svdd" and self.wloss is not None:
            print("calc semi-svdd ..")
            dist = torch.where(
                labels == 0, dist, self.eta * ((dist + self.eps) ** labels.float())
            )

        loss_ = torch.mean(dist)

        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters()) / 2

        svdd_loss = loss_ + l2_lambda * l2_norm

        if self.wloss is None:
            print(f"\t Batch SVDD Loss: {svdd_loss};")
            return svdd_loss

        ga_losses = {}
        weighted_loss = 0.0
        # print(f'labels: {labels}')
        if self.main_loss == "semi-svdd":
            # print('convert semi-svdd labels..')
            supervised_labels = torch.where(labels > 0, 0.0, 1.0)
        else:
            supervised_labels = labels

        if self.wloss is not None:
            for ga_method in ga_methods.unique():
                if ga_method == 0:  # 0 if input batch data
                    continue
                ga_mask = ga_methods == ga_method
                ga_batch_mask = ga_methods == 0
                n_ = ga_mask.sum()
                ga_outputs = torch.cat([outputs[ga_mask], outputs[ga_batch_mask][:n_]])
                ga_labels = torch.cat(
                    [supervised_labels[ga_mask], supervised_labels[ga_batch_mask][:n_]]
                )

                # print(f'ga_outputs: {ga_outputs}')
                # print(f'ga_labels: {ga_labels}')

                ga_weighted_loss = self.wloss(ga_outputs, ga_labels)

                # print(f'ga_weighted_loss: {ga_weighted_loss}')

                ga_losses[ga_method.item()] = ga_weighted_loss.item()
                weighted_loss += ga_weighted_loss

            # TODO: individual loss weights for different GA methods
            # weighted_loss = torch.tensor(list(ga_losses.values()), dtype=torch.float).flatten().to(self.device).sum()

        print(f"\t\t GA Method Loss: {ga_losses}")
        print(f"\t Batch SVDD Loss: {svdd_loss}; Batch Weighted Loss: {weighted_loss};")

        loss = svdd_loss + weighted_loss * self.loss_weight
        return loss

    def deviation_loss(self, y_pred, y_true):
        """
        z-score-based deviation loss
        """
        confidence_margin = 5.0
        ## size=5000 is the setting of l in algorithm 1 in the paper
        ref = torch.tensor(
            np.random.normal(loc=0.0, scale=1.0, size=5000), dtype=torch.float32
        )
        dev = (y_pred - torch.mean(ref)) / torch.std(ref)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs(
            torch.maximum(confidence_margin - dev, torch.tensor(0.0))
        )
        # print(f'dev: {dev}')
        # print(f'inlier_loss: {inlier_loss}')
        # print(f'outlier_loss: {outlier_loss}')
        # print(f'y_true: {y_true}')
        # print(f'y_true: {y_pred}')
        return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)
