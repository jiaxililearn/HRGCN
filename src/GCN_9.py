# from tqdm import tqdm
import torch
from torch import nn
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


class HetGCN_9(nn.Module):
    def __init__(
        self,
        model_path=None,
        dataset=None,
        source_types=None,
        feature_size=7,
        out_embed_s=32,
        random_seed=32,
        hidden_channels=16,
        model_sub_version=0,
        num_edge_types=1,
        **kwargs,
    ):
        """
        DeepTraLog Paper Baseline implemented
        """
        super().__init__()
        torch.manual_seed(random_seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.svdd_center = None
        self.model_path = model_path
        self.dataset = dataset
        self.source_types = source_types
        self.model_sub_version = model_sub_version

        self.embed_d = feature_size
        self.out_embed_d = out_embed_s
        self.hidden_channels = hidden_channels

        self.num_edge_types = num_edge_types

        # node feature content encoder
        if model_sub_version == 0:
            in_fcs = []
            out_fcs = []
            for _ in range(self.num_edge_types):
                in_fcs.append(nn.Linear(self.embed_d, self.hidden_channels))
                out_fcs.append(nn.Linear(self.embed_d, self.hidden_channels))
            self.in_fcs = torch.nn.ModuleList(in_fcs)
            self.out_fcs = torch.nn.ModuleList(out_fcs)

            self.reset_gate = nn.Sequential(
                nn.Linear(self.hidden_channels * 2, self.hidden_channels), nn.Sigmoid()
            )
            self.update_gate = nn.Sequential(
                nn.Linear(self.hidden_channels * 2, self.hidden_channels), nn.Sigmoid()
            )
            self.tansform = nn.Sequential(
                nn.Linear(self.hidden_channels * 3, self.hidden_channels), nn.Tanh()
            )

            self.out = nn.Sequential(
                nn.Linear(self.hidden_channels, self.out_embed_d), nn.Tanh()
            )

        else:
            pass

        # Others
        self.relu = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        """
        init weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                # nn.init.xavier_uniform_(m.weight)
                # nn.init.xavier_normal_(m.weight)
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, gid_batch):
        """
        forward propagate based on gid batch
        """
        batch_data = [self.dataset[i] for i in gid_batch]

        # print(f'x_node_feature shape: {x_node_feature.shape}')
        # print(f'x_edge_index shape: {x_edge_index.shape}')
        _out = torch.zeros(len(gid_batch), self.out_embed_d, device=self.device)
        for i, g_data in enumerate(batch_data):
            node_feature, edge_index, (_, edge_type), _ = g_data

            a_stack = []
            for etype in range(self.num_edge_types):
                edge_type_index = self.get_edge_index_by_edege_type(
                    edge_index, edge_type, etype
                )

                in_state = self.in_fcs[etype](node_feature)
                out_state = self.out_fcs[etype](node_feature)
                # print(f'in_state shape: {in_state.shape}')
                # print(f'out_state shape: {out_state.shape}')

                edge_type_index, _ = add_remaining_self_loops(edge_index)

                a_in = scatter_add(
                    torch.index_select(in_state, 0, edge_type_index[1]),
                    edge_type_index[0],
                    0,
                )
                a_out = scatter_add(
                    torch.index_select(out_state, 0, edge_type_index[0]),
                    edge_type_index[1],
                    0,
                )
                # print(f'a_in shape: {a_in.shape}')
                # print(f'a_out shape: {a_out.shape}')

                a_cat = torch.cat((a_in, a_out), 1)
                # print(f'a_cat shape: {a_cat.shape}')

                a_stack.append(a_cat)

            a_stack = torch.stack(a_stack).view(
                self.num_edge_types * a_cat.shape[0], a_cat.shape[1]
            )
            # print(f'a_stack shape: {a_stack.shape}')

            r = self.reset_gate(a_stack)
            z = self.update_gate(a_stack)

            joined_input = torch.cat((a_stack, r), 1)
            # print(f'joined_input shape: {joined_input.shape}')

            h_hat = self.tansform(joined_input)
            prop_output = (1 - z) + z * h_hat

            output = self.out(prop_output)
            output = output.sum(0)
            # print(f'output shape: {output.shape}')

            _out[i] = output
        return _out

    def get_edge_index_by_edege_type(self, edge_index, edge_type_list, etype):
        row, col = edge_index
        edge_mask = edge_type_list == etype
        return torch.stack([row[edge_mask], col[edge_mask]])

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

    def predict_score(self, g_data):
        """
        calc dist given graph features
        """
        with torch.no_grad():
            _out = self(g_data)
            score = torch.mean(torch.square(_out - self.svdd_center), 1)  # mean on rows
        return score


def svdd_batch_loss(model, embed_batch, l2_lambda=0.001, fix_center=True):
    """
    Compute SVDD Loss on batch
    """
    # TODO
    out_embed_d = model.out_embed_d

    _batch_out = embed_batch
    _batch_out_resahpe = _batch_out.view(
        _batch_out.size()[0] * _batch_out.size()[1], out_embed_d
    )

    if fix_center:
        if model.svdd_center is None:
            with torch.no_grad():
                print("Set initial center ..")
                hypersphere_center = torch.mean(_batch_out_resahpe, 0)
                model.set_svdd_center(hypersphere_center)
                torch.save(
                    hypersphere_center, f"{model.model_path}/HetGNN_SVDD_Center.pt"
                )
        else:
            hypersphere_center = model.svdd_center
            #  with torch.no_grad():
            #     hypersphere_center = (model.svdd_center + torch.mean(_batch_out_resahpe, 0)) / 2
            #     model.set_svdd_center(hypersphere_center)
    else:
        with torch.no_grad():
            print("compute batch center ..")
            hypersphere_center = torch.mean(_batch_out_resahpe, 0)

    dist = torch.square(_batch_out_resahpe - hypersphere_center)
    loss_ = torch.mean(torch.sum(dist, 1))

    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    loss = loss_ + l2_lambda * l2_norm * 0.5
    return loss
