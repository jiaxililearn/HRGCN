from tqdm import tqdm
import torch
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops, degree


class HetGCNConv_8(MessagePassing):
    """
    self implemented
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_node_types,
        hidden_channels=16,
        num_hidden_conv_layers=1,
        num_src_types=2,
        num_edge_types=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_node_types = num_node_types
        self.hidden_channels = hidden_channels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_src_types = num_src_types
        self.num_hidden_conv_layers = num_hidden_conv_layers
        self.num_edge_types = num_edge_types

        # self.k = 12
        # first het node hidden layer
        hidden_conv_layers = []

        for i in range(num_hidden_conv_layers):
            edge_type_layers = []
            for _ in range(self.num_edge_types):
                fc_node_content_layers = []
                for _ in range(self.num_node_types * self.num_src_types):
                    if i == 0:
                        _in_channels = in_channels
                    else:
                        _in_channels = hidden_channels
                    fc_node_content_layers.append(
                        torch.nn.Linear(_in_channels, hidden_channels, bias=True)
                    )
                edge_type_layers.append(torch.nn.ModuleList(fc_node_content_layers))
            hidden_conv_layers.append(torch.nn.ModuleList(edge_type_layers))
        self.hidden_conv_layers = torch.nn.ModuleList(hidden_conv_layers)
        print(self.hidden_conv_layers)
        # hidden_conv_layers[n_hidden][etype][ntype]

        self.fc_het_layer = torch.nn.Linear(
            hidden_channels * num_node_types * num_src_types * num_edge_types,
            out_channels,
            bias=True,
        )

        self.relu = torch.nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters from Layers and Parameters
        """
        for hidden_layers in self.hidden_conv_layers:
            for etypes in hidden_layers:
                for lin in etypes:
                    lin.reset_parameters()
        self.fc_het_layer.reset_parameters()

    def forward(self, graph_data, source_types=[0, 1]):
        """
        forward method
        """
        node_feature, edge_index, (edge_weight, edge_type), node_types = graph_data
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        het_h_embeddings = []
        for ntype in range(self.num_node_types * self.num_src_types):
            for etype in range(self.num_edge_types):
                _, het_edge_index, het_edge_weight = self.get_het_edge_index(
                    edge_index,
                    edge_weight,
                    node_types,
                    ntype,
                    source_types=source_types,
                    edge_type_list=edge_type,
                    edge_type=etype,
                )

                if het_edge_index is None:
                    content_h = torch.zeros(
                        node_feature.shape[0],
                        self.hidden_channels,
                        device=edge_index.device,
                    )
                else:
                    het_edge_index, het_edge_weight = self._norm(
                        het_edge_index,
                        size=node_feature.size(0),
                        edge_weight=het_edge_weight,
                    )

                    content_h = self.hidden_conv_layers[0][etype][ntype](node_feature)
                    content_h = self.propagate(
                        het_edge_index, x=content_h, edge_weight=het_edge_weight
                    )
                    content_h = self.relu(content_h)

                    for i in range(1, self.num_hidden_conv_layers):
                        # print(f'hidden {i}')
                        content_h = self.hidden_conv_layers[i][etype][ntype](content_h)
                        content_h = self.propagate(
                            het_edge_index, x=content_h, edge_weight=het_edge_weight
                        )
                        content_h = self.relu(content_h)

                het_h_embeddings.append(content_h)

        combined_het_embedding = torch.cat(het_h_embeddings, 1).view(
            node_feature.shape[0],
            self.hidden_channels
            * self.num_node_types
            * self.num_src_types
            * self.num_edge_types,
        )

        # max pooling
        combined_het_embedding, _ = torch.max(combined_het_embedding, dim=0)
        # print(f'combined_het_embedding shape: {combined_het_embedding.shape}')
        # print(f'combined_het_embedding: {combined_het_embedding}')

        out = self.fc_het_layer(combined_het_embedding.view(1, -1))
        # print(f'out shape: {out.shape}')
        return out

    def get_het_edge_index(
        self,
        edge_index,
        edge_weight,
        node_types,
        ntype,
        source_types=None,
        edge_type_list=None,
        edge_type=None,
    ):
        """
        get het edge index by given type
        """
        row, col = edge_index

        if source_types is not None:
            try:
                num_src_types = len(source_types)
                src_type_idx = int(ntype / self.num_node_types)
                dst_type = ntype - self.num_node_types * src_type_idx
                src_type = source_types[src_type_idx]

                if len(node_types[dst_type]) == 0 or len(node_types[src_type]) == 0:
                    return ntype, None, None

                # TODO: handle edge type

                src_het_mask = sum(row == i for i in node_types[src_type]).bool()
                dst_het_mask = sum(col == i for i in node_types[dst_type]).bool()

                if edge_type is not None and edge_type_list is not None:
                    edge_mask = edge_type_list == edge_type
                    cmask = src_het_mask & dst_het_mask & edge_mask
                else:
                    cmask = src_het_mask & dst_het_mask
            except Exception as e:
                print(f"{src_type_idx} - {dst_type}")
                print(f"row: {row}")
                print(f"node_types[src_type]: {node_types[src_type]}")
                raise Exception(e)
            return ntype, torch.stack([row[cmask], col[cmask]]), edge_weight[cmask]
        else:
            if len(node_types[ntype]) == 0:
                return ntype, None, None

            het_mask = sum(col == i for i in node_types[ntype]).bool()
            return (
                ntype,
                torch.stack([row[het_mask], col[het_mask]]),
                edge_weight[het_mask],
            )

    def het_edge_index(self, edge_index, edge_weight, node_types):
        """
        return a generator of het neighbour edge indices
        """
        row, col = edge_index
        for ntype, n_list in enumerate(node_types):
            # print(f'col: {col}')
            # print(f'n_list: {n_list}')

            if len(n_list) == 0:
                yield ntype, None, None
                continue
            # TODO: look into the masking shape of the results
            het_mask = sum(col == i for i in n_list).bool()
            # print(f'het mask: {het_mask}')

            yield ntype, torch.stack([row[het_mask], col[het_mask]]), edge_weight[
                het_mask
            ]

    def _norm(self, edge_index, size, edge_weight=None, flow="source_to_target"):
        assert flow in ["source_to_target", "target_to_source"]

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_attr=edge_weight, num_nodes=size
        )

        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

        row, col = edge_index
        if flow == "source_to_target":
            deg = scatter_add(edge_weight, col, dim=0, dim_size=size)
        else:
            deg = scatter_add(edge_weight, row, dim=0, dim_size=size)

        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)

        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    def message(self, x_j, edge_weight):
        # x_j has shape [num_edges, out_channels]
        return edge_weight.view(-1, 1) * x_j

    def update(self, inputs):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        return inputs
