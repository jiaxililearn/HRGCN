import random
import copy
import math
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, k_hop_subgraph

# RuntimeError: CUDA error: device-side assert triggered
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.


class GraphAugmentator:
    def __init__(
        self,
        num_node_types=8,
        num_edge_types=1,
        edge_perturbation_method="xor",
        prior_dist=None,
        subgraph_ratio=0.01,
        insertion_iteration=1,
        node_insertion_method="target_to_source",
        swap_edge_pct=0.05,
        swap_node_pct=0.05,
        edge_mutate_prob=None,
        add_method="rare",
        edge_addition_pct=0.1,
        replace_edges=False,
        **kwarg,
    ):
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.edge_perturbation_method = edge_perturbation_method
        self.prior_dist = prior_dist
        self.edge_mutate_prob = edge_mutate_prob

        self.subgraph_ratio = subgraph_ratio

        self.insertion_iteration = insertion_iteration
        self.node_insertion_method = node_insertion_method

        self.swap_node_pct = swap_node_pct
        self.swap_edge_pct = swap_edge_pct

        self.add_method = add_method
        self.edge_addition_pct = edge_addition_pct
        self.replace_edges = replace_edges

        self.func_list = [self.create_edge_addition, self.create_node_type_swap]
        if edge_mutate_prob is not None:
            self.func_list.append(self.create_het_edge_perturbation)

        # skip edge swap if only 1 edge type
        if num_edge_types > 1:
            self.func_list.append(self.create_edge_type_swap)

        self.ga_dict = {
            self.create_het_edge_perturbation.__name__: 1,
            self.create_edge_addition.__name__: 2,
            self.create_node_type_swap.__name__: 3,
            self.create_edge_type_swap.__name__: 4,
        }

    def get_augment_func(self, augmentation_method):
        """
        getting augmentation functions
        """
        augment_func_dict = {
            "node_insertion": self.create_het_node_insertion,
            "edge_perturbation": self.create_het_edge_perturbation,
            "all": self.create_all_augmentations,
        }
        return augment_func_dict[augmentation_method]

    def create_all_augmentations(self, batch_data):
        """
        create all augmentations at random
        """
        new_batch = []
        new_batch_ga = []
        # for i in range(len(batch_data)):
        i = 0
        while len(new_batch) < len(batch_data):
            g_data = random.choice(batch_data)
            func = random.choice(self.func_list)
            # print(f'+ Apply GA Method: {func.__name__} ..')
            new_g_data = func([g_data])
            if len(new_g_data) > 0:
                new_batch.extend(new_g_data)
                new_batch_ga.append(self.ga_dict[func.__name__])
                # TODO: adding ga dict for other GA methods
            i += 1

            if i % (len(batch_data) * 10) == 0:
                print(f"Tried to produce augmentation {i} times.")
        return new_batch, new_batch_ga

    def create_het_edge_perturbation(self, batch_data):
        """
        generate the same number of 'synthetic' abnormal graphs as the input batch
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_batch.append(
                self.het_edge_perturbation_from_prior(
                    g_data,
                    self.prior_dist,
                    num_node_types=self.num_node_types,
                    method=self.edge_perturbation_method,
                )
            )
        return new_batch

    def het_edge_perturbation_from_prior(
        self, g_data, prior_dist, num_node_types=8, method="xor", size=None
    ):
        """
        create edge perturbation based on prior distributions
        """
        node_feature, edge_index, (edge_weight, edge_type), node_types = g_data
        device = "cpu"  # edge_index.device

        if size is None:
            size = node_feature.shape[0]

        total_num_edges = edge_index.shape[1]
        num_edge_types = len(prior_dist.keys())

        new_edge_index = []
        new_edge_type = []

        for etype in range(num_edge_types):
            for src_type in range(num_node_types):
                for dst_type in range(num_node_types):
                    src_dst_type = f"{src_type}_{dst_type}"

                    try:
                        edge_ratio = prior_dist[etype][src_dst_type]
                    except Exception as e:
                        # print(f"could not found key {src_dst_type} in edge type {etype}")
                        continue

                    src_node_list = node_types[src_type]
                    dst_node_list = node_types[dst_type]

                    if len(src_node_list) == 0 or len(dst_node_list) == 0:
                        continue
                    # print(f"src_node_list: {src_node_list}")
                    # print(f"dst_node_list: {dst_node_list}")

                    num_edges = int(edge_ratio * total_num_edges) + 1

                    sampled_edge_index = torch.tensor(
                        [
                            random.choices(src_node_list, k=num_edges),
                            random.choices(dst_node_list, k=num_edges),
                        ]
                    )
                    sampled_edge_type = torch.tensor([etype] * num_edges)

                    new_edge_index.append(sampled_edge_index)
                    new_edge_type.append(sampled_edge_type)

        new_edge_index = torch.cat(new_edge_index, dim=1).long().view(2, -1)
        new_edge_type = (
            torch.cat(new_edge_type)
            .int()
            .view(
                -1,
            )
        )

        # print(f'new_edge_index: {new_edge_index.shape}')
        # print(f'new_edge_type: {new_edge_type.shape}')
        # print(f'num_edge_types: {num_edge_types}')

        generated_adj_matrix = to_dense_adj(
            new_edge_index, edge_attr=new_edge_type + 1, max_num_nodes=size
        ).view(size, -1)

        origin_adj_matrix = to_dense_adj(
            edge_index.cpu(), edge_attr=edge_type.cpu() + 1, max_num_nodes=size
        ).view(size, -1)

        # create a probability matrix to decide if permute the edge
        m = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([self.edge_mutate_prob])
        )
        edge_mutate_mask = m.sample((size, size))
        # print(f'generated_adj_matrix: {generated_adj_matrix.shape}')
        # print(f'edge_mutate_mask: {edge_mutate_mask.shape}')
        masked_generated_adj_matrix = generated_adj_matrix * edge_mutate_mask
        # masked_generated_adj_matrix = masked_generated_adj_matrix.to(device)

        mask = torch.logical_xor(origin_adj_matrix, masked_generated_adj_matrix)

        new_adj_matrix = origin_adj_matrix.masked_fill(
            ~mask, 0
        ) + masked_generated_adj_matrix.masked_fill(~mask, 0)

        new_edge_index, new_edge_type = dense_to_sparse(new_adj_matrix)
        new_edge_type -= 1
        return (
            node_feature,
            new_edge_index.to(edge_index.device),
            (None, new_edge_type.to(edge_index.device)),
            node_types,
        )  # ignores edge weight for now. TODO: need to add support for CMU data. CMU data skips edge perturbation

    def create_het_node_insertion(self, batch_data):
        """
        create node insertion
        TODO: Add node deletion
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)

            for iter_ in range(self.insertion_iteration):
                g_data = self.het_node_insertion(
                    g_data,
                    subgraph_ratio=self.subgraph_ratio,
                    method=self.node_insertion_method,
                )
            new_batch.append(g_data)
        return new_batch

    def het_node_insertion(
        self,
        g_data,
        subgraph_ratio=0.01,
        method="target_to_source",
        size=None,
    ):
        """
        add / removing node from the graphs
        It will also remove/add edges to the graph with the associated node
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        device = edge_index.device

        sampled_nodes = random.sample(
            range(node_features.shape[0]),
            int(node_features.shape[0] * subgraph_ratio) + 1,
        )

        try:
            _, sub_edge_index, _, sub_edge_mask = k_hop_subgraph(
                node_idx=sampled_nodes,
                num_hops=1,
                edge_index=edge_index,
                flow=method,
            )
        except Exception as e:
            print(f"sampled_nodes: {sampled_nodes}")
            raise Exception from e

        row, col = edge_index
        retained_edges = torch.stack([row[~sub_edge_mask], col[~sub_edge_mask]]).to(
            edge_index.device
        )
        retained_edge_types = edge_type[~sub_edge_mask]

        new_node_list = []

        # rewiring edge to new node
        last_node_id = node_features.shape[0] - 1
        for ntype, ntype_list in enumerate(node_types):
            if len(ntype_list) == 0:
                # print(f"skip node type {ntype}")
                continue
            _mask = sum(sub_edge_index[1] == i for i in ntype_list).bool()

            # skip if no node matched
            if _mask.sum() == 0:
                # print("skip mask")
                continue

            new_node_id = last_node_id + 1
            new_node_list.append((new_node_id, ntype))
            sub_edge_index[1] = sub_edge_index[1].masked_fill_(_mask, new_node_id)

            last_node_id = new_node_id
        new_edge_index = torch.cat([retained_edges, sub_edge_index], dim=1)

        # add new node to node feature matrix and node type list
        new_node_types = copy.deepcopy(node_types)
        new_node_features = [node_features]
        for new_node_id, new_node_type in new_node_list:
            new_node_feature = node_features[
                sample_one_node_from_list(node_types[new_node_type])
            ].view(1, -1)
            new_node_types[new_node_type].append(new_node_id)

            new_node_features.append(new_node_feature)
        new_node_features = torch.cat(new_node_features, dim=0)

        # add new edge type to the existing
        added_edge_types = edge_type[sub_edge_mask]
        new_edge_types = torch.cat([retained_edge_types, added_edge_types])

        return (
            new_node_features,
            new_edge_index,
            (None, new_edge_types),
            new_node_types,
        )  # default edge weight to None. TODO: update for CMU dataset. Skips node insertion

    def create_edge_addition(self, batch_data):
        """
        create random edge addition
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_batch.append(
                self.edge_addition(
                    g_data,
                    edge_addition_pct=self.edge_addition_pct,
                    replace_edges=self.replace_edges,
                    add_method=self.add_method,
                )
            )
        return new_batch

    def edge_addition(
        self, g_data, edge_addition_pct, replace_edges, add_method="rare"
    ):
        """
        Edge addition based on different method
        """
        if add_method == "rare":
            return self.edge_addition_with_rare_freq(
                g_data, edge_addition_pct, replace_edges
            )
        elif add_method == "simple":
            return self.edge_addition_simple(g_data, edge_addition_pct, replace_edges)

    def edge_addition_simple(self, g_data, edge_addition_pct, replace_edges=False):
        """
        add edge to the graph
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data

        device = "cpu"  # edge_index.device
        size = node_features.shape[0]
        origin_adj_matrix = to_dense_adj(
            edge_index.cpu(), edge_attr=edge_type.cpu() + 1, max_num_nodes=size
        ).view(size, -1)

        row, col = edge_index.cpu()
        add_edge_types = []
        add_edge_index = []
        for src_type, src_type_list in enumerate(node_types):
            for dst_type, dst_type_list in enumerate(node_types):
                if len(src_type_list) == 0 or len(dst_type_list) == 0:
                    continue

                src_mask = sum(row == i for i in src_type_list).bool()
                dst_mask = sum(col == i for i in dst_type_list).bool()
                edge_mask = src_mask & dst_mask
                _num_edges = edge_mask.sum()

                # skip when no edges in this src_dst types
                if _num_edges == 0:
                    continue

                num_add = int(_num_edges * edge_addition_pct) + 1
                _edge_types = edge_type.cpu()[edge_mask].unique()

                _add_edge_types = random.choices(_edge_types, k=num_add)
                _add_src_node_id = random.choices(src_type_list, k=num_add)
                _add_dst_node_id = random.choices(dst_type_list, k=num_add)

                _add_edge_index = torch.tensor([_add_src_node_id, _add_dst_node_id])

                add_edge_types.append(torch.tensor(_add_edge_types))
                add_edge_index.append(_add_edge_index)

        add_edge_index = torch.cat(add_edge_index, dim=1)
        add_edge_types = torch.cat(add_edge_types)

        add_adj_matrix = (
            to_dense_adj(
                add_edge_index, edge_attr=add_edge_types + 1, max_num_nodes=size
            )
            .view(size, -1)
            .to(device)
        )

        if replace_edges:
            # TODO: remove original edge:
            add_src = add_edge_index[0]
            for _src_id in add_src:
                _src_node_neigh = origin_adj_matrix[_src_id]
                _dst_node_ids = _src_node_neigh.nonzero().flatten()

                if _dst_node_ids.shape[0] == 0:
                    continue

                # remove a random edge from this neighbourhood
                offset_dst_idx = random.choice(_dst_node_ids)
                _src_node_neigh[offset_dst_idx] = 0

        # resolve duplicated edges
        xor_mask = torch.logical_xor(origin_adj_matrix, add_adj_matrix)

        new_adj_matrix = add_adj_matrix.masked_fill(~xor_mask, 0) + origin_adj_matrix

        new_edge_index, new_edge_type = dense_to_sparse(new_adj_matrix)
        new_edge_type -= 1

        # TODO: default edge weight to None. need to add for CMU dataset like the below rare_freq edge addition method
        return (
            node_features,
            new_edge_index.to(edge_index.device),
            (None, new_edge_type.to(edge_index.device)),
            node_types,
        )

    def edge_addition_with_rare_freq(
        self, g_data, edge_addition_pct, replace_edges=False
    ):
        """
        TODO: Add new edges between nodes based on rare, common possibility
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        device = "cpu"  # edge_index.device

        size = node_features.shape[0]
        num_edges = edge_index.shape[1]

        num_sample = math.ceil(num_edges * edge_addition_pct)

        origin_adj_matrix = to_dense_adj(
            edge_index.cpu(), edge_attr=edge_type.cpu() + 1, max_num_nodes=size
        ).view(size, -1)

        # define weight matrix
        if edge_weight is not None:
            origin_weight_matrix = to_dense_adj(
                edge_index.cpu(), edge_attr=edge_weight.cpu(), max_num_nodes=size
            ).view(size, -1)
            edge_weight_dict = {}

        row, col = edge_index
        edge_prob = {}
        new_edge_weights = None

        # count number of edges for every type
        for etype in range(self.num_edge_types):
            for src_type in range(self.num_node_types):
                for dst_type in range(self.num_node_types):
                    if len(node_types[dst_type]) == 0 or len(node_types[src_type]) == 0:
                        continue

                    src_het_mask = sum(row == i for i in node_types[src_type]).bool()
                    dst_het_mask = sum(col == i for i in node_types[dst_type]).bool()
                    edge_mask = edge_type == etype
                    cmask = src_het_mask & dst_het_mask & edge_mask
                    _num_edges = cmask.sum().item()

                    if _num_edges == 0:
                        continue

                    edge_prob[(etype, src_type, dst_type)] = (
                        1 / _num_edges if _num_edges > 0 else 0.0
                    )
                    # get a list of existing weights in this type
                    if edge_weight is not None:
                        _unique_edge_weights, _unique_edge_weight_cnt = torch.unique(
                            edge_weight[cmask], return_counts=True
                        )
                        _, idx = _unique_edge_weight_cnt.sort(descending=True)
                        val, _ = _unique_edge_weight_cnt.sort(descending=False)
                        _unique_edge_weight_cnt_reversed = val[idx]
                        edge_weight_dict[(etype, src_type, dst_type)] = (
                            _unique_edge_weights,
                            _unique_edge_weight_cnt_reversed,
                        )

        sum_all = sum(edge_prob.values())
        for k, v in edge_prob.items():
            edge_prob[k] = v / sum_all

        sample_edge_list = random.choices(
            list(edge_prob.keys()), weights=list(edge_prob.values()), k=num_sample
        )

        src_id_list = []
        dst_id_list = []
        add_edge_types = []
        if edge_weight is not None:
            add_edge_weights = []

        # create potential edges as well as necessary edge weights and edge types
        for etype, src_type, dst_type in sample_edge_list:
            src_id = random.choice(node_types[src_type])
            dst_id = random.choice(node_types[dst_type])

            add_edge_types.append(etype)
            src_id_list.append(src_id)
            dst_id_list.append(dst_id)

            if edge_weight is not None:
                weights_, weight_probs_ = edge_weight_dict[(etype, src_type, dst_type)]
                weight_ = random.choices(weights_, weights=weight_probs_, k=1)[0]
                add_edge_weights.append(weight_)

        add_edge_index = torch.tensor([src_id_list, dst_id_list])
        add_edge_types = torch.tensor(add_edge_types)

        if edge_weight is not None:
            add_edge_weights = torch.tensor(add_edge_weights)

        add_adj_matrix = (
            to_dense_adj(
                add_edge_index, edge_attr=add_edge_types + 1, max_num_nodes=size
            )
            .view(size, -1)
            .to(device)
        )

        if edge_weight is not None:
            add_weights_matrix = (
                to_dense_adj(
                    add_edge_index, edge_attr=add_edge_weights, max_num_nodes=size
                )
                .view(size, -1)
                .to(device)
            )

        if replace_edges:
            # TODO: remove original edge:
            add_src = add_edge_index[0]
            for _src_id in add_src:
                _src_node_neigh = origin_adj_matrix[_src_id]
                _dst_node_ids = _src_node_neigh.nonzero().flatten()

                if _dst_node_ids.shape[0] == 0:
                    continue

                # remove a random edge from this neighbourhood, as well as edge weight
                offset_dst_idx = random.choice(_dst_node_ids)
                _src_node_neigh[offset_dst_idx] = 0

                if edge_weight is not None:
                    origin_weight_matrix[_src_id][offset_dst_idx] = 0

        # resolve duplicated edges
        xor_mask = torch.logical_xor(origin_adj_matrix, add_adj_matrix)

        new_adj_matrix = add_adj_matrix.masked_fill(~xor_mask, 0) + origin_adj_matrix
        if edge_weight is not None:
            new_weights_matrix = (
                add_weights_matrix.masked_fill(~xor_mask, 0) + origin_weight_matrix
            )
            _, new_edge_weights = dense_to_sparse(new_weights_matrix)

        new_edge_index, new_edge_type = dense_to_sparse(new_adj_matrix)
        new_edge_type -= 1

        # print(f'new_edge_type: {new_edge_type.shape}')
        # print(f'new_edge_weights: {new_edge_weights.shape}')
        # print(f'new_edge_index: {new_edge_index.shape}')
        if new_edge_weights is not None:
            new_edge_weights = new_edge_weights.to(edge_index.device)

        return (
            node_features,
            new_edge_index.to(edge_index.device),
            (new_edge_weights, new_edge_type.to(edge_index.device)),
            node_types,
        )

    def create_edge_type_swap(self, batch_data):
        """
        generate edge swap
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_data = self.edge_type_swap(g_data, swap_pct=self.swap_edge_pct)
            if new_data is not False:
                new_batch.append(new_data)
        return new_batch

    def edge_type_swap(self, g_data, swap_pct=0.05):
        """
        swap edge types
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        unique_edge_types = torch.unique(
            edge_type.view(
                -1,
            )
        )

        # print(f'edge_type shape: {edge_type.shape}')
        # TODO: handle case when cannot augment the data
        # skip if no enough edge types
        if unique_edge_types.shape[0] < 2:
            return False
        sampled_indices = torch.multinomial(unique_edge_types.float(), 2).long()
        swap_edge_types = torch.index_select(
            unique_edge_types, 0, sampled_indices
        ).long()

        # print(f'swap_edge_types: {swap_edge_types}')

        src_edge_indices = (
            (edge_type == swap_edge_types[0])
            .nonzero()
            .view(
                -1,
            )
            .cpu()
        )
        dst_edge_indices = (
            (edge_type == swap_edge_types[1])
            .nonzero()
            .view(
                -1,
            )
            .cpu()
        )

        num_edge_swap = int(
            min(
                src_edge_indices.shape[0] * swap_pct + 1,
                dst_edge_indices.shape[0] * swap_pct + 1,
            )
        )
        # print(f'src_edge_indices: {src_edge_indices}')
        # print(f'dst_edge_indices: {dst_edge_indices}')

        swap_src = torch.multinomial(src_edge_indices.float(), num_edge_swap).long()
        swap_dst = torch.multinomial(dst_edge_indices.float(), num_edge_swap).long()

        # print(f'swap_src: {swap_src}')
        # print(f'swap_dst: {swap_dst}')

        new_edge_type = copy.deepcopy(edge_type.cpu())

        for a_edge, b_edge in zip(swap_src, swap_dst):
            self.swap_values(new_edge_type, a_edge, b_edge)

        # print(f'new_edge_type shape: {new_edge_type.shape}')
        return (
            node_features,
            edge_index,
            (
                edge_weight,
                new_edge_type.view(
                    -1,
                ).to(edge_type.device),
            ),
            node_types,
        )

    def swap_values(self, values, src_idx, dst_idx):
        tmp = values[src_idx]
        values[src_idx] = values[dst_idx]
        values[dst_idx] = tmp
        # return values

    def create_node_type_swap(self, batch_data):
        """
        create node swap
        """
        new_batch = []
        for i in range(len(batch_data)):
            g_data = random.choice(batch_data)
            new_data = self.node_type_swap(g_data, swap_pct=self.swap_node_pct)
            if new_data is not False:
                new_batch.append(new_data)
        return new_batch

    def node_type_swap(self, g_data, swap_pct=0.05):
        """
        Swap node types if valid
        """
        node_features, edge_index, (edge_weight, edge_type), node_types = g_data
        valid_node_types = [
            idx
            for idx, node_type_list in enumerate(node_types)
            if len(node_type_list) > 0
        ]
        if len(valid_node_types) >= 2:
            swap_node_types = random.sample(valid_node_types, 2)
        else:
            return False
        num_node_swap = int(
            min(
                len(node_types[swap_node_types[0]]) * swap_pct + 1,
                len(node_types[swap_node_types[1]]) * swap_pct + 1,
            )
        )

        swap_nodes = [
            random.sample(node_types[swap_node_types[0]], num_node_swap),
            random.sample(node_types[swap_node_types[1]], num_node_swap),
        ]

        new_node_types = copy.deepcopy(node_types)

        self.remove_from_list(new_node_types[swap_node_types[0]], swap_nodes[0])
        self.add_to_list(new_node_types[swap_node_types[0]], swap_nodes[1])
        self.remove_from_list(new_node_types[swap_node_types[1]], swap_nodes[1])
        self.add_to_list(new_node_types[swap_node_types[1]], swap_nodes[0])

        return node_features, edge_index, (edge_weight, edge_type), new_node_types

    def remove_from_list(self, src_list, removal):
        """
        remove from list
        """
        for i in removal:
            src_list.remove(i)

    def add_to_list(self, src_list, addition):
        """
        add to list
        """
        src_list.extend(addition)


def sample_one_node_from_list(node_list):
    """
    random sample a het node
    """
    return random.choice(node_list)


# class GraphAugementor:
#     """
#     Method for add/remove nodes/edges from the het graph
#     """

#     def __init__(self):
#         pass

#     def add_new_node(self):
#         """
#         add new node
#         """

#     def remove_node(self):
#         """
#         remove node
#         """

#     def add_new_edge(self):
#         """
#         add new edge
#         """

#     def remove_edge(self):
#         """
#         remove edge
#         """
