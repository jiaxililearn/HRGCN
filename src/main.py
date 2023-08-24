import os
from pydoc import resolve
import click


@click.command()
@click.option("--lr", default=0.0001, help="learning rate")
@click.option("--save_model_freq", default=2, help="data_path")
@click.option("--model_path", default="../model_save_clean", help="model path dir")
@click.option("--data_path", default="../ProcessedData_clean", help="data path dir")
@click.option("--num_train", default=64000, help="number of training graphs")
@click.option(
    "--num_eval",
    default=None,
    type=int,
    help="limit number of eval graphs, None for do not limit",
)
@click.option("--batch_s", default=2000, help="batch size")
@click.option("--mini_batch_s", default=500, help="mini batch size")
@click.option("--train_iter_n", default=100, help="max train iter")
@click.option("--num_node_types", default=8, help="num of node types in data")
@click.option("--num_edge_types", default=1, help="num of edge types in data")
@click.option("--source_types", default=None, type=str, help="consider Source types")
@click.option(
    "--input_type",
    default="batch",
    type=str,
    help="the way of feeding model. i.e, single | batch",
)
@click.option("--hidden_channels", default=16, help="size of hidden channels")
@click.option("--feature_size", default=7, help="input node feature size")
@click.option("--out_embed_s", default=32, help="output feature size")
@click.option(
    "--num_hidden_conv_layers", default=1, help="hidden conv size for het node features"
)
@click.option(
    "--sampling_size",
    default=None,
    type=int,
    help="sampling size for each epoch. default None to use all training in each epoch",
)
@click.option(
    "--eval_size",
    default=None,
    type=int,
    help="evaluation size for each epoch. default None to use all the rest eval dataset in each epoch",
)
@click.option("--random_seed", default=36, help="random seed")
@click.option("--trainer_version", default=2, help="trainer version")
@click.option("--model_version", default=11, help="train with model version")
@click.option("--model_sub_version", default=0, help="train with sub model version")
@click.option("--checkpoint", default=None, type=int, help="model checkpoint to load")
@click.option("--checkpoint_path", default=None, type=str, help="model checkpoint path")
@click.option(
    "--embed_activation",
    default="sigmoid",
    type=str,
    help="activation func for embeddings. relu | sigmoid",
)
@click.option(
    "--edge_perturbation_p",
    default=0.001,
    type=float,
    help="probability of edge perturbation",
)
@click.option(
    "--augmentation_method",
    default="all",
    type=str,
    help="graph augment method",
)
@click.option(
    "--edge_ratio_percentile",
    default=0.95,
    type=float,
    help="percentile used for edge perturbation",
)
@click.option(
    "--edge_mutate_prob",
    default=None,
    type=float,
    help="probability if an edge should be mutated",
)
@click.option(
    "--add_method", default="rare", type=str, help="edge addition method. rare | simple"
)
@click.option(
    "--edge_addition_pct",
    default=0.1,
    type=float,
    help="percentage of new edges to add",
)
@click.option(
    "--replace_edges",
    default=True,
    type=bool,
    help="if replacing the old edges with new",
)
@click.option(
    "--subgraph_ratio",
    default=0.01,
    type=float,
    help="subgraph size to sample when node insertion",
)
@click.option(
    "--insertion_iteration",
    default=1,
    type=int,
    help="num of iteration to insert new nodes",
)
@click.option(
    "--swap_node_pct",
    default=0.05,
    type=float,
    help="percentage of nodes to be swapped",
)
@click.option(
    "--swap_edge_pct",
    default=0.05,
    type=float,
    help="percentage of edges to be swapped",
)
@click.option(
    "--main_loss",
    default="svdd",
    type=str,
    help="the main unsupervised loss used. svdd | semi-svdd",
)
# @click.option('--known_abnormal_ratio', default=0.1, type=float, help='ratio of known abnormal graphs to add during semi-svdd training')
@click.option(
    "--weighted_loss",
    default="bce",
    type=str,
    help="supervise loss used for training. bce | deviation",
)
@click.option("--loss_weight", default=0, type=float, help="weight of the bce weight")
@click.option(
    "--ablation",
    default=None,
    type=str,
    help="ablation setup for exp, requires the num_node_types/num_edge_types to be 1. no-edge-relation | no-node-relation | no-edge-node-relation",
)
@click.option(
    "--eval_method",
    default="svdd",
    type=str,
    help="method used to do the eval for the model. svdd | bce | both",
)
@click.option(
    "--job_prefix",
    default="any",
    type=str,
    help="job prefix when uploading checkpoint to S3",
)
@click.option("--dataset_id", default=0, help="choose dataset used for training")
@click.option(
    "--fix_center",
    default=True,
    type=bool,
    help="if fix the svdd center on first batch pass",
)
@click.option(
    "--test_set", default=True, type=bool, help="if create test dataset from input"
)
@click.option(
    "--ignore_weight", default=False, type=bool, help="if ignore the edge weight"
)
@click.option(
    "--split_data",
    default=False,
    type=bool,
    help="if random split data on train or read from existings",
)
@click.option("--sagemaker", default=False, type=bool, help="is it running in SageMaker")
@click.option("--unzip", default=False, type=bool, help="if unzip feature lists first")
@click.option(
    "--tolerance", default=5, type=int, help="early stopping criteria tolerance"
)
@click.option("--s3_stage", default=False, type=bool, help="if stage results to s3")
@click.option(
    "--s3_bucket",
    default="prod-tpgt-knowledge-lake-sandpit-v1",
    help="S3 bucket to upload intermediate artifacts",
)
@click.option(
    "--s3_prefix",
    default="application/anomaly_detection/deeptralog/HetGNN/model_save_clean/",
    help="S3 prefix to upload intermediate artifacts",
)
def main(**args):
    args = resolve_args(args)
    print(args)
    if args["trainer_version"] == 2:
        from train_2 import Train2 as Train
    else:
        from train import Train

    t = Train(**args)
    t.train()


def resolve_args(args):
    if args["sagemaker"]:
        args["model_path"] = os.environ["SM_MODEL_DIR"]
        args["data_path"] = os.environ["SM_CHANNEL_TRAIN"]
        if args["checkpoint"]:
            args["checkpoint_path"] = os.environ["SM_CHANNEL_MODEL"]
    return args


if __name__ == "__main__":
    main()
