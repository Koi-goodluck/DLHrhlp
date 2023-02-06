import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

BASE_DIR = str(Path(__file__).parent.parent.resolve())


@dataclass
class Config:

    w_neighbors_per: float = field(default=0.2, metadata={"help": "the percentage of flows the neighbors of node i covers"})  # type: ignore

    c_neighbors_per: float = field(default=1.0, metadata={"help": "the numbers of neighbors in term of c"})

    JK: int = field(default=1, metadata={"help": "layer aggregation, 0:do not use;1:max_pooling; 2:min_pooling; 3:mean_pooling"})  # type: ignore

    combineID: int = field(default=1, metadata={"help": "0:graphsage; 1:gru"})

    max_bp_iter: int = field(default=5, metadata={"help": "steps of neighbor propagation"})

    input_embedding: int = field(default=64, metadata={"help": "input embedding size"})

    EMBEDDING_SIZE: int = field(default=128, metadata={"help": "embedding size"})

    REG_HIDDEN: int = field(default=64, metadata={"help": "hidden untis in decoder"})

    LEARNING_RATE: float = field(default=0.001, metadata={"help": "learning rate"})

    BATCH_SIZE: int = field(default=64, metadata={"help": "the number of graphs in a batch to input"})

    MAX_ITERATION: int = field(default=30000, metadata={"help": "maximum training iterations"})

    scale: float = field(default=0.1, metadata={"help": "scale in l2 regularization"})

    model_name: str = field(default="DLHr_gru", metadata={"help": "the name of model to save"})

    initialization_stddev: float = field(default =0.01, metadata={"help": "the parameter stddev in initialization"})
    
    early_stop_sgn: bool = field(default = False, metadata={"help": "whether to employ early stopping"})
    
    node_feat_dim_1: int = 36

    node_feat_dim_2: int = 3  # distance from gravity center, Oi+Di, Ci

    size_training_set: int = 10000 # the size of training set

    size_validation_set: int = 100 # the size of training set

    training_networks_size: int = 25

    test_networks_size: int = 25

    h_m: int = 5 # the size of hub set for generating training labels  


def get_config():
    parser = argparse.ArgumentParser(description="optimization")
    for arg in fields(Config):
        help_msg = arg.metadata.get("help")
        parser.add_argument(
            f"--{arg.name}",
            type=arg.type,
            default=arg.default,
            help=help_msg,
        )

    config = Config(**vars(parser.parse_args()))

    return config
