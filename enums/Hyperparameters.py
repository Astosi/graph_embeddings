from typing import NamedTuple


class Hyperparameters(NamedTuple):
    p: float  # Random walk return parameter
    q: float
    num_walks: int  # Number of iterations of random walks   #5
    num_steps: int  # Number of steps of each random walk   #10
    num_negative_samples: int
    batch_size: int
    learning_rate: float
    embedding_dim: int
    num_epochs: int
