import random

import numpy as np
from tqdm import tqdm


def next_step(graph_to_walk, previous, current, p, q):

    neighbors = list(graph_to_walk.neighbors(current))

    weights = []
    # Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            # Control the probability to return to the previous node.
            weights.append(graph_to_walk[current][neighbor]["weight"] / p)
        elif graph_to_walk.has_edge(neighbor, previous):
            # The probability of visiting a local node.
            weights.append(graph_to_walk[current][neighbor]["weight"])
        else:
            # Control the probability to move forward.
            weights.append(graph_to_walk[current][neighbor]["weight"] / q)

    # Compute the probabilities of visiting each neighbor.
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    # Probabilistically select a neighbor to visit.
    next = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next


def random_walk(graph_to_walk, vocabulary_lookup, hparams):

    num_walks = hparams.num_walks
    num_steps = hparams.num_steps
    p = hparams.p
    q = hparams.q

    walks = []
    nodes = list(graph_to_walk.nodes())

    # Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)

        for node in tqdm(
            nodes,
            position=0,
            leave=True,
            desc=f"Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):

            # Start the walk with a random node from the graph.
            walk = [node]
            # Randomly walk for num_steps.
            while len(walk) < num_steps:

                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None

                # Compute the next node to visit.
                next = next_step(graph_to_walk, previous, current, p, q)
                walk.append(next)
            # Replace node ids (movie ids) in the walk with token ids.
            walk = [vocabulary_lookup[token] for token in walk]
            # Add the walk to the generated sequence.
            walks.append(walk)

    return walks