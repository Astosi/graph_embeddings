import numpy as np
from numpy import float64

from components.embeddings import get_embeddings, save_embeddings
from components.graph import get_poi_graph, get_house_graph
from components.logs.Logger import get_logger
from components.prepare_data import get_house_geopandas, get_pois
from components.random_walk import random_walk
from components.skipgram import generate_examples, train_model, create_dataset
from enums.Category import Category
from enums.Hyperparameters import Hyperparameters
import pandas as pd

logger = get_logger(__name__)


def get_dataset(walks, vocabulary_len, hparams):
    targets, contexts, labels, weights = generate_examples(
        sequences=walks,
        vocabulary_size=vocabulary_len,
        window_size=hparams.num_steps,
        num_negative_samples=hparams.num_negative_samples
    )

    dataset = create_dataset(
        targets=targets,
        contexts=contexts,
        labels=labels,
        weights=weights,
        batch_size=hparams.batch_size
    )

    return dataset


def generate_embeddings(house_id, x: float, y: float, address: str, province: str, category: Category,
                        hparams: Hyperparameters):

    logger.info(f"Generating embeddings for {house_id} in {province} for category: {category.name}")
    embedding_col = f"embeddings_{category.name}"

    # Create an empty DataFrame for embeddings
    embedding_df = pd.Series(dtype=float64)

    # Get the house's location and nearby points of interest
    logger.info(f"Fetching geolocation and points of interest for {house_id} in {province}")

    house = get_house_geopandas(house_id=house_id, x=x, y=y, address=address, province=province)
    pois_graph = get_poi_graph(province=province, category=category)
    pois_near_by = get_pois(province=province, category=category)

    ########################################################################

    # Check if there's a pre-existing embedding for the house
    logger.info(f"Checking for pre-existing embedding for {house_id}")
    embedding = get_embeddings(house_id=house_id)

    if embedding is not None and embedding.all():
        logger.info(f"{house_id} already has an embedding. Setting and skipping...")
        embedding_df[embedding_col] = embedding
        for i, val in enumerate(embedding.tolist()):
            col_name = f"{embedding_col}_d{i + 1}"
            embedding_df[col_name] = val
        return embedding_df

    # Generate a graph from the house's location and nearby points of interest
    logger.info(f"Generating graph for {house_id}")
    the_graph = get_house_graph(house=house.values, pois_graph=pois_graph,
                                pois_near_by=pois_near_by.values, category=category)

    ########################################################################
    # Create a vocabulary for the graph and generate random walks on it
    logger.info(f"Creating vocabulary and generating random walks for {house_id}")
    vocabulary = list(the_graph.nodes)
    vocabulary_len = len(vocabulary)
    vocabulary_lookup = {token: idx for idx, token in enumerate(vocabulary)}
    logger.info(f"Vocabulary size: {vocabulary_len}")

    ########################################################################

    walks = random_walk(graph_to_walk=the_graph, vocabulary_lookup=vocabulary_lookup, hparams=hparams)
    logger.info(f"Number of walks generated: {len(walks)}")

    ########################################################################

    # Prepare the dataset for training the skip-gram model and train it
    logger.info(f"Preparing dataset and training model for {house_id}")
    dataset = get_dataset(walks=walks, vocabulary_len=vocabulary_len, hparams=hparams)
    model = train_model(vocabulary_len=vocabulary_len, dataset=dataset, hparams=hparams)

    ########################################################################

    # Get the embedding for the house, save it, and store it in the DataFrame
    logger.info(f"Generating embedding and saving it for {house_id}")
    embedding = np.array(model.get_layer("item_embeddings").get_weights()[0])[-1]
    logger.info(f"Generated embedding: {embedding}")
    save_embeddings(house_id, embedding)

    embedding_df[embedding_col] = embedding

    for i, val in enumerate(embedding.tolist()):
        col_name = f"{embedding_col}_d{i + 1}"
        embedding_df[col_name] = val

    return embedding_df
