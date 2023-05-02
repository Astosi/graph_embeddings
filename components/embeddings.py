import os
import pickle

import numpy as np

from components.logger import get_logger
logger = get_logger(__name__)

embeddings_dir = "/home/astosi/UH/GraphThings/gsn/data/houses/embeddings"


def get_embeddings(house_id):
    embedding_file = os.path.join(embeddings_dir, f"{house_id}.npy")

    if os.path.exists(embedding_file):
        return np.load(embedding_file, allow_pickle=True)

    return None


def save_embeddings(house_id, embedding):
    embedding_file = os.path.join(embeddings_dir, f"{house_id}.npy")

    with open(embedding_file, "wb") as fp:
        pickle.dump(embedding, fp)

        logger.info('Embedding has saved successfully!')
