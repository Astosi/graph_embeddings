import os
import pickle

import numpy as np
import pandas as pd

from components.logs.Logger import get_logger

logger = get_logger(__name__)

embeddings_dir = "/home/astosi/PycharmProjects/graph_embeddings/data/houses/embeddings"

csv_file = os.path.join(embeddings_dir, f"embeddings.csv")


def get_embeddings(house_id):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if house_id in df['house_id'].values:
            # Load the embedding from the CSV file
            embedding_str = df[df['house_id'] == house_id]['embedding'].iloc[0]
            embedding = np.fromstring(embedding_str, sep=',')
            return embedding

    return None


def save_embeddings(house_id, embedding):
    # Convert the embedding array to a comma-separated string
    embedding_str = ','.join(map(str, embedding))
    data = {'house_id': house_id, 'embedding': embedding_str}
    new_row = pd.DataFrame([data])

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if house_id in df['house_id'].values:
            # Update the existing embedding
            df.loc[df['house_id'] == house_id, 'embedding'] = embedding_str
        else:
            # Append the new embedding
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(csv_file, index=False)
    logger.info('Embedding has saved successfully!')
