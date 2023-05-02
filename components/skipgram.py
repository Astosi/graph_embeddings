import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from tensorflow import keras
from collections import defaultdict

from keras import layers
from tqdm import tqdm


def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)
    # Iterate over all sequences (walks).
    for sequence in tqdm(
            sequences,
            position=0,
            leave=True,
            desc=f"Generating postive and negative examples",
    ):
        # Generate positive and negative skip-gram pairs for a sequence (walk).
        pairs, labels = keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocabulary_size,
            window_size=window_size,
            negative_samples=num_negative_samples,
        )
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in example_weights:
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)


# 1024
def create_dataset(targets, contexts, labels, weights, batch_size):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_model(vocabulary_size, embedding_dim):
    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32"),
    }
    # Initialize item embeddings.
    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    # Lookup embeddings for target.
    target_embeddings = embed_item(inputs["target"])
    # Lookup embeddings for context.
    context_embeddings = embed_item(inputs["context"])

    # Compute dot similarity between target and context embeddings.
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    # Create the model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def train_model(vocabulary_len, dataset, hparams):
    model = create_model(vocabulary_len, hparams.embedding_dim)
    model.compile(
        optimizer=keras.optimizers.Adam(hparams.learning_rate),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )
    model_v2 = model.fit(dataset, epochs=hparams.num_epochs)

    # plt.plot(model_v2.history["loss"])
    # plt.ylabel("loss")
    # plt.xlabel("epoch")
    # plt.show()

    return model


def plot_embeddings(model):
    # Apply t-SNE transformation on node embeddings
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(model.get_layer("item_embeddings").get_weights()[0])

    # node_embeddings_2d

    # draw the points
    alpha = 0.7

    plt.figure(figsize=(10, 8))
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        cmap="jet",
        alpha=alpha,
    )
