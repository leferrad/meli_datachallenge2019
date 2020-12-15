"""Module to fit the classifier model"""
# See https://keras.io/examples/nlp/pretrained_word_embeddings/

import gc
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import Input, Model, initializers, optimizers, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from melidatachall19.base import Step
from melidatachall19.utils import get_logger
from melidatachall19 import kb


class ModelingStep(Step):
    """
    Step defined to perform the modeling over the dataset belonging to "language".

    In particular, this approach is composed by:
    - Tokenize input column to transformt text into numeric data
    - Build a neural network to learn how to map numeric data into the desired categories

    Parameters
    ----------
    profile : {dict}
        Configuration of execution profile (e.g. from "profile.yml"), having at least the entries:
        - paths: dict with str / dicts of paths to resources
        - model: dict with these entries:
            - max_tokens: int, max number of tokens to learn to extract from text
            - max_len_seq: int, max number of tokens in a sequence
            - embedding_dim: int, size of embeddings to learn
            - valid_size: float, ratio of validation set (e.g. 0.2 for 20% of train data)
            - n_epochs: int, number of epochs for training
            - batch_size: int, number of records to process per batch in training
        - logger: dict with entry:
            - level: str, indicating logging level
        - seed: int, seed for random operations
    """
    def __init__(self, profile, language="es"):
        self.profile = profile
        self.language = language
        self.max_tokens = int(profile["model"]["max_tokens"])
        self.max_len_seq = profile["model"]["max_len_seq"]
        self.embedding_dim = profile["model"]["embedding_dim"]
        self.n_epochs = profile["model"]["n_epochs"]
        self.batch_size = profile["model"]["batch_size"]
        self.valid_size = profile["model"]["valid_size"]
        self.seed = profile["seed"]
        np.random.seed(self.seed)
        # TODO: fix seed everywhere!

        # Create logger for execution
        self.logger = get_logger(__name__, level=profile["logger"]["level"])
        self.logger.debug("ModelingStep initialized")

        # Read data to initialize text vectorization
        df_train = pd.read_parquet(profile["paths"]["train"][language])

        # Split data into train / valid datasets, stratified by category
        df_train, df_valid = train_test_split(df_train, test_size=self.valid_size,
                                              stratify=df_train[kb.LABEL_COLUMN],
                                              random_state=self.seed)

        # Get number of labels to predict from train labels
        y_train = to_categorical(df_train[kb.LABEL_COLUMN])
        self.n_labels = len(y_train)

        # Create text vectorizer
        self.vectorizer = TextVectorization(max_tokens=self.max_tokens,
                                            ngrams=None,
                                            output_sequence_length=self.max_len_seq)

        text_ds = tf.data.Dataset.from_tensor_slices(
            df_train[kb.TITLE_COLUMN]).batch(self.batch_size)
        self.vectorizer.adapt(text_ds)
        self.vocabulary = self.vectorizer.get_vocabulary()
        self.num_tokens = len(self.vocabulary)

        # Create model based on configured network
        self._build_network()

        # Get arrays of data for train and valid
        self.x_train = self.vectorizer(np.array([[s] for s in df_train[kb.TITLE_COLUMN]])).numpy()
        self.y_train = np.array(df_train[kb.LABEL_COLUMN])
        self.x_val = self.vectorizer(np.array([[s] for s in df_valid[kb.TITLE_COLUMN]])).numpy()
        self.y_val = np.array(df_valid[kb.LABEL_COLUMN])

    def _build_network(self):
        """Function to build a custom network for the model"""
        # Learn embedding from input tokens
        embedding_layer = layers.Embedding(
            self.num_tokens,
            self.embedding_dim,
            embeddings_initializer=initializers.glorot_normal(),
            trainable=True,
        )

        int_sequences_input = Input(shape=(None,), dtype="int64")
        x = embedding_layer(int_sequences_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, activation="relu")(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        y = layers.Dense(self.n_labels, activation="softmax")(x)
        self.model = Model(int_sequences_input, y)

        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(lr=1e-3),
            metrics=["accuracy",
                     tf.keras.metrics.CategoricalAccuracy()]
        )

    def train(self):
        """Train model with already loaded data"""
        t = time.time()
        self.logger.info("Running ModelingStep ...")
        self.logger.info("Model summary:")
        self.model.summary()
        self.logger.info("Start training model...")
        self.model.fit(self.x_train, self.y_train,
                       validation_data=(self.x_val, self.y_val),
                       validation_batch_size=self.batch_size,
                       batch_size=self.batch_size, epochs=self.n_epochs)

        self.logger.info("Execution took %.3f seconds", time.time() - t)

    def flush(self):
        """Remove from memory heavy objects that are not longer needed after run()"""
        self.logger.info("Flushing objects")
        del self.x_train
        del self.y_train
        del self.x_val
        del self.y_val
        gc.collect()

    def run(self):
        """Entry point to run step"""
        self.train()
        self.save()
        self.flush()

    def save(self):
        """Save both model + vectorizer for later usage"""
        self.logger.info("Saving model + vectorizer")
        self.model.save(self.profile["paths"]["model"][self.language])
        # From https://stackoverflow.com/a/65225240/5484690
        with open(self.profile["paths"]["vectorizer"][self.language], "wb") as f:
            # pickle.dump(self.vectorizer, f)
            pickle.dump({'config': self.vectorizer.get_config(),
                         'weights': self.vectorizer.get_weights()}, f)
