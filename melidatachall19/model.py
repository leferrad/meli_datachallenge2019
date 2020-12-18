"""Module to fit the classifier model"""
# See https://keras.io/examples/nlp/pretrained_word_embeddings/

import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import initializers, layers, optimizers, regularizers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from melidatachall19.base import Step
from melidatachall19.metrics import evaluate_metrics
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
        tf.random.set_seed(self.seed)

        # Create logger for execution
        self.logger = get_logger(__name__ + f"_lang={language}", level=profile["logger"]["level"])
        self.logger.debug("ModelingStep initialized")

    def _build_network(self):
        """Function to build a custom network for the model"""
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.Input(shape=(None,), dtype="int64"))
        # Learn embedding from input tokens
        self.model.add(layers.Embedding(
            self.num_tokens,
            self.embedding_dim,
            embeddings_initializer=initializers.glorot_normal(),
            trainable=True))
        self.model.add(layers.SpatialDropout1D(0.2))
        self.model.add(layers.Bidirectional(layers.LSTM(self.embedding_dim,
                                                        activation="tanh",
                                                        return_sequences=True)))
        self.model.add(layers.Conv1D(64, 5, activation="tanh",
                                     strides=3,
                                     kernel_regularizer=regularizers.l2(0.01)))
        self.model.add(layers.GlobalMaxPooling1D())
        self.model.add(layers.Dropout(0.2))
        # self.model.add(layers.Dense(64, activation="elu"))
        # self.model.add(layers.BatchNormalization())
        # self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(256, activation="elu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(self.n_labels, activation="softmax"))

        # Define callbacks for model optimization
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          min_delta=0,
                                                          patience=0,
                                                          verbose=0,
                                                          mode='auto')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                         patience=0, min_lr=1e-4)
        history = tf.keras.callbacks.History()
        self.callbacks = [early_stopping, reduce_lr, history]

        # NOTE: ensure both loss and metrics are sparse
        # See: https://github.com/tensorflow/tensorflow/issues/42045#issuecomment-674232499
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizers.Adam(lr=1e-3),
            metrics=["sparse_categorical_accuracy"],
        )

    def load(self):
        """Load resources before the modeling"""
        # Read data to initialize text vectorization
        df_train = pd.read_parquet(self.profile["paths"]["train"][self.language])

        self.logger.info(f"Dataset loaded has shape: {df_train.shape}")

        # Split data into train / valid datasets, stratified by category
        df_train, df_valid = train_test_split(df_train, test_size=self.valid_size,
                                              stratify=df_train[kb.LABEL_COLUMN],
                                              shuffle=True,
                                              random_state=self.seed)

        self.logger.info(f"Dataset for TRAIN has shape: {df_train.shape}")
        self.logger.info(f"Dataset for VALID has shape: {df_valid.shape}")

        # Get arrays from DF
        x_train = df_train[kb.TITLE_COLUMN]
        y_train = df_train[kb.LABEL_COLUMN]
        x_val = df_valid[kb.TITLE_COLUMN]
        y_val = df_valid[kb.LABEL_COLUMN]

        # Get number of labels to predict from train labels
        self.n_labels = len(y_train.unique())
        self.logger.info(f"Dataset has {self.n_labels} labels to predict")

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

        # Transform labels into integer in range [0, N-1]
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(y_train)
        y_val = self.label_encoder.transform(y_val)

        # Get arrays of data for train and valid
        self.x_train = self.vectorizer(np.array([[s] for s in x_train])).numpy()
        self.y_train = y_train
        self.x_val = self.vectorizer(np.array([[s] for s in x_val])).numpy()
        self.y_val = y_val

        self.results = dict()

    def train(self):
        """Train model with already loaded data"""
        t = time.time()
        self.logger.info("Running ModelingStep ...")
        self.logger.info("Model summary:")
        self.model.summary()
        self.logger.info("Start training model...")
        self.history = self.model.fit(self.x_train, self.y_train,
                                      validation_data=(self.x_val, self.y_val),
                                      validation_batch_size=self.batch_size,
                                      callbacks=self.callbacks,
                                      batch_size=self.batch_size,
                                      epochs=self.n_epochs)

        self.logger.info("Execution took %.3f seconds", time.time() - t)

    def evaluate(self):
        """Evaluate results of train + valid of obtained model"""
        self.results = dict()
        for k, x, y in [("train", self.x_train, self.y_train),
                        ("valid", self.x_val, self.y_val)]:
            self.logger.info(f"Computing metrics for dataset '{k}'")
            res = evaluate_metrics(x=x, y=y,
                                   model=self.model,
                                   # x arrays already tokenized
                                   vectorizer=None,
                                   # y vectors already encoded
                                   label_encoder=None)
            self.logger.info("Results for model : ")
            self.logger.info(res)
            self.results[k] = res

    def flush(self):
        """Remove from memory heavy objects that are not longer needed after run()"""
        self.logger.info("Flushing objects")
        del self.x_train
        del self.y_train
        del self.x_val
        del self.y_val
        del self.model
        del self.vectorizer
        del self.label_encoder
        gc.collect()

    def run(self):
        """Entry point to run step"""
        self.load()
        self.train()
        self.evaluate()
        self.save()
        self.flush()

    def save(self):
        """Save all relevant artifacts for later usage and analysis"""
        self.logger.info("Saving modeling artifacts")
        self.model.save(self.profile["paths"]["model"][self.language])
        # From https://stackoverflow.com/a/65225240/5484690
        with open(self.profile["paths"]["vectorizer"][self.language], "wb") as f:
            pickle.dump({'config': self.vectorizer.get_config(),
                         'weights': self.vectorizer.get_weights()}, f)
        with open(self.profile["paths"]["label_encoder"][self.language], "wb") as f:
            pickle.dump(self.label_encoder, f)

        for k, res in self.results.items():
            with open(self.profile["paths"]["results"][k][self.language], "w") as f:
                json.dump(res, f)

        # with open(self.profile["paths"]["results"]["fit_history"][self.language], "w") as f:
        #     json.dump(self.history.history, f)
