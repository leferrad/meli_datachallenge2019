"""Module to provide logic for the model evaluation"""

import gc
import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from melidatachall19.base import Step
from melidatachall19.metrics import evaluate_metrics
from melidatachall19.utils import get_logger
from melidatachall19 import kb


class EvaluateStep(Step):
    """
    Step defined to perform the evaluation of models over the test dataset.

    In particular, the evaluation will be performed per model (and overall data),
    by applying some selected classification metrics.

    Parameters
    ----------
    profile : {dict}
        Configuration of execution profile, having at least the entries:
        - paths: dict with str / dicts of paths to resources
        - logger: dict with entry:
            - level: str, indicating logging level
        - seed: int, seed for random operations
    """
    def __init__(self, profile):
        self.profile = profile
        self.seed = profile["seed"]
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Create logger for execution
        self.logger = get_logger(__name__, level=profile["logger"]["level"])
        self.logger.debug("ModelingStep initialized")

        # Variables to be filled in evaluation
        self.data = dict()
        self.models = dict()
        self.vectorizers = dict()
        self.label_encoders = dict()
        self.results = dict()

    def load(self):
        """Load resources for the evaluation"""
        for lang in ["es", "pt"]:
            self.logger.info(f"Loading resources for language={lang}")
            # Load model from disk
            self.models[lang] = load_model(self.profile["paths"]["model"][lang], compile=True)

            # Load vectorizer from disk (config + weights)
            # From https://stackoverflow.com/a/65225240/5484690
            with open(self.profile["paths"]["vectorizer"][lang], "rb") as f:
                vec = pickle.load(f)

            self.vectorizers[lang] = TextVectorization.from_config(vec['config'])
            # You have to call `adapt` with some dummy data (BUG in Keras)
            # self.vectorizer_es.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
            self.vectorizers[lang].set_weights(vec['weights'])

            # Load data from disk
            self.data[lang] = pd.read_parquet(self.profile["paths"]["test"][lang])
            # TODO: data already having column "language"
            self.data[lang]["language"] = kb.LANGUAGE_ES if lang == "es" else kb.LANGUAGE_PT

            # Load label encoder
            with open(self.profile["paths"]["label_encoder"][lang], "rb") as f:
                self.label_encoders[lang] = pickle.load(f)

    def evaluate(self):
        x_all, y_all = [], []
        for lang in ["es", "pt"]:
            x = self.data[lang]["title"]
            y = self.data[lang]["label"]
            res = evaluate_metrics(x=x, y=y,
                                   model=self.models[lang],
                                   vectorizer=self.vectorizers[lang],
                                   label_encoder=self.label_encoders[lang])
            self.logger.info(f"Results of test for model of language={lang}: ")
            self.logger.info(res)
            self.results[lang] = res

            # Save x y arrays for overall metrics
            x_all.append(x)
            y_all.append(y)

    # TODO: get predictions and evaluate, to then evaluate results overall

    def save(self):
        """Save all results for later analysis"""
        for lang, res in self.results.items():
            with open(self.profile["paths"]["results"]["test"][lang], "w") as f:
                json.dump(res, f)

    def flush(self):
        """Remove from memory heavy objects that are not longer needed after run()"""
        self.logger.info("Flushing objects")
        del self.data
        del self.models
        del self.vectorizers
        del self.label_encoders
        del self.results
        gc.collect()

    def run(self):
        """Entry point to run step"""
        self.load()
        self.evaluate()
        self.save()
        self.flush()
