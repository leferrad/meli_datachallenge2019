"""Module to preprocess text input data"""

import gc
import re
import time

try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import unidecode

from melidatachall19.base import Step
from melidatachall19.utils import get_logger
from melidatachall19 import kb

pd.options.mode.chained_assignment = None  # default='warn'


class PreProcessStep(Step):
    """
    Step defined to perform the following preprocess steps in the data:
    - Clean text in column 'title'
    - Convert string categories into integer ones in 'category'
    - Sample data (if required, for quicker tests)
    - Divide datasets by "language" (to have one model per language)
    - Split data in train/test datasets per language, stratified by categories
    - Save resulting datasets

    Parameters
    ----------
    profile : {dict}
        Configuration of execution profile (e.g. from "profile.yml"), having at least the entries:
        - paths: dict with str / dicts of paths to resources
        - preprocess: dict with these entries:
            - nrows: int, number of rows to sample (-1 if no sampling to do)
            - test_size: float, ratio of test dataset (e.g. 0.3 for 30% of total data)
            - min_count_category: int, minimum count for a category to include it in the model
        - logger: dict with entry:
            - level: str, indicating logging level
        - seed: int, seed for random operations
    """
    def __init__(self, profile):
        self.profile = profile
        self.path_to_data = profile["paths"]["raw"]
        self.nrows = profile["preprocess"]["nrows"]
        self.test_size = profile["preprocess"]["test_size"]
        self.min_count_category = profile["preprocess"]["min_count_category"]
        self.seed = profile["seed"]

        # Create label encoder to map string categories into integer labels
        self.label_encoder = LabelEncoder()

        # Create logger for execution
        self.logger = get_logger(__name__, level=profile["logger"]["level"])
        self.logger.debug("PreProcessStep initialized")

        # Blacklist of words per language
        self.blacklist_words = {
            "spanish": ["original", "nuevo", "oferta", "modelo", "caja", "kit", "pack", "negro",
                        "blanco", "cuota", "combo", "nueva", "color", "set", "x", "cp", "cm"],
            "portuguese": ["original", "novo", "promocao", "kit", "caixa", "modelo", "peca",
                           "preto", "branco", "unidade", "frete", "grati"],
        }

    def _split(self, df):
        """Split data into train / test datasets, stratified by category"""
        df_train, df_test = train_test_split(df, test_size=self.test_size,
                                             stratify=df[kb.LABEL_COLUMN],
                                             random_state=self.seed)

        return df_train, df_test

    def _process_title(self, df, language="spanish"):
        """Function to clean and prepare the text column 'title'"""
        # To lower
        title = df[kb.TITLE_COLUMN].str.lower()

        # Regex to clean text
        special_chars = re.escape(',`\'"|#=_![](){}<>^\\+/*?%.~:@;')
        special_chars = re.compile(r'[%s]' % special_chars)
        multiple_whitespaces = re.compile(r'\s\s+')
        words_with_dashes = re.compile(r' -(?=[^\W\d_])')
        numbers = re.compile(r'\d')

        # Apply regex on text column
        title = title.str.replace(words_with_dashes, ' - ')
        title = title.str.replace(special_chars, ' ')
        title = title.str.replace(numbers, '0')
        title = title.str.replace(multiple_whitespaces, ' ')

        # Remove stopwords based on language
        # Also, apply unidecode to remove accents
        title = title.apply(lambda line: " ".join(
            [unidecode.unidecode(x) for x in line.split(" ")
             if x not in stopwords.words(language) and x not in self.blacklist_words[language]]))

        # Trim text to a max range based on 90th percentile
        max_len = int(np.quantile(title.apply(len), 0.9))
        title = title.str.slice(0, max_len)

        # Not needed steps
        # - Remove nans (not seen)

        df[kb.TITLE_COLUMN] = title

        return df

    def preprocess(self):
        t = time.time()
        self.logger.info("Running PreProcessStep ...")

        # - Read raw dataset from CSV file
        df = pd.read_csv(self.path_to_data)
        self.logger.info(f"Dataset loaded has shape: {df.shape}")

        if self.nrows > 0:
            # To work with a random smaller sample (e.g. for quicker testing)
            df = df.sample(n=self.nrows, replace=False, random_state=self.seed)

        # - Process "category" column
        # Map category text -> category number
        # NOTE: done in all the data before splitting datasets for easy usage
        df[kb.LABEL_COLUMN] = self.label_encoder.fit_transform(df[kb.CATEGORY_COLUMN])

        # - Separate ES / PT data
        # NOTE: this separation is done under the assumption of having
        #       one model per language.
        df_es = df[df[kb.LANGUAGE_COLUMN] == kb.LANGUAGE_ES]
        df_pt = df[df[kb.LANGUAGE_COLUMN] == kb.LANGUAGE_PT]

        # - Remove classes with very low support
        # NOTE: do this per language, to have it at model level
        label_count = df_es[kb.LABEL_COLUMN].value_counts()
        label_count = label_count[label_count >= self.min_count_category]
        df_es = df_es[df_es[kb.LABEL_COLUMN].isin(label_count.index)]

        label_count = df_pt[kb.LABEL_COLUMN].value_counts()
        label_count = label_count[label_count >= self.min_count_category]
        df_pt = df_pt[df_pt[kb.LABEL_COLUMN].isin(label_count.index)]

        # - Process "title" column
        df_es = self._process_title(df_es, language="spanish")
        df_pt = self._process_title(df_pt, language="portuguese")

        # - Drop not needed columns
        df_es.drop([kb.LABEL_QUALITY_COLUMN,
                    kb.LANGUAGE_COLUMN,
                    kb.CATEGORY_COLUMN], axis=1, inplace=True)
        df_pt.drop([kb.LABEL_QUALITY_COLUMN,
                    kb.LANGUAGE_COLUMN,
                    kb.CATEGORY_COLUMN], axis=1, inplace=True)

        # Split train / test data (stratified)
        df_train_es, df_test_es = self._split(df_es)
        df_train_pt, df_test_pt = self._split(df_pt)

        # - Save datasets
        # NOTE: resetting index since it won't be used anymore
        # NOTE: saved in Parquet format for lower space
        # NOTE: test sets separated to evaluate performance per model
        #       (overall performance also computed)
        self.logger.info("Saving resulting datasets...")
        df_train_es.reset_index(drop=True).to_parquet(self.profile["paths"]["train"]["es"])
        df_test_es.reset_index(drop=True).to_parquet(self.profile["paths"]["test"]["es"])
        df_train_pt.reset_index(drop=True).to_parquet(self.profile["paths"]["train"]["pt"])
        df_test_pt.reset_index(drop=True).to_parquet(self.profile["paths"]["test"]["pt"])

        self.logger.info("Execution took %.3f seconds", time.time() - t)

    def flush(self):
        """Remove from memory heavy objects that are not longer needed after run()"""
        self.logger.info("Flushing objects")
        del self.label_encoder
        gc.collect()

    def run(self):
        """Entry point to run step"""
        self.preprocess()
        self.flush()
