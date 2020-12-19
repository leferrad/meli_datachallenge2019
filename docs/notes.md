Notes about approach taken
==========================

**Description**: This document is intended to explain the approach taken for the modeling of the solution, as well as provide some notes about the work done in this project.

## Scope

As a design decision, these are some statements about the scope of the work done:

- **Using neural networks as Machine Learning approach:** Since the main source of information to use is text, this is an NLP problem to solve and it is well known that the state-of-art algorithms in this problematic are neural networks. Therefore, despite the lack of experience in the recent techniques, this was the path to take for the solution. 
- **No pretrained solutions:** Due to the lack of time to validate improvements, as well as the "challenge" of doing a solution from scratch, no pretrained models will be used for this approach.

As mentioned in the project's README, the main goal of this work was to do a productive solution, with good enough results but probably not ready to compete in a leaderboard of a competence (due to the lack of time and experience on NLP projects).
            
Due to the large amount of records available (+20M), the general tuning of the solution was done with a sample and then used with all possible data.                     
                      
## Preprocessing

In order to prepare the data for the modeling, these are the steps defined:

- Label encoding on "category" column to convert it into range [0, N-1] 
- Classes with a very low coverage (e.g. <20 samples) will be filtered
- Split datasets per language (spanish and portuguese)
    - Train / Test split as cross validation strategy
    - During modeling, Train is split into Train / Valid
    - IMPORTANT: split must be stratified per categories, to ensure the same balance of classes on each dataset.  
- Clean text:
    - Lower case
    - Remove accents through unidecode
    - Remove special characters
    - Remove multiple whitespaces
    - Replace numbers
    - Remove stopwords (depending on language)    
    - Blacklist of words to be removed
        - e.g. in spanish: original, nuevo, combo, oferta 
    - Trim text to a max range based on 90th percentile

**Next steps**
- Optimize operations to make preprocessing faster
- Refine blacklists of words to remove
- Refine ways to remove "garbage" from tex
- Handle encoding of text (some issues already identified)


## Modeling

Once the text was prepared, these are the steps followed to do the modeling:

- Tokenization to transform text into integer tokens:
    - Done with Keras's `TextVectorization`, which allows to perform classical steps (e.g. normalization, tokenization, etc) as well as to get a "vocabulary" from the text (see [this article](https://towardsdatascience.com/you-should-try-the-new-tensorflows-textvectorization-layer-a80b3c6b00ee))
- Network architecture
    - Embeddings as first layer to learn from tokens in contextual way
    - Once the embeddings are processed, a recurrent layer helps to ...
    - Then, the sequences are processed in terms of 1D convolutions to ...
    - Loss used: sparse (see [this](https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c))
- Metrics implemented:
    - **Accuracy**: The ratio of classes correctly predicted, as a quick reference of the performance of the model.
    - **Balanced accuracy**: Since there is some imbalance of the categories to predict (some classes have very few records), it is convenient to correct the accuracy to take into account the balance of the classes by averaging the recall per class over all the classes.       
    - **Precision**: This metric allows to understand how good the categories are predicted in terms of having low false positives. A micro averaging is used to deal with imbalance of classes.
    - **Recall**: Metric used to understand how exhaustive the models are in terms of predicting all the available instances of classes. Computed a micro average to deal with imbalance of classes.
    - **F1-Score**: Obtained as a summary of the Precision and Recall computed (micro avg) through the harmonic mean.      
      
- Some considerations   
    - Initialization is important (e.g. from uniform to glorot results improved)  
    - Usage of Embeddings improved the results from the begin.
    - Adding layers BatchNormalization improved the results significantly, especially in last layers
    - Using Bidirectional improved results (for the recurrent layer)
    - Tricks for ovefitting:
        - Dropout + SpatialDropout to force robustness of network
        - Activation function "elu" in last Dense layers to avoid vanishing gradient 
    - Callbacks to stop learning when overfitting occurs
    - Transformers technique was tried, with not good results:
        - Examples of tokenizations:
            - Aireador -> aire-ador
            - Mosquitero -> mos-qui-tero

- Seed fixed to have reproducible results


**Next steps:**

- Improve model architecture to mitigate overfitting
    - Better embeddings
    - Better layers
    - Better loss function (for many classes problem)
- Implement data generator to have balanced data during training
- Compute metrics overall data (not only per language)

