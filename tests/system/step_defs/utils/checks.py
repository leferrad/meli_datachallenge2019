"""Module with utils about modeling and predictions checks"""

# Here, tune thresholds according to the stage of the project
MODELING_ACCEPTANCE_THRESHOLDS = {
    "train": {
        "accuracy": 0.7,
        "balanced_accuracy": 0.7,
        "f1_micro": 0.7,
    },
    "valid": {
        "accuracy": 0.6,
        "balanced_accuracy": 0.6,
        "f1_micro": 0.6,
    },
    "test": {
        "accuracy": 0.6,
        "balanced_accuracy": 0.6,
        "f1_micro": 0.6,
    },
}
