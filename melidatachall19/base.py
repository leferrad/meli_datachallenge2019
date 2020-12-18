"""Module to provide base abstractions for the development"""

import abc


class Step(abc.ABC):
    """Abstract base class used to build new steps
    of a Machine Learning pipeline."""
    @abc.abstractmethod
    def run(self):
        pass
