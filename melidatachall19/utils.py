"""Module to provide utils in general"""

import logging
import os
import yaml

# Useful paths
CWD = os.path.abspath(__file__)
PROJECT_PATH = os.path.abspath(os.path.join(CWD, os.pardir, os.pardir))


def replace_path(d, root_path):
    """
    Replace paths in entries of 'd' to start from 'root_path'.
    Recursion is performed if 'd' has nested entries.
    Parameters
    ----------
    d : {dict}
        Having values corresponding either to paths or to dicts with paths
    root_path : {string}
        Root path to be added on entries in 'd'

    Returns
    -------
    d : dict with modified entries
    """
    for k in d.keys():
        if isinstance(d[k], dict):
            d[k] = replace_path(d[k], root_path)
        else:
            d[k] = os.path.join(root_path, d[k])
    return d


def load_profile(filename, root_path=None):
    """
    Load execution profile from YAML file.

    Parameters
    ----------
    filename : {string}
        Path to YAML file with execution profile
    root_path : {string or None}
        Root path for relative paths in profile. If None, using root of project.

    Returns
    -------
    profile : dict with execution profile entries
    """

    if not os.path.exists(filename):
        raise ValueError(f"Argument 'filename' must be an existing path. Got: {filename}")

    if root_path is None:
        # Using root path of project by default
        root_path = PROJECT_PATH

    # Load YAML file
    with open(filename) as f:
        profile = yaml.safe_load(f)

    # Solve paths that are relative to 'root_path'
    profile["paths"] = replace_path(profile["paths"],
                                    root_path=root_path)

    return profile


def get_logger(name='melidatachall19', level="INFO"):
    """
    Function to obtain a normal logger

    Parameters
    ----------
    name : {string}
        Name for the logger
    level : {string}
        Log level, which can be 'INFO', 'DEBUG', etc

    Returns
    -------
    logging.Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


# TODO: script to download dataset?
