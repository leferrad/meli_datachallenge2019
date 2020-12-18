"""Steps for model.feature"""

import json
import os
import subprocess

import pytest
from pytest_bdd import given, when, then, scenarios, parsers

from melidatachall19.utils import load_profile

from .utils.checks import MODELING_ACCEPTANCE_THRESHOLDS
from .utils.paths import PROJECT_PATH


scenarios('../features/model.feature')

PROFILE = "profile_sampled_data"


@pytest.fixture
def path_to_profile():
    yield os.path.join(PROJECT_PATH, "profiles", f"{PROFILE}.yml")


@given(parsers.parse("the execution profile '{profile}'"))
def given_profile(profile):
    global PROFILE
    PROFILE = profile
    return path_to_profile


@when('the modeling script is executed')
def modeling_script_executed(path_to_profile):
    tools_script = os.path.join(PROJECT_PATH, "tools", "script.sh")
    args = os.path.join(PROJECT_PATH, f'scripts/modeling.py -p {path_to_profile}')
    cmd = f"{tools_script} {args}"
    out = subprocess.run(cmd, shell=True)
    assert out.returncode == 0, f"Execution of command '{cmd}' returned {out.returncode}. " \
                                f"Stderr: {out.stderr}"


@then('the model and results are saved')
def model_results_saved(path_to_profile):
    # Folder with models
    folder = os.path.join(PROJECT_PATH, "models")
    out = os.listdir(folder)

    profile = load_profile(path_to_profile)

    for k, path in profile["paths"]["model"].items():
        model_filename = path.split("/")[-1]
        assert model_filename in out, f"The model {model_filename} was not found"

    # Folder with results
    folder = os.path.join(folder, "results")
    out = os.listdir(folder)

    # TRAIN
    for k, path in profile["paths"]["results"]["train"].items():
        results_filename = path.split("/")[-1]
        assert results_filename in out, f"TRAIN results {results_filename} not found"

    # VALID
    for k, path in profile["paths"]["results"]["valid"].items():
        results_filename = path.split("/")[-1]
        assert results_filename in out, f"VALID results {results_filename} not found"


@then('the train and valid results pass the acceptance criteria')
def train_valid_results_pass_acceptance_criteria(path_to_profile):
    profile = load_profile(path_to_profile)

    # Folder with results
    folder = os.path.join(PROJECT_PATH, "models", "results")

    # TRAIN
    for model, path in profile["paths"]["results"]["train"].items():
        with open(os.path.join(folder, "train", path), "r") as f:
            results_train = json.load(f)

        for k, v in MODELING_ACCEPTANCE_THRESHOLDS["train"].items():
            assert results_train[k] >= v, \
                f"Acceptance criteria '{k} >= {v}' was not passed " \
                f"for TRAIN results of model {model}"

    # VALID
    for model, path in profile["paths"]["results"]["valid"].items():
        with open(os.path.join(folder, "valid", path), "r") as f:
            results_train = json.load(f)

        for k, v in MODELING_ACCEPTANCE_THRESHOLDS["valid"].items():
            assert results_train[k] >= v, \
                f"Acceptance criteria '{k} >= {v}' was not passed " \
                f"for VALID results of model {model}"


@when('the evaluation script is executed')
def evaluation_script_executed(path_to_profile):
    tools_script = os.path.join(PROJECT_PATH, "tools", "script.sh")
    args = os.path.join(PROJECT_PATH, f'scripts/evaluation.py -p {path_to_profile}')
    cmd = f"{tools_script} {args}"
    out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert out.returncode == 0, f"Execution of command '{cmd}' returned {out.returncode}. " \
                                f"Stderr: {out.stderr}"


@then('the test results are saved')
def test_results_saved(path_to_profile):
    profile = load_profile(path_to_profile)

    # Folder with results
    folder = os.path.join(PROJECT_PATH, "models", "results")
    out = os.listdir(folder)

    # TEST
    for k, path in profile["paths"]["results"]["test"].items():
        results_filename = path.split("/")[-1]
        assert results_filename in out, f"TEST results {results_filename} not found"


@then('the test results pass the acceptance criteria')
def test_results_pass_acceptance_criteria(path_to_profile):
    profile = load_profile(path_to_profile)

    # Folder with results
    folder = os.path.join(PROJECT_PATH, "models", "results")

    # TEST
    for model, path in profile["paths"]["results"]["test"].items():
        with open(os.path.join(folder, "test", path), "r") as f:
            results_train = json.load(f)

        for k, v in MODELING_ACCEPTANCE_THRESHOLDS["test"].items():
            assert results_train[k] >= v, \
                f"Acceptance criteria '{k} >= {v}' was not passed " \
                f"for TEST results of model {model}"

