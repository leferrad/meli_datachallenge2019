"""Script to run modeling scenario to fit a model from input data"""

import argparse
import time

from melidatachall19.utils import load_profile, get_logger
import melidatachall19


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute modeling scenario")
    parser.add_argument('-p', '--profile', dest='profile', default="profile.yml",
                        help='Profile file for execution')
    args = parser.parse_args()

    t = time.time()

    # Load execution profile to be used
    profile = load_profile(args.profile)

    # Create logger for execution
    logger = get_logger(__name__, level=profile["logger"]["level"])
    logger.debug("Modeling script started")

    # Define steps to run
    steps = [
        melidatachall19.PreProcessStep(profile),
        melidatachall19.ModelingStep(profile, language="es"),
        melidatachall19.ModelingStep(profile, language="pt"),
    ]

    # Run steps
    for step in steps:
        step.run()

    logger.info("Total execution took %.3f seconds", time.time() - t)
