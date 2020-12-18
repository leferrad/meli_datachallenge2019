"""Script to run evaluation scenario to evaluate the model with test data"""

import argparse
import os
import time

from melidatachall19.utils import load_profile, get_logger
import melidatachall19


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute evaluation scenario")
    parser.add_argument('-p', '--profile', dest='profile',
                        default="../profiles/profile_sampled_data.yml",
                        help='Profile file for execution')
    parser.add_argument('-r', '--root', dest='root',
                        default=".",
                        help='Root path of execution')
    args = parser.parse_args()

    t = time.time()

    # Load execution profile to be used
    profile = load_profile(args.profile, root_path=os.path.abspath(args.root))

    # Create logger for execution
    logger = get_logger(__name__, level=profile["logger"]["level"])
    logger.debug("Evaluation script started")

    # Define steps to run
    steps = [
        melidatachall19.EvaluateStep(profile)
    ]

    # Run steps
    for step in steps:
        step.run()

    logger.info("Total execution took %.3f seconds", time.time() - t)
