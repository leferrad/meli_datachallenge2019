"""Module with utils about paths"""
import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.abspath(__file__),  # paths.py
                                            os.path.pardir,             # utils
                                            os.path.pardir,             # step_defs
                                            os.path.pardir,             # system
                                            os.path.pardir,             # tests
                                            os.path.pardir))            # root of project


