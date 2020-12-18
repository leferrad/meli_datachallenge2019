Feature: Fit and validate a model to predict item categories from text titles
  As a data scientist,
  I want to fit and validate a model to predict categories from text titles of items
  in order to get a classification model with proper results in some datasets

  Background:
    Given the execution profile 'profile_sampled_data'

  Scenario: Execute modeling script with accepted results
    When the modeling script is executed
    Then the model and results are saved
    And the train and valid results pass the acceptance criteria

  Scenario: Execute evaluation script with accepted results
    When the evaluation script is executed
    Then the test results are saved
    And the test results pass the acceptance criteria