from SalaryPredictor import SalaryPredictor
from hypothesis import run_hypothesis


def run():
    # Perform hypothesis testing to examine potential salary disparities by gender
    run_hypothesis()

    # Initialize the salary predictor model to assess the impact of various features, including gender
    predictor = SalaryPredictor()

    # Execute the salary predictor's operations which include data loading, model training, and feature importance
    # evaluation
    predictor.run()


if __name__ == '__main__':
    run()
