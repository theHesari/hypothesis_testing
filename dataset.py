import os

import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, n=1000, gender_bias=5000, dataset_root='Data', dataset_name='dataset.csv'):
        """
        Initialize the Dataset class with a default size and gender bias.

        Parameters:
        - n (int): Number of records in the dataset.
        - gender_bias (float): Amount added to the salary of one gender to simulate bias.
        """
        self.n = n
        self.gender_bias = gender_bias
        self.df = self._generate_data()
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name


    def _generate_data(self):
        """
        Generate the dataset with specified biases and distributions.

        Returns:
        - df (DataFrame): A pandas DataFrame containing the generated data.
        """
        # Random choices for categorical data
        genders = np.random.choice(['Male', 'Female'], size=self.n)
        specializations = np.random.choice(['Finance', 'Marketing', 'Operations', 'HR', 'IT'], size=self.n)

        # Numeric data distributions
        gpas = np.round(np.random.normal(3.5, 0.25, self.n).clip(2.0, 4.0), 2)
        experiences = np.random.poisson(5, self.n).clip(0, 10)

        # Salary calculations
        base_salary = 50000 + 2000 * experiences + 3000 * (gpas - 2.0)
        salary_noise = np.random.normal(0, 3000, self.n)
        salaries = base_salary + np.where(genders == 'Male', self.gender_bias, 0) + salary_noise
        salaries = np.round(salaries, -3)  # Round to the nearest thousand

        # Create DataFrame
        df = pd.DataFrame({
            'Gender': genders,
            'GPA': gpas,
            'Experience': experiences,
            'Specialization': specializations,
            'Starting Salary': salaries
        })

        return df

    def get_dataframe(self):
        """
        Return the generated DataFrame.

        Returns:
        - df (CSV) : The dataframe generated previously and saved in CSV format
        - df (DataFrame): The DataFrame stored in the instance.

        """
        # path = os.path.join(self.dataset_root, self.dataset_name)
        # if os.path.exists(path):
        #     print('Reading dataset from CSV...')
        #     return pd.read_csv(path)
        # else:
        #     print('Generating a new dataset...')
        #     self.save()
        #     return self.df
        return self.df

    def describe_data(self):
        """
        Print descriptive statistics of the dataset.
        """
        return self.df.describe()

    def head(self, n=5):
        """
        Return the first n rows of the DataFrame.

        Parameters:
        - n (int): Number of rows to return.

        Returns:
        - (DataFrame): The first n rows of the DataFrame.
        """
        return self.df.head(n)

    def save(self):
        """
        Save the DataFrame to a CSV file within the 'Data' directory.

        Parameters:
        - filename (str): Name of the file to save the data.
        """
        # Check if the directory 'Data' exists, and create it if it doesn't
        if not os.path.exists(self.dataset_root):
            os.makedirs(self.dataset_root)

        # Define the path to save the file
        path = os.path.join(self.dataset_root, self.dataset_name)

        # Save the DataFrame to CSV
        self.df.to_csv(path, index=False)
        print(f'Data saved to {path}')

# Example of using the Dataset class
dataset = Dataset(n=1000, gender_bias=5000)
print('Dataset: ')
print(dataset.get_dataframe())
print(dataset.head())
print(dataset.describe_data())
