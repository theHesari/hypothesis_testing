from scipy import stats
from dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_data():
    """Load/Generate the dataset """
    dataset = Dataset(n=1000, gender_bias=-3100)
    print('Loading Dataset... ')
    df = dataset.get_dataframe()
    return df


def describe_gender_data(df):
    """Calculate proportions, mean, std, and SE of salaries for each gender."""
    # Gender counts and proportions
    gender_counts = df['Gender'].value_counts()
    total_count = df['Gender'].count()
    proportions = gender_counts / total_count

    # Group by gender and calculate statistics
    grouped = df.groupby('Gender')['Starting Salary']
    mean_salaries = grouped.mean()
    std_salaries = grouped.std()
    std_error_salaries = grouped.sem()  # Standard error of the mean

    # Assemble the results
    summary = pd.DataFrame({
        'Proportion': proportions,
        'Mean Salary': mean_salaries,
        'STD Salary': std_salaries,
        'SE Salary': std_error_salaries
    })

    return summary


def plot_gender_salary_boxplot(df):
    """
    Plots a boxplot of starting salaries by gender.

    Args:
    df (pandas.DataFrame): The data frame containing the 'Gender' and 'Starting Salary' columns.
    """
    plt.figure(figsize=(8, 6))  # Set the figure size for better readability
    sns.boxplot(x='Gender', y='Starting Salary', data=df, palette='Set2')
    plt.title('Starting Salary Distribution by Gender')  # Title for the boxplot
    plt.xlabel('Gender')  # Label for the X-axis
    plt.ylabel('Starting Salary ($)')  # Label for the Y-axis
    plt.grid(True)  # Add grid for better readability of plot
    plt.show()  # Display the plot


def check_assumptions(male_salaries, female_salaries):
    """Check for normality and homogeneity of variances."""
    # Normality test
    normality_male = stats.shapiro(male_salaries)
    normality_female = stats.shapiro(female_salaries)
    print(f"Normality test results -- Males: {normality_male}, Females: {normality_female}")

    # Variance equality test
    variance_test = stats.levene(male_salaries, female_salaries)
    print(f"Levene's test result for equal variances: {variance_test}")

    return normality_male, normality_female, variance_test


def perform_ttest(male_salaries, female_salaries):
    """Perform an independent samples t-test between two arrays."""
    equal_var = True
    _, p_value_var = check_assumptions(male_salaries, female_salaries)[2]
    if p_value_var < 0.05:
        equal_var = False

    t_stat, p_value = stats.ttest_ind(male_salaries, female_salaries, equal_var=equal_var)
    return t_stat, p_value


def main():
    # Load the dataset
    df = load_data()
    if df is None:
        return

    # Get descriptive statistics
    stats_summary = describe_gender_data(df)
    print(stats_summary)

    # Plot the boxplot for salary distribution by gender
    # plot_gender_salary_boxplot(df)

    # Prepare data
    male_salaries = df[df['Gender'] == 'Male']['Starting Salary']
    female_salaries = df[df['Gender'] == 'Female']['Starting Salary']

    # Perform hypothesis testing
    t_stat, p_value = perform_ttest(male_salaries, female_salaries)
    print(f"T-Statistic: {t_stat}, P-value: {p_value}")

    # Decision
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis - Significant difference in salaries.")
    else:
        print("Fail to reject the null hypothesis - No significant difference in salaries.")


if __name__ == "__main__":
    main()
