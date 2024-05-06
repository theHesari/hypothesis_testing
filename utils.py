from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from dataset import Dataset


def load_data():
    """Load/Generate the dataset """
    dataset = Dataset(n=1000, gender_bias=20000)
    print('Loading Dataset... ')
    df = dataset.get_dataframe()
    return df


def load_and_preprocess_data(target):
    # Load the dataset using a utility function
    df = load_data()
    if df is None:
        return None, None, None, None

    # Specify numeric and categorical features
    numeric_features = ['GPA', 'Experience', 'Starting Salary']
    categorical_features = ['Gender', 'Specialization']

    # Remove target from features if it is present
    if target in numeric_features:
        numeric_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    # Define transformations for numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first')

    # Setup the ColumnTransformer with appropriate transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Define features and target variable
    X = df.drop(target, axis=1)
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor
