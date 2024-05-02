from dataset import Dataset


def load_data():
    """Load/Generate the dataset """
    dataset = Dataset(n=1000, gender_bias=20000)
    print('Loading Dataset... ')
    df = dataset.get_dataframe()
    return df
