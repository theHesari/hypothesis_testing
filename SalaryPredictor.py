import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data
import matplotlib.pyplot as plt
import seaborn as sns


class SalaryPredictor:
    def __init__(self):
        self.preprocessor = None
        self.models = {}

    def load_and_preprocess_data(self):
        df = load_data()
        if df is None:
            return None, None, None, None

        X = df.drop('Starting Salary', axis=1)
        y = df['Starting Salary']

        numeric_features = ['GPA', 'Experience']
        categorical_features = ['Gender', 'Specialization']

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first')

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        if X_train is None:
            return

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }

        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                       ('regressor', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            self.models[name] = pipeline
            print(f"{name} - MSE: {mse}, R^2: {r2}")

    def feature_importance(self):
        model = self.models.get('Random Forest')
        if model is None:
            print("Random Forest model is not trained.")
            return

        feature_names = self.preprocessor.get_feature_names_out()
        importances = model.named_steps['regressor'].feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(len(feature_names)):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.3f})")

    def plot_feature_importances(self):
        model = self.models.get('Random Forest')
        if model is None:
            print("Random Forest model is not trained.")
            return

        feature_names = self.preprocessor.get_feature_names_out()
        importances = model.named_steps['regressor'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

        plt.figure(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=feature_importances, palette='Set2')
        plt.title('Feature Importances in Random Forest Model')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()

    def run(self):
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        self.train_and_evaluate(X_train, X_test, y_train, y_test)
        self.feature_importance()
        self.plot_feature_importances()


# Usage
if __name__ == "__main__":
    predictor = SalaryPredictor()
    predictor.run()
