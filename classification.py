from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from utils import load_and_preprocess_data, load_data


target='Gender'
X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data(target=target)


classifiers = [LDA(), QDA()]
classifier_names = ['LDA', 'QDA']

for clf, name in zip(classifiers, classifier_names):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {name}: {accuracy:.2f}')
