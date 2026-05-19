# models/transformer_baseline.py

from sklearn.ensemble import RandomForestClassifier


class TransformerBaseline:

    def __init__(self):

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)