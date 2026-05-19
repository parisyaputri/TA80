# models/lstm_baseline.py

from sklearn.ensemble import IsolationForest


class LSTMBaseline:

    def __init__(self):

        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )

    def fit(self, X):

        self.model.fit(X)

    def predict(self, X):

        preds = self.model.predict(X)

        return [1 if p == -1 else 0 for p in preds]