import joblib
import numpy as np

class HousingServeRF:
    def __init__(self, model_file='models/rf_regressor.dat'):
        self.model = joblib.load(model_file)

    def predict(self, X, feature_names):
        prediction = self.model.predict(X)
        return [[prediction.item(0), prediction.item(0)]]

if __name__ == '__main__':
    serve = HousingServeRF()
    print(serve.predict(np.ndarray([1, 37]), None))
