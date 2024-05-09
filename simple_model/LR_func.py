from joblib import load
import numpy as np


model_loc_name = 'location_logistic_regression.joblib'
model_vec_name = 'vector_logistic_regression.joblib'
model_meta_name = 'combined_logistic_regression.joblib'


class CustomLRModel:
    def __init__(self, loc_model, vec_model, meta_model):
        self.preds = None
        self.loc_model = loc_model
        self.vec_model = vec_model
        self.meta_model = meta_model

    @classmethod
    def load_model(self, path):
        model_loc = load(path + model_loc_name)
        model_vec = load(path + model_vec_name)
        model_meta = load(path + model_meta_name)

        return CustomLRModel(model_loc, model_vec, model_meta)

    def predict(self, x_loc, x_vec):
        loc_preds = self.loc_model.predict_proba(x_loc)[:, 1]
        vec_preds = self.vec_model.predict_proba(x_vec)[:, 1]
        x = np.column_stack((loc_preds, vec_preds))

        preds = self.meta_model.predict(x)
        self.preds = preds
        return preds

