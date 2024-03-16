from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import pandas as pd
import numpy as np
import os
import shutil
import pickle
import hashlib
from datetime import datetime
from api.utils import prepare_logger
from fastapi import HTTPException
from joblib import Parallel, delayed
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

logger = prepare_logger()


class RepresentativenessModel:
    models = None

    def __init__(self):
        self.set_paths()

    def load_models(self, token: str):
        """
        Load enseble model by token
        - token: ensemble model token
        """
        logger.info(f"Loading ensemble model {token}")
        if token not in os.listdir(self.models_path):
            message = "Ensemble model not found"
            logger.error(message)
            raise HTTPException(status_code=500, detail=message)
        if not (models_list := os.listdir(self.models_path / token)):
            message = f"Empty ensemble model {token}"
            logger.error(message)
            raise HTTPException(status_code=500, detail=message)
        models = []
        for model in models_list:
            model_path = self.models_path / token / model
            models.append(pickle.load(open(model_path, "rb")))
            logger.info(f"Model {model} loaded")
        self.models = models
        self.n_features = models[0].n_features_in_
        logger.info("Ensemble model loaded successfully")

    def predict(self, input: np.array) -> np.array:
        """
        Makes ensemble model predictions
        - input: array of objects for which predictions are to be made
        """
        if not all(
            isinstance(object, list)
            and len(object) == self.n_features
            and all(isinstance(element, (int, float)) for element in object)
            for object in input
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input (objects should be represented by {self.n_features}-number arrays)",
            )
        preds = []
        for object in input:
            logger.info(f"Ensemble model prediction for object {object}")
            pred = self.ensemble_predict(object)
            preds.append(pred)
            logger.info(f"Ensemble model prediction: {pred}")
        return preds

    def ensemble_predict(self, input: np.array) -> float:
        """
        Makes single ensemble model prediction
        Ensemble model prediction is mean of all models predicitions
        - input: single objects for which prediction is to be made
        """
        preds = []
        for idx, model in enumerate(self.models):
            pred = model.predict([input])
            preds.append(pred)
            logger.info(f"Model {idx} prediction: {pred[0]}")
        return np.mean(preds)

    def make_ensemble(self, data: np.array, n_split: int, n_nearest: int) -> str:
        """
        Creates new ensemble model
        - data: training data
        - n_split: number of splits of input data (number of models in ensemble)
        - n_nearest: k parameter for KNN
        """
        logger.info("Starting new ensemble model training")
        token = self.get_token()
        logger.info(f"New ensemble model token: {token}")
        os.makedirs(self.models_path / token, exist_ok=True)
        data_slices = self.split_input_data(data=data, n_split=n_split)
        self.run_training(
            token=token,
            data=data_slices,
            n_nearest=n_nearest,
        )
        return token
    
    def run_training(self, token: str, data: list[pd.DataFrame], n_nearest: int) -> None:
        """
        Starts parallel training of models in ensemble
        - token: ensemble model token
        - data: list of training datasets
        - n_nearest: k parameter for KNN
        """
        if any([n_nearest > len(X) for X in data]):
            raise Exception("Provided data and parameters require lower 'n_nearest'")
        self.models = Parallel(n_jobs=-1)(
            delayed(self.train)(idx,df,n_nearest,token) for idx, df in enumerate(data)
        )
        self.n_features = self.models[0].n_features_in_
        logger.info("Ensemble model training finished")
    
    def train(self, idx: int, X: pd.DataFrame, n_nearest: int, token: str) -> DecisionTreeRegressor:
        """
        Trains single model (in this case DecisionTreeRegressor)
        - idx: id of a single model in an ensemble
        - X: training dataset
        - n_nearest: k parameter for KNN 
        - token: ensemble model token
        """
        neighbors = NearestNeighbors(n_neighbors=n_nearest).fit(X)
        distances, _ = neighbors.kneighbors(X)
        distances_avg = np.mean(distances[:, 1:], axis=1)
        y = 1 / (1 + distances_avg)
        dtr = DecisionTreeRegressor()
        dtr = dtr.fit(X, y)
        model_name = f"model_{idx}.pickle"
        pickle.dump(dtr, open(self.models_path / token / model_name, "wb"))
        logger.info(f"New model trained: {model_name}")
        return dtr

    @staticmethod
    def split_input_data(data: np.array, n_split: int) -> np.array:
        """
        Randomly shuffles dataset and splits it equally
        - data: dataset to shuffle and split
        - n_split: number of splits
        """
        shuffled = pd.DataFrame(data).sample(frac=1)
        return np.array_split(shuffled, n_split)

    @staticmethod
    def get_token() -> str:
        """
        Generates unique token based on current timestamp
        """
        time_string = datetime.now().isoformat()
        token = hashlib.sha256(time_string.encode()).hexdigest()
        return token

    def set_paths(self) -> None:
        """
        Creates directory to store trained models
        """
        self.models_path = Path(__file__).resolve().parents[1] / "models"
        os.makedirs(self.models_path, exist_ok=True)

    def delete(self, keep: list = []) -> None:
        """
        Deletes ensemble models
        - keep: tokens of models to keep
        """
        for element in os.listdir(self.models_path):
            element_path = os.path.join(self.models_path, element)
            if os.path.isdir(element_path):
                if element not in keep:
                    logger.info(f"Deleting {element}")
                    shutil.rmtree(element_path)
