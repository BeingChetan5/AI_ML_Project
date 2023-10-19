# Train different machine learning model and check accuracy score
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, training_array, test_array):
        try:
            logging.info("Initiating model trainer..")
            X_train, y_train, X_test, y_test = (
                training_array[:, :-1],
                training_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Initializing Models...")
            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors Classifier': KNeighborsRegressor(),
                'AdaBoost Classifier': AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models)
            
            # To get best model score from model report dict
            best_model_score = max(sorted(model_report.values()))

            # To Get best fit model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No Best Model Found..')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted_best_model = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted_best_model)
            return r2score
        except Exception as e:
            raise CustomException(e, sys)