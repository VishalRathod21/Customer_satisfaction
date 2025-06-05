import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
import pandas as pd

class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

class LinearRegressionModel(Model):
    def train(self, X_train, y_train,**kwargs):
        try:
            reg = LinearRegression(**kwargs)
            
            reg.fit(X_train, y_train)
            logging.info("Model training complete")
            return reg
        except Exception as e:
            logging.error(e)
            raise e
        