import pandas as pd
import matplotlib.pyplot as plt
import ta
import optuna

class TechnicalData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.technical_data = self.extract_technical_data()

    def load_data(self):
        return pd.read_csv(self.file_path).dropna()

    def extract_technical_data(self):
        technical_data = pd.DataFrame()
        technical_data["Close"] = self.data["Close"]
        return technical_data

technical_data_A1 = TechnicalData("data/aapl_project_1m_train.csv")
technical_data_A5 = TechnicalData("data/aapl_project_train.csv")
technical_data_B1 = TechnicalData("data/btc_project_1m_train.csv")
technical_data_B5 = TechnicalData("data/btc_project_train.csv")