import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix, mean_squared_error

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.closed = False

class TradingStrategy:
    def __init__(self, file):
        self.data = None
        self.operations = []
        self.cash = 1_000_000
        self.com = 0.00125
        self.strategy_value = [1_000_000]
        self.n_shares = 10
        self.file = file
        self.file_mapping = {
            "A1": "data/aapl_project_1m_train.csv",
            "A5": "data/aapl_project_train.csv",
            "B1": "data/btc_project_1m_train.csv",
            "B5": "data/btc_project_train.csv"
        }
        self.load_data(self.file)
        self.indicators = {}
        self.active_indicators = []
        self.calculate_variables()
        self.best_combination = None
        self.best_value = 0
        self.stop_loss = 0.85
        self.stop_loss = 1.15

    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_variables(self):
        self.data = self.data.loc[:, ["Close"]]

        # 20 Lagged close values
        for i in range(1, 21):
            self.data[f"Close_t{i}"] = self.data["Close"].shift(i)

        # Some of the best indicators on our Technical Analysis
        # Stochastic K and D
        self.data["Stoch_K_16"] = ta.momentum.stochrsi_k(self.data["Close"], window=16, smooth1=3, smooth2=3)
        self.data["Stoch_K_18"] = ta.momentum.stochrsi_k(self.data["Close"], window=18, smooth1=3, smooth2=3)
        self.data["Stoch_D_5"] = ta.momentum.stochrsi_d(self.data["Close"], window=16, smooth1=3, smooth2=3,smooth3=5)
        self.data["Stoch_D_7"] = ta.momentum.stochrsi_d(self.data["Close"], window=16, smooth1=3, smooth2=3,smooth3=7)

        # RSI
        self.data["RSI_10"] = ta.momentum.rsi(self.data["Close"], window=10)
        self.data["RSI_11"] = ta.momentum.rsi(self.data["Close"], window=11)
        self.data["RSI_15"] = ta.momentum.rsi(self.data["Close"], window=15)

        # Volatility & Returns
        self.data["Returns"] = self.data["Close"].pct_change()
        self.data["Volatility"] = self.data["Returns"].rolling(window=14).std()

        #Ewma
        self.data["EMA_20"] = ta.trend.ema_indicator(self.data["Close"], window=20)

        self.data.dropna(inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def prepare_data(self, train_size = 0.8):
