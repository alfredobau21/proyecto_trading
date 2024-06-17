import pandas as pd
import ta
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class TradingStrategy:
    def __init__(self, train_path, test_path):
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.cash = 100000
        self.operations = []
        self.strategy_value = [self.cash]
        self.stop_loss = 0.903418952201153 # Change
        self.take_profit = 1.1315385529293625 # Change
        self.n_shares = 50 # Change
        self.com = 0.00125

    def calculate_features(self, data):
        for i in range(1, 21):
            data[f"Close_t{i}"] = data["Close"].shift(i)

        stoch_indicator_16 = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'],
                                                              close=data['Close'], window=16, smooth_window=3)
        data['Stoch_K_16'] = stoch_indicator_16.stoch()
        data['Stoch_D_16'] = stoch_indicator_16.stoch_signal()

        stoch_indicator_18 = ta.momentum.StochasticOscillator(high=data['High'], low=data['Low'],
                                                              close=data['Close'], window=18, smooth_window=3)
        data['Stoch_K_18'] = stoch_indicator_18.stoch()
        data['Stoch_D_18'] = stoch_indicator_18.stoch_signal()

        rsi_indicator_10 = ta.momentum.RSIIndicator(close=data['Close'], window=10)
        data['RSI_10'] = rsi_indicator_10.rsi()

        rsi_indicator_11 = ta.momentum.RSIIndicator(close=data['Close'], window=11)
        data['RSI_11'] = rsi_indicator_11.rsi()

        rsi_indicator_15 = ta.momentum.RSIIndicator(close=data['Close'], window=15)
        data['RSI_15'] = rsi_indicator_15.rsi()

        data['Returns'] = data["Close"].pct_change()
        data['Volatility'] = data["Returns"].rolling(window=10).std()
        data['EMA_20'] = ta.trend.ema_indicator(data["Close"], window=20)

        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])

        data['MACD'] = ta.trend.macd(data['Close'])
        data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])

        data['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])

        data['Engulfing'] = np.where((data['Open'] < data['Close'].shift(1)) & (data['Close'] > data['Open'].shift(1)),
                                     1, 0)
        data['Doji'] = np.where(np.abs(data['Close'] - data['Open']) <= (data['High'] - data['Low']) * 0.1, 1, 0)

        data['buy_s'] = (data.Close < data.Close.shift(-10)).astype(int)
        data['sell_s'] = (data.Close > data.Close.shift(-10)).astype(int)

        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def train_model(self):
        self.train_data = self.calculate_features(self.train_data)
        self.test_data = self.calculate_features(self.test_data)

        features = ['Close_t1', 'Close_t2', 'Close_t3', 'Close_t4', 'Close_t5',
                    'Close_t6', 'Close_t7', 'Close_t8', 'Close_t9', 'Close_t10',
                    'Close_t11', 'Close_t12', 'Close_t13', 'Close_t14', 'Close_t15',
                    'Close_t16', 'Close_t17', 'Close_t18', 'Close_t19', 'Close_t20',
                    'Stoch_K_16', 'Stoch_K_18', 'Stoch_D_16', 'Stoch_D_18', 'RSI_10',
                    'RSI_11', 'RSI_15', 'Returns', 'Volatility', 'EMA_20', 'ATR',
                    'MACD', 'MACD_Signal', 'ADX', 'Engulfing', 'Doji', 'buy_s', 'sell_s']

        X_train = self.train_data[features]
        y_train_buy = self.train_data['buy_s']
        y_train_sell = self.train_data['sell_s']

        X_test = self.test_data[features]
        y_test_buy = self.test_data['buy_s']
        y_test_sell = self.test_data['sell_s']

        # CHANGE IF YOU GET A DIFFERENT MODEL, and also change params
        svm_buy = SVC(C=9.858561168648484, kernel='rbf', gamma=1.5129335462716913e-05, max_iter=10_000).fit(X_train, y_train_buy)
        y_pred_buy = svm_buy.predict(X_test)
        buy_accuracy = accuracy_score(y_test_buy, y_pred_buy)
        buy_f1 = f1_score(y_test_buy, y_pred_buy)
        print(f'Buy Signal Accuracy: {buy_accuracy}')
        print(f'Buy Signal F1 Score: {buy_f1}')

        # CHANGE IF YOU GET A DIFFERENT MODEL, and also change params
        svm_sell = SVC(C=24818.142277526218, kernel='linear', max_iter=10_000).fit(X_train, y_train_sell)
        y_pred_sell = svm_sell.predict(X_test)
        sell_accuracy = accuracy_score(y_test_sell, y_pred_sell)
        sell_f1 = f1_score(y_test_sell, y_pred_sell)
        print(f'Sell Signal Accuracy: {sell_accuracy}')
        print(f'Sell Signal F1 Score: {sell_f1}')

        self.test_data['buy_signal'] = y_pred_buy
        self.test_data['sell_signal'] = y_pred_sell

    def execute_trades(self, stop_loss=None, take_profit=None, n_shares=None):

        stop_loss = stop_loss or self.stop_loss
        take_profit = take_profit or self.take_profit
        n_shares = n_shares or self.n_shares

        self.test_data['total_buy_signals'] = self.test_data['buy_signal']
        self.test_data['total_sell_signals'] = self.test_data['sell_signal']

        print(self.test_data)

        for i, row in self.test_data.iterrows():
            if self.test_data.total_buy_signals.iloc[i] > 0:
                self._open_operation('long', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
            elif self.test_data.total_sell_signals.iloc[i] > 0:
                self._open_operation('short', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)

            self.check_close_operations(row, stop_loss, take_profit, n_shares)

            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close'], n_shares) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

    def _open_operation(self, operation_type, row, stop_loss, take_profit, n_shares):
        if operation_type == 'long':
            cost = row['Close'] * n_shares * (1 + self.com)
            if self.cash >= cost:  # Check if we have enough cash to open a long position
                stop_loss = row['Close'] * stop_loss
                take_profit = row['Close'] * take_profit
                self.operations.append(Operation(operation_type, row['Close'], row.name, n_shares, stop_loss, take_profit))
                self.cash -= cost
        else:  # 'short'
            stop_loss = row['Close'] * take_profit
            take_profit = row['Close'] * stop_loss
            self.operations.append(Operation(operation_type, row['Close'], row.name, n_shares, stop_loss, take_profit))
            self.cash += row['Close'] * n_shares * (1 - self.com)

    def check_close_operations(self, row, stop_loss, take_profit, n_shares):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= take_profit or row['Close'] <= stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= take_profit or row['Close'] >= stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * n_shares * (1 + self.com)

                op.closed = True

    def calculate_operation_value(self, op, current_price, n_shares):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * n_shares if not op.closed else 0

    def plot_results(self):
        self.reset_strategy()
        self.execute_trades()
        final_portfolio_value = self.strategy_value[-1]
        print(f'Final Portfolio Value: {final_portfolio_value}')

        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()

    def reset_strategy(self):
        self.cash = 100000
        self.operations = []
        self.strategy_value = [self.cash]

class Operation:
    def __init__(self, operation_type, bought_at, timestamp, n_shares, stop_loss, take_profit):
        self.operation_type = operation_type
        self.bought_at = bought_at
        self.timestamp = timestamp
        self.n_shares = n_shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.closed = False

