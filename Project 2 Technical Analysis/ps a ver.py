import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta
from itertools import combinations
import optuna

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
        self.commission = 0.00125
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
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.best_combination = None
        self.best_value = 0

    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_indicators(self):
        # RSI
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=14)
        self.data['RSI'] = rsi_indicator.rsi()

        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(close=self.data['Close'], window=20, window_dev=2)
        self.data['BB_Middle'] = bb_indicator.bollinger_mavg()
        self.data['BB_Upper'] = bb_indicator.bollinger_hband()
        self.data['BB_Lower'] = bb_indicator.bollinger_lband()

        # Moving Averages
        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=5)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=21)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()

        # MACD
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

        # Stochastic Oscillator
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'], window=14, smooth_window=3)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()

        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'BB': {'buy': self.bb_buy_signal, 'sell': self.bb_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal}
        }

    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row['stoch_%D'] and row['stoch_%K'] < 20

    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row['stoch_%D'] and row['stoch_%K'] > 80

    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def bb_buy_signal(self, row, prev_row=None):
        return row.Close < row.BB_Lower

    def bb_sell_signal(self, row, prev_row=None):
        return row.Close > row.BB_Upper

    def sma_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] > prev_row['SHORT_SMA'] and row['LONG_SMA'] < row['SHORT_SMA']

    def sma_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] < prev_row['SHORT_SMA'] and row['LONG_SMA'] > row['SHORT_SMA']

    def macd_buy_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
        return False

    def macd_sell_signal(self, row, prev_row=None):
        if prev_row is not None:
            return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
        return False

    def run_signals(self):
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['buy'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
            self.data[indicator + '_sell_signal'] = self.data.apply(
                lambda row: self.indicators[indicator]['sell'](row, self.data.iloc[row.name - 1] if row.name > 0 else None), axis=1)

        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)

    def execute_trades(self, best=False):
        if best:
            indicators_to_use = self.best_combination
        else:
            indicators_to_use = self.active_indicators

        for indicator in indicators_to_use:
            self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in indicators_to_use]].sum(axis=1)
            self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in indicators_to_use]].sum(axis=1)
            total_active_indicators = len(indicators_to_use)

        for i, row in self.data.iterrows():
            if total_active_indicators <= 2:
                if self.data.total_buy_signals.iloc[i] == total_active_indicators:
                    self._open_operation('long', row)
                elif self.data.total_sell_signals.iloc[i] == total_active_indicators:
                    self._open_operation('short', row)
            else:
                if self.data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row)
                elif self.data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row)

            self.check_close_operations(row)

            total_value = self.cash + sum(
                self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

    def _open_operation(self, operation_type, row):
        if operation_type == 'long':
            stop_loss = row['Close'] * 0.95
            take_profit = row['Close'] * 1.05
        else:  # 'short'
            stop_loss = row['Close'] * 1.05
            take_profit = row['Close'] * 0.95

        self.operations.append(
            Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * self.n_shares * (1 + self.commission)
        else:  # 'short'
            self.cash += row['Close'] * self.n_shares * (1 - self.commission)

    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (
                    row['Close'] <= op.stop_loss or row['Close'] >= op.take_profit)) or
                                  (op.operation_type == 'short' and (
                                          row['Close'] >= op.stop_loss or row['Close'] <= op.take_profit))):
                op.closed = True
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.commission)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.commission)

    def plot_results(self, best=False):
        plt.figure(figsize=(16, 8))
        plt.plot(self.strategy_value)
        plt.xlabel('Time')
        plt.ylabel('Strategy Value ($)')
        plt.title('Strategy Performance Over Time')
        plt.grid(True)
        plt.show()

    def run_combinations(self):
        indicators = list(self.indicators.keys())
        for r in range(2, len(indicators) + 1):
            for comb in combinations(indicators, r):
                self.active_indicators = list(comb)
                self.execute_trades()
                if self.strategy_value[-1] > self.best_value:
                    self.best_combination = self.active_indicators
                    self.best_value = self.strategy_value[-1]

    def reset_strategy(self):
        self.operations = []
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]
        self.best_combination = None
        self.best_value = 0

    def optimize_parameters(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=100)
        best_params = study.best_params
        print(f"Best parameters found: {best_params}")
        self.reset_strategy()
        self.set_indicator_parameters(best_params)
        self.execute_trades(best=True)
        self.plot_results(best=True)

    def set_indicator_parameters(self, params):
        for indicator, values in params.items():
            if indicator == 'RSI':
                self.set_rsi_parameters(values['window'])
            elif indicator == 'SMA':
                self.set_sma_parameters(values['short_window'], values['long_window'])
            elif indicator == 'MACD':
                self.set_macd_parameters(values['fast'], values['slow'], values['signal'])
            elif indicator == 'Stoch':
                self.set_stoch_parameters(values['k_window'], values['d_window'], values['smoothing'])

    def set_rsi_parameters(self, window):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
        self.data['RSI'] = rsi_indicator.rsi()

    def set_sma_parameters(self, short_window, long_window):
        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=short_window)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=long_window)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()

    def set_macd_parameters(self, fast, slow, signal):
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=signal)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

    def set_stoch_parameters(self, k_window, d_window, smoothing):
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],
                                                           close=self.data['Close'], window=k_window,
                                                           smooth_window=smoothing)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()


if __name__ == "__main__":
    strategy = TradingStrategy(file="A5")  # Cambia el marco temporal según tus datos
    strategy.run_combinations()
    strategy.plot_results()
