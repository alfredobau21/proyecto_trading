import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna

<<<<<<< HEAD

# load training data
def load_train(file):
    file_mapping = {
        "A1": "data/aapl_project_1m_train.csv",
        "A5": "data/aapl_project_train.csv",
        "B1": "data/btc_project_1m_train.csv",
        "B5": "data/btc_project_train.csv",
    }
    file_name = file_mapping.get(file)
    if not file_name:
        raise ValueError("File not found.")
    data = pd.read_csv(file_name).dropna()
    return data


# get indicators
def calc_indicators(data):
    rsi_indicator = ta.momentum.RSIIndicator(close=data["Close"], window=15)
    data["RSI"] = rsi_indicator.rsi()

    bollinger = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["Bollinger_High"] = bollinger.bollinger_hband()
    data["Bollinger_Low"] = bollinger.bollinger_lband()

    macd = ta.trend.MACD(close=data["Close"], window_fast=10, window_slow=30, window_sign=10)
    data["MACD"] = macd.macd()
    data["Signal_Line"] = macd.macd_signal()

    stoch_indicator = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"],
                                                       window=15, smooth_window=5)
    data["stoch_%K"] = stoch_indicator.stoch()
    data["stoch_%D"] = stoch_indicator.stoch_signal()

    short_ma = ta.trend.SMAIndicator(data["Close"], window=5)
    long_ma = ta.trend.SMAIndicator(data["Close"], window=25)
    data["Short_SMA"] = short_ma.sma_indicator()
    data["Long_SMA"] = long_ma.sma_indicator()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


# create signals
def buy_sell_signals(data, indicators):
    # indicators = {
    #     "RSI": {"buy": rsi_buy_signal, "sell": rsi_sell_signal},
    #     "BOL": {"buy": bol_buy_signal, "sell": bol_sell_signal},
    #     "MACD": {"buy": macd_buy_signal, "sell": macd_sell_signal},
    #     "STOCH": {"buy": stoch_buy_signal, "sell": stoch_sell_signal},
    #     "SMA": {"buy": sma_buy_signal, "sell": sma_sell_signal}
    # }

    for indicator in indicators.keys():
        data[indicator + '_buy_signal'] = data.apply(
            lambda row: indicators[indicator]['buy'](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
        data[indicator + '_sell_signal'] = data.apply(
            lambda row: indicators[indicator]['sell'](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)

    for indicator in indicators.keys():
        data[indicator + '_buy_signal'] = data[indicator + '_buy_signal'].astype(int)
        data[indicator + '_sell_signal'] = data[indicator + '_sell_signal'].astype(int)

    return data, indicators


# activate de indicators
def active(active_indicators, indicator_name):
    if indicator_name in active_indicators:
        active_indicators.append(indicator_name)
    return active_indicators


# RSI SIGNALS
def rsi_buy_signal(row, value, prev_row=None):
    return row.RSI < value


def rsi_sell_signal(row, value, prev_row=None):
    return row.RSI > value


# BOL SIGNALS
def bol_buy_signal(row, prev_row=None):
    return row["Close"] < row["Bollinger_Low"]


def bol_sell_signal(row, prev_row=None):
    return row["Close"] > row["Bollinger_High"]


# MACD SIGNALS
def macd_buy_signal(row, prev_row=None):
    if prev_row is not None:
        return row.MACD > row.Signal_Line and prev_row.MACD < prev_row.Signal_Line
    return False


def macd_sell_signal(row, prev_row=None):
    if prev_row is not None:
        return row.MACD < row.Signal_Line and prev_row.MACD > prev_row.Signal_Line
    return False


# STOCH SIGNALS
def stoch_buy_signal(row, prev_row=None):
    return prev_row is not None and prev_row["stoch_%K"] < prev_row["stoch_%D"] and row["stoch_%K"] > row[
        "stoch_%D"] and row["stoch_%K"] < 20


def stoch_sell_signal(row, prev_row=None):
    return prev_row is not None and prev_row["stoch_%K"] > prev_row["stoch_%D"] and row["stoch_%K"] < row[
        "stoch_%D"] and row["stoch_%K"] > 80


# SMA SIGNALS
def sma_buy_signal(row, prev_row=None):
    return prev_row is not None and prev_row["Long_SMA"] > prev_row["Short_SMA"] and row["Long_SMA"] < row["Short_SMA"]


def sma_sell_signal(row, prev_row=None):
    return prev_row is not None and prev_row["Long_SMA"] < prev_row["Short_SMA"] and row["Long_SMA"] > row["Short_SMA"]


# run the signals
def run_signals(data, indicators):
    for indicator in indicators.keys():
        data[indicator + "_buy_signal"] = data.apply(
            lambda row: indicators[indicator]["buy"](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
        data[indicator + "_sell_signal"] = data.apply(
            lambda row: indicators[indicator]["sell"](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)

    for indicator in indicators.keys():
        data[indicator + "_buy_signal"] = data[indicator + "_buy_signal"].astype(int)
        data[indicator + "_sell_signal"] = data[indicator + "_sell_signal"].astype(int)
    return data


# do trades
def execute_trades(data, active_indicators, best_combination=None, cash=1_000_000, com=0.125 / 100, n_shares=10):
    operations = []
    strategy_value = [cash]

    if best_combination:
        for indicator in best_combination:
            data["total_buy_signals"] = data[[indicator + "_buy_signal" for indicator in best_combination]].sum(axis=1)
            data["total_sell_signals"] = data[[indicator + "_sell_signal" for indicator in best_combination]].sum(
                axis=1)
        total_active_indicators = len(best_combination)
    else:
        for indicator in active_indicators:
            data["total_buy_signals"] = data[[indicator + "_buy_signal" for indicator in active_indicators]].sum(axis=1)
            data["total_sell_signals"] = data[[indicator + "_sell_signal" for indicator in active_indicators]].sum(
                axis=1)
        total_active_indicators = len(active_indicators)

    for i, row in data.iterrows():
        if total_active_indicators <= 2:
            if data.total_buy_signals.iloc[i] == total_active_indicators:
                # if ...: # TODO: HAVE ENOUGH CASH?
                cash, operations = open_operation("long", row, cash, com, n_shares, operations)
            elif data.total_sell_signals.iloc[i] == total_active_indicators:
                cash, operations = open_operation("short", row, cash, com, n_shares, operations)
        else:
            if data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                cash, operations = open_operation("long", row, cash, com, n_shares, operations)
            elif data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                cash, operations = open_operation("short", row, cash, com, n_shares, operations)

        # close the operation if needed, base on TP & SL
        cash, operations = check_close_operations(row, cash, operations, com)

        current_value = cash
        for operation in operations:
            if not operation["closed"]:
                current_value += calculate_operation_value(operation, row["Close"])

        strategy_value.append(current_value)

    return strategy_value


# when we have an open operation
def open_operation(operation_type, row, cash, com, n_shares, operations):
    if operation_type == "long":
        stop_loss = row["Close"] * 0.85
        take_profit = row["Close"] * 1.15
    else:  # short
        stop_loss = row["Close"] * 1.15
        take_profit = row["Close"] * 0.85

    operations.append(
        {"operation_type": operation_type, "bought_at": row["Close"], "timestamp": row.name, "n_shares": n_shares,
         "stop_loss": stop_loss, "take_profit": take_profit, "closed": False})
    if operation_type == "long":
        cash -= row["Close"] * n_shares * (1 + com)
    else:  # short
        cash += row["Close"] * n_shares * (1 - com)  # we get chash for a short position

    return cash, operations


# see if we close the operations or not, depending on our SL and TP
def check_close_operations(row, cash, operations, com):
    for operation in operations:
        if not operation["closed"]:
            if operation["operation_type"] == "long":
                if row["Close"] <= operation["stop_loss"] or row["Close"] >= operation["take_profit"]:
                    operation["closed"] = True
                    cash += row["Close"] * operation["n_shares"] * (1 - com)
            elif operation["operation_type"] == "short":
                if row["Close"] >= operation["stop_loss"] or row["Close"] <= operation["take_profit"]:
                    operation["closed"] = True
                    cash -= row["Close"] * operation["n_shares"] * (1 + com)

    return cash, operations


# check de value of the trade
def calculate_operation_value(operation, cp):  # cp = current price
    if operation["operation_type"] == "long":
        return operation["n_shares"] * cp
    else:  # short
        return (2 * operation["bought_at"] - cp) * operation["n_shares"]  # !!!!


# graphs
def plot_results(strategy_value, data):
    plt.figure(figsize=(12, 5))
    plt.plot(data["Datetime"], strategy_value[1:], label="Value of the Strategy")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


# do combinations of indicators
def combinatinos(data, n, indicators, cash):
    best_result = 0
    best_combination = None
    all_indicators = list(indicators.keys())
    combinations_of_indicators = combinations(all_indicators, n)

    for combination in combinations_of_indicators:
        active_indicators = list(combination)
        strategy_value = execute_trades(data, active_indicators, best_combination, cash)
        result = strategy_value[-1]

        if result > best_result:
            best_result = result
            best_combination = combination

    return best_combination, best_result


# new strategy
def reset_strategy(data, initial_cash=1_000_000):
    active_indicators = []
    operations = []
    cash = initial_cash
    strategy_value = [cash]
    return active_indicators, cash, operations, strategy_value


# optimize parameters for indicators
def optimize_parameters(data, indicators_list, n_trials=100, initial_cash=1_000_000):

    def objective(trial):

        indicators = {
            "RSI": {"buy": rsi_buy_signal, "sell": rsi_sell_signal},
            "BOL": {"buy": bol_buy_signal, "sell": bol_sell_signal},
            "MACD": {"buy": macd_buy_signal, "sell": macd_sell_signal},
            "STOCH": {"buy": stoch_buy_signal, "sell": stoch_sell_signal},
            "SMA": {"buy": sma_buy_signal, "sell": sma_sell_signal}
        }

        indicators = buy_sell_signals(data, indicators)

        active_indicators = []

        for indicator in active_indicators:
            if indicator == 'RSI':
                rsi_window = trial.suggest_int('rsi_window', 5, 30)
                indicators[indicator]['params'] = {'rsi_window': rsi_window}

            elif indicator == 'Bollinger':
                bollinger_window = trial.suggest_int('bollinger_window', 10, 50)
                indicators[indicator]['params'] = {'bollinger_window': bollinger_window}

            elif indicator == 'MACD':
                macd_fast = trial.suggest_int('macd_fast', 10, 20)
                macd_slow = trial.suggest_int('macd_slow', 21, 40)
                macd_sign = trial.suggest_int('macd_sign', 5, 15)
                indicators[indicator]['params'] = {'macd_fast': macd_fast, 'macd_slow': macd_slow,
                                                   'macd_sign': macd_sign}

            elif indicator == 'Stoch':
                stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21)
                stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14)
                stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 14)
                indicators[indicator]['params'] = {'stoch_k_window': stoch_k_window, 'stoch_d_window': stoch_d_window,
                                                   'stoch_smoothing': stoch_smoothing}

            elif indicator == 'SMA':
                short_ma_window = trial.suggest_int('short_ma_window', 5, 20)
                long_ma_window = trial.suggest_int('long_ma_window', 21, 50)
                indicators[indicator]['params'] = {'short_ma_window': short_ma_window, 'long_ma_window': long_ma_window}

        # Execute strategy
        strategy_value = execute_trades(data, active_indicators, cash=initial_cash)
        return strategy_value[-1]

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Get and apply best params for each indicator
    print(f"Mejores parámetros encontrados: {study.best_params}")
    for indicator_name in indicators.keys():
        if study.best_params.get(indicator_name):
            indicators[indicator_name]['params'] = study.best_params[indicator_name]

    return indicators, study.best_value


def test_strategy(data, indicators, best_combination, cash):
    strategy_value = execute_trades(data, indicators, best_combination, cash)
    return strategy_value


# How we do?
def calculate_performance(data_path, cash=1_000_000):
    data = load_data(data_path)
    data = calc_indicators(data)
    data, indicators = buy_sell_signals(data)

    # Optimización de parámetros con Optuna
    indicators, best_value = optimize_parameters(data)

    # Prueba de estrategia
    strategy_value = test_strategy(data, indicators, cash)
    plot_results(strategy_value, data)

    # Calcular rendimiento pasivo
    rend_pasivo = calculate_performance(data_path)
    print(f"Rendimiento pasivo para {data_path}: {rend_pasivo}%")

    return indicators


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    return data


def test():
    test_file_mapping = {
        "A1T": "data/aapl_project_1m_test.csv",
        "A5T": "data/aapl_project_test.csv",
        "B1T": "data/btc_project_1m_test.csv",
        "B5T": "data/btc_project_test.csv"
    }
    for interval, file_path in test_file_mapping.items():
        data = load_data(file_path)
        data = calc_indicators(data)
        data, indicators = buy_sell_signals(data)
        strategy_value = execute_trades(data, list(indicators.keys()))
        plot_results(strategy_value, data)


def main():
    test()


if __name__ == "__main__":
    main()
=======
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
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.run_signals()
        self.best_combination = None
        self.best_value = 0

    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_indicators(self):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=14)
        self.data['RSI'] = rsi_indicator.rsi()

        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=5)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=21)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()

        macd = ta.trend.MACD(close=self.data['Close'], window_slow=26, window_fast=12, window_sign=9)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

        self.data['SAR'] = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'],close=self.data['Close']).psar()

        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],
                                                           close=self.data['Close'], window=14, smooth_window=3)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()

        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    # Gets the buy or sell signal
    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'SAR': {'buy': self.sar_buy_signal, 'sell': self.sar_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal}
        }

    # Depends on each indicator
    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 25

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 75

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

    def sar_buy_signal(self, row, prev_row=None):
        return prev_row is not None and row['SAR'] < row['Close'] and prev_row['SAR'] > prev_row['Close']

    def sar_sell_signal(self, row, prev_row=None):
        return prev_row is not None and row['SAR'] > row['Close'] and prev_row['SAR'] < prev_row['Close']

    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row['stoch_%D'] and row['stoch_%K'] < 20

    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row['stoch_%D'] and row['stoch_%K'] > 80

    def run_signals(self):
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(lambda row: self.indicators[indicator]['buy'](row,self.data.iloc[row.name - 1] if row.name > 0 else None),axis=1)
            self.data[indicator + '_sell_signal'] = self.data.apply(lambda row: self.indicators[indicator]['sell'](row,self.data.iloc[row.name - 1] if row.name > 0 else None),axis=1)

        # make sura buy and sell signals are numeric values (1 = True, 0 = False)
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)

    def execute_trades(self, best=False):
        if best == True:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
                total_active_indicators = len(self.best_combination)

        else:  # False
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[[indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[[indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
                total_active_indicators = len(self.active_indicators)

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

            # See if operations needs to be closed or not
            self.check_close_operations(row)

            # Updates strategy value for each iteraton
            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

    def _open_operation(self, operation_type, row):
        if operation_type == 'long':
            stop_loss = row['Close'] * 0.85
            take_profit = row['Close'] * 1.15
        else:  # 'short'
            stop_loss = row['Close'] * 1.15
            take_profit = row['Close'] * 0.85

        self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * self.n_shares * (1 + self.com)
        else:  # 'short'
            self.cash += row['Close'] * self.n_shares * (1 - self.com) # ***

    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (1 + self.com) # *****

                op.closed = True

    def calculate_operation_value(self, op, current_price):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * op.n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * op.n_shares if not op.closed else 0

    def plot_results(self, best=False):
        self.reset_strategy()
        if best == False:
            self.execute_trades()
        else:
            self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()

    def run_combinations(self):
        all_indicators = list(self.indicators.keys())
        for r in range(1, len(all_indicators) + 1):
            for combo in combinations(all_indicators, r):
                self.active_indicators = list(combo)
                print(f"Using indicators: {self.active_indicators}")
                self.execute_trades()

                final_value = self.strategy_value[-1]
                if final_value > self.best_value:
                    self.best_value = final_value
                    self.best_combination = self.active_indicators.copy()
                self.reset_strategy()

        print(
            f"Best combination of indicators: {self.best_combination} with a value of: {self.best_value}")

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]

    def optimize_parameters(self):
        def objective(trial):
            self.reset_strategy()
            # Gets better params for indicators in best combination
            for indicator in self.best_combination:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 10, 35)
                    self.set_rsi_parameters(rsi_window)
                elif indicator == 'SMA':
                    short_ma_window = trial.suggest_int('short_ma_window', 5, 20)
                    long_ma_window = trial.suggest_int('long_ma_window', 21, 50)
                    self.set_sma_parameters(short_ma_window, long_ma_window)
                elif indicator == 'MACD':
                    macd_fast = trial.suggest_int('macd_fast', 10, 20)
                    macd_slow = trial.suggest_int('macd_slow', 21, 40)
                    macd_sign = trial.suggest_int('macd_sign', 5, 15)
                    self.set_macd_parameters(macd_fast, macd_slow, macd_sign)
                elif indicator == 'SAR':
                    sar_step = trial.suggest_float('sar_step', 0.01, 0.1)
                    sar_max_step = trial.suggest_float('sar_max_step', 0.1, 0.5)
                    self.set_sar_parameters(sar_step, sar_max_step)
                if indicator == 'Stoch':
                    stoch_k_window = trial.suggest_int('stoch_k_window', 5, 21)
                    stoch_d_window = trial.suggest_int('stoch_d_window', 3, 14)
                    stoch_smoothing = trial.suggest_int('stoch_smoothing', 3, 14)

                    self.set_stoch_parameters(stoch_k_window, stoch_d_window, stoch_smoothing)

            # Execute the strategy with best combination
            self.run_signals()
            self.execute_trades(best=True)

            return self.strategy_value[-1]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)  # Number of trails, can add more or less depending on the time you have :)

        # Print and apply the best params for each indicator
        print(f"Best parameters found: {study.best_params}")
        for indicator in self.best_combination:

            if indicator == 'RSI':
                self.set_rsi_parameters(study.best_params['rsi_window'])
            elif indicator == 'SMA':
                self.set_sma_parameters(study.best_params['short_ma_window'], study.best_params['long_ma_window'])
            elif indicator == 'MACD':
                self.set_macd_parameters(study.best_params['macd_fast'], study.best_params['macd_slow'],
                                         study.best_params['macd_sign'])
            elif indicator == 'SAR':
                self.set_sar_parameters(study.best_params['sar_step'], study.best_params['sar_max_step'])
            elif indicator == 'Stoch':
                self.set_stoch_parameters(study.best_params['stoch_k_window'], study.best_params['stoch_d_window'],
                                          study.best_params['stoch_smoothing'])

    def set_rsi_parameters(self, window):
        rsi_indicator = ta.momentum.RSIIndicator(close=self.data['Close'], window=window)
        self.data['RSI'] = rsi_indicator.rsi()

    def set_sma_parameters(self, short_window, long_window):
        short_ma = ta.trend.SMAIndicator(self.data['Close'], window=short_window)
        long_ma = ta.trend.SMAIndicator(self.data['Close'], window=long_window)
        self.data['SHORT_SMA'] = short_ma.sma_indicator()
        self.data['LONG_SMA'] = long_ma.sma_indicator()

    def set_macd_parameters(self, fast, slow, sign):
        macd = ta.trend.MACD(close=self.data['Close'], window_slow=slow, window_fast=fast, window_sign=sign)
        self.data['MACD'] = macd.macd()
        self.data['Signal_Line'] = macd.macd_signal()

    def set_sar_parameters(self, step, max_step):
        sar_indicator = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'], close=self.data['Close'],
                                               step=step, max_step=max_step)
        self.data['SAR'] = sar_indicator.psar()

    def set_stoch_parameters(self, k_window, d_window, smoothing):
        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],
                                                           close=self.data['Close'], window=k_window,
                                                           smooth_window=d_window)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal().rolling(window=smoothing).mean()

    def test(self):
        test_file_mapping = {
            "A1": "data/aapl_project_1m_test.csv",
            "A5": "data/aapl_project_test.csv",
            "B1": "data/btc_project_1m_test.csv",
            "B5": "data/btc_project_test.csv"
        }
        self.load_data(self.file)
        self.calculate_indicators()
        self.define_buy_sell_signals()
        self.run_signals()
        self.execute_trades(best=True)
        plt.figure(figsize=(12, 8))
        plt.plot(self.strategy_value)
        plt.title('Trading Strategy Performance')
        plt.xlabel('Number of Trades')
        plt.ylabel('Strategy Value')
        plt.show()

    def rendimiento(data_path, cash=1000000):
        data = pd.read_csv(data_path)

        # Turn date into datetime
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        # Get first and last close data
        primer_cierre = data.iloc[0]['Close']
        ultimo_cierre = data.iloc[-1]['Close']

        # Get asset yeild
        rend_pasivo = (ultimo_cierre - primer_cierre) / primer_cierre
        print("The passive asset return from the first close to the last close is: {:.2%}".format(rend_pasivo))

        # Compare with used strategy
        cashfinal = 1348097.4477885822  #CHANGE !!!
        rend_estrategia = (cashfinal - cash) / cash
        print("The strategy return from the first close to the last close is: {:.2%}".format(rend_estrategia))

        # Sort data
        data = data.sort_values(by='Date')

        # Rend
        data['Returns'] = data['Close'].pct_change().fillna(0)

        # See the value passive
        initial_investment = cash
        data['Investment_Value'] = (1 + data['Returns']).cumprod() * initial_investment

        # Graficar el rendimiento de la inversión
        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['Investment_Value'], label='Investment Value', color='green')
        plt.title('Investment Return')
        plt.xlabel('Date')
        plt.ylabel('Investment Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        valor_final = data['Investment_Value'].iloc[-1]
        print("The final value of the investment: ${:,.2f}".format(valor_final))
>>>>>>> 967e68842785bb57cc73376858929c28ba703391
