import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna

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

    macd = ta.trend.MACD(close=data["Close"], w_fast=10, w_slow=30, window_sign=10)
    data["MACD"] = macd.macd()
    data["Signal_Line"] = macd.macd_signal()

    stoch_indicator = ta.momentum.StochasticIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=15, smooth_window=5)
    data["Stoch_%K"] = stoch_indicator.stoch()
    data["Stoch_%D"] = stoch_indicator.stoch_signal()

    short_ma = ta.trend.SMAIndicator(data["Close"], window=5)
    long_ma = ta.trend.SMAIndicator(data["Close"], window=25)
    data["Short_SMA"] = short_ma.sma_indicator()
    data["Long_SMA"] = long_ma.sma_indicator()

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# create signals
def buy_sell_signals():
    indicators = {
        "RSI": {"buy": rsi_buy_signal, "sell":rsi_sell_signal},
        "BOL": {"buy": bol_buy_signal, "sell":bol_sell_signal},
        "MACD": {"buy": macd_buy_signal, "sell": macd_sell_signal},
        "STOCH": {"buy": stoch_buy_signal, "sell": stoch_sell_signal},
        "SMA": {"buy": sma_buy_signal, "sell": sma_sell_signal}
    }
    return indicators

# activate de indicators
def active(active_indicators, indicator_name):
    if indicator_name in active_indicators:
        active_indicators.append(indicator_name)
    return active_indicators

# RSI SIGNALS
def rsi_buy_signal(row, prev_row=None):
    return row.RSI < 20

def rsi_sell_signal(row, prev_row=None):
    return row.RSI < 80

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
    return prev_row is not None and prev_row["Stoch_%K"] < prev_row["Stoch_%D"] and row["Stoch_%K"] > row["Stoch_%D"] and row["stoch_%K"] < 20

def stoch_sell_signal(row, prev_row=None):
    return prev_row is not None and prev_row["Stoch_%K"] > prev_row["Stoch_%D"] and row["Stoch_%K"] < row["Stoch_%D"] and row["stoch_%K"] > 80

# SMA SIGNALS
def sma_buy_signal(row, prev_row=None):
    return prev_row is not None and prev_row["Long_SMA"] > prev_row["Short_SMA"] and row["Long_SMA"] < row["Short_SMA"]

def sma_sell_signal(row, prev_row=None):
    return prev_row is not None and prev_row["Long_SMA"] < prev_row["Short_SMA"] and row["Long_SMA"] > row["Short_SMA"]

# run the signals
def run_signals(data, indicators):
    for indicator in indicators.keys():
        data[indicator + "_buy_signal"] = data.apply(lambda row: indicators[indicator]["buy"](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)
        data[indicator + "_sell_signal"] = data.apply(lambda row: indicators[indicator]["sell"](row, data.iloc[row.name - 1] if row.name > 0 else None), axis=1)

    for indicator in indicators.keys():
        data[indicator + "_buy_signal"] = data[indicator + "_buy_signal"].astype(int)
        data[indicator + "_sell_signal"] = data[indicator + "_sell_signal"].astype(int)
    return data

# do trades
def execute_trades(data, active_indicators, best_combination=None, cash=1_000_000, com=0.125/100, n_shares=10):
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
            data["total_sell_signals"] = data[[indicator + "_sell_signal" for indicator in active_indicators]].sum(axis=1)
        total_active_indicators = len(active_indicators)

    for i, row in data.iterrows():
        if total_active_indicators <=2:
            if data.total_buys_signals.iloc[i] == total_active_indicators:
                cash, operations = open_operation("long", row, cash, com, n_shares, operations)
            elif data.total_sell_signals.iloc[i] == total_active_indicators:
                cash, operations = open_operation("short", row, cash, com, n_shares, operations)
        else:
            if data.total_buys_signals.iloc[i] > (total_active_indicators / 2):
                cash, operations = open_operation("long", row, cash, com, n_shares, operations)
            elif data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                cash, operations = open_operation("short", row, cash, com, n_shares, operations)

        cash, operations = check_close_operations(row, cash, operations, com)
        total_value = cash + sum(calculate_operation_value(op, row["Close"]) for op in operations if not op["closed"])
        strategy_value.append(total_value)

    return strategy_value

# when we have an open operation
def open_operation(operation_type, row, cash, com, n_shares, operations):
    if operation_type == "long":
        stop_loss = row["Close"] * 0.85
        take_profit = row["Close"] * 1.10
    else: # short
        stop_loss = row["Close"] * 1.10
        take_profit = row["Close"] * 0.85

    operations.append({"operation_type": operation_type, "bought_at": row["Close"], "timestamp": row.name, "n_shares": n_shares, "stop_loss": stop_loss, "take_profit": take_profit, "closed": False})
    if operation_type == "long":
        cash -= row["Close"] * n_shares * (1 + com)
    else: # short
        cash += row["Close"] * n_shares * (1 - com) # we get chash for a short position

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
def calculate_operation_value(operation, cp): # cp = current price
    if operation["operation_type"] == "long":
        return operation["n_shares"] * cp
    else: # short
        return (2 * operation["bought_at"] - cp) * operation["n_shares"]

# graphs
def plot_results(strategy_value, data):
    plt.figure(figsize=(12,8))
    plt.plot(data["Date"], strategy_value[1:], label="Value of the Strategy")
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

    return  best_combination, best_result

# new strategy
def reset_strategy(data, initial_cash=1_000_000):
    active_indicators = []
    operations = []
    cash = initial_cash
    strategy_value = [cash]
    return active_indicators, cash, operations, strategy_value

# optimize
def optimize_parameters(data, indicators, n_trails=200, initial_cash=1_000_000):
    def objective(trail):
        active_indicators = []
        for indicator_name in indicators.keys():
            if trail.suggest_categorical(indicator_name, [True, False]):
                active_indicators.append(indicator_name)
        strategy_value = execute_trades(data, active_indicators, cash=initial_cash)
        return strategy_value[-1]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trails=n_trails)
    best_indicators = [indicator for indicator in indicators.keys() if study.best_params[indicator]]
    return best_indicators, study.best_value

# Let's try it!
def test_strategy(data, indicators, best_combination, cash):
    strategy_value = execute_trades(data, indicators, best_combination, cash)
    return strategy_value

# How we do?
def calculate_performance(data_path, cash=1_000_000):
    data = pd.read_csv(data_path)
    data["Date"] = pd.to_datetime(data["Date"])
    first_close = data.iloc[0]["Close"]
    last_close = data.iloc[-1]["Close"]
    rend_p = (last_close - first_close) / first_close * 100
    return rend_p