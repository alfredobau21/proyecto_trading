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

#*****
    for i,row