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
class StrategyOptimization:
    def __init__(self, data):
        self.data = data
        self.indicator_combinations = self.generate_combinations()

    def generate_combinations(self):
        indicators = ["RSI","Bollinger","MACD","Stoch","SMA"]
        all_combo = []

        for i in range(1, len(indicators) + 1):
            all_combo.extend(combinations(indicators, i))

        all_combo = [list[combo] for combo in all_combo]
        # strategy = all_combo.copy()



        # for combination in combinations_of_indicators:
        #     active_indicators = list(combination)
        #     strategy_value = execute_trades(data, active_indicators, best_combination, cash)
        #     result = strategy_value[-1]
        #
        #     if result > best_result:
        #         best_result = result
        #         best_combination = combination
        #
        # return best_combination, best_result


    # new strategy
    def reset_strategy(data, initial_cash=1_000_000):
        active_indicators = []
        operations = []
        cash = initial_cash
        strategy_value = [cash]
        return active_indicators, cash, operations, strategy_value


    # optimize parameters for indicators
    def optimize_parameters(data, indicators_list, n_trials=100, initial_cash=1_000_000):

        def objective(trial,data, self):
            strategy = {}
            if "RSI" in self.indicator_combinations:
                strategy["RSI"] = {
                    "window": trial.suggest_int("rsi_window", 5, 50),
                    "lower_threshold": trial.suggest_int("rsi_lower_threshold", 10, 30)
                }
            if "Bollinger" in self.indicator_combinations:
                strategy["Bollinger"] = {
                    "window": trial.suggest_int("bollinger_window", 5, 50),
                    "std": trial.suggest_float("bollinger_std", 1.5, 3.5)
                }
            if "MACD" in self.indicator_combinations:
                strategy["MACD"] = {
                    "window_slow": trial.suggest_int("macd_window_slow", 20, 50),
                    "window_fast": trial.suggest_int("macd_window_fast", 5, 20),
                    "window_sign": trial.suggest_int("macd_window_sign", 5, 20)
                }
            if "Stoch" in self.indicator_combinations:
                strategy["Stoch"] = {
                    "window": trial.suggest_int("stoch_window", 5, 50),
                    "smooth_window": trial.suggest_int("stoch_smooth_window", 3, 10),
                    "threshold": trial.suggest_float("stoch_threshold", 0.2, 0.8)
                }
            if "SMA" in self.indicator_combinations:
                strategy["SMA"] = {
                    "window": trial.suggest_int("sma_window", 5, 50)
                }
            strategy["n_shares"] = trial.suggest_int("n_shares", 50, 150)
            strategy["stop_loss"] = trial.suggest_float("stop_loss", 0.05, 0.15)
            strategy["take_profit"] = trial.suggest_float("take_profit", 0.05, 0.15)

            backtest = Backtest(self.data, strategy)
            portfolio_value = backtest.run_backtest()
            return portfolio_value[-1]

            def optimize(self):
                study = optuna.create_study(direction='maximize')
                study.optimize(self.objective, n_trials=50)
                return study.best_params
            # strategies = {"RSI": {"window": rsi_window, "lower_threshold": rsi_lower_threshold},
            #          "Bollinger": {"window": boll_window, "std": std_boll},
            #          "MACD": {"slow": MACD_slow, "fast": MACD_fast, "sign": MACD_sign},
            #          "Stoch": {"window": stoch_window, "smooth": stoch_smooth, "threshold": stoch_threshold},
            #          "SMA": {"window": sma_window}}
            strategy = {"RSI": {"window": 14, "lower_threshold": 30},
                          "Bollinger": {"window": 20, "std": 2},
                          "MACD": {"slow": 26, "fast": 12, "sign": 9},
                          "Stoch": {"window": 14, "smooth": 3, "threshold": 0.2},
                          "SMA": {"window": 30}
                          "n_shares": 100,
                          "stop_loss": 0.1,
                          "take_profit": 0.2
                          }
        def generate_signals(data, strategies):
            if "RSI" in strategies.keys():
                rsi_ta = ta.momentum.RSI(data["Close"], window=strategies["RSI"]["window"])
                rsi_df = rsi_ta.rsi()



            #
            # strategy["n_shares"] = trial.suggest_int("n_shares", 50, 150)
            # strategy["stop_loss"] = trial.suggest_float("stop_loss", 0.05, 0.15)
            # strategy["take_profit"] = trial.suggest_float("take_profit", 0.05, 0.15)

            backtest = Backtest(self.data, strategy)
            portfolio_value = backtest.run_backtest()
            return portfolio_value[-1]
        # "window": trial.suggest_int("rsi_window", 5, 50),
        # "lower_threshold": trial.suggest_int("rsi_lower_threshold", 10, 30)
        # strategies["Bollinger"] = {
        #     "window": trial.suggest_int("bollinger_window", 5, 50),
        #     "std": trial.suggest_float("bollinger_std", 1.5, 3.5)
        # }
        # if "MACD" in self.indicator_combinations:
        #     strategies["MACD"] = {
        #         "window_slow": trial.suggest_int("macd_window_slow", 20, 50),
        #         "window_fast": trial.suggest_int("macd_window_fast", 5, 20),
        #         "window_sign": trial.suggest_int("macd_window_sign", 5, 20)
        #     }
        # if "Stoch" in self.indicator_combinations:

            # indicators = {
            #     "RSI": {"buy": rsi_buy_signal, "sell": rsi_sell_signal},
            #     "BOL": {"buy": bol_buy_signal, "sell": bol_sell_signal},
            #     "MACD": {"buy": macd_buy_signal, "sell": macd_sell_signal},
            #     "STOCH": {"buy": stoch_buy_signal, "sell": stoch_sell_signal},
            #     "SMA": {"buy": sma_buy_signal, "sell": sma_sell_signal}
            # }

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
