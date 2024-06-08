import pandas as pd
import matplotlib.pyplot as plt
import ta
import optuna
from itertools import combinations

class TradingStrategy:
    def __init__(self, file):
        self.file = file
        self.data = self.load_train()
        self.best_params = None

    def load_train(self):
        file_mapping = {
            "A1": "data/aapl_project_1m_train.csv",
            "A5": "data/aapl_project_train.csv",
            "B1": "data/btc_project_1m_train.csv",
            "B5": "data/btc_project_train.csv"
        }
        file_name = file_mapping.get(self.file)
        if not file_name:
            raise ValueError("File not found.")
        data = pd.read_csv(file_name).dropna()
        return data

    def create_signals(self, data, **kwargs):
        data = data.copy()

        # Initialize BUY_SIGNAL and SELL_SIGNAL columns
        data['BUY_SIGNAL'] = False
        data['SELL_SIGNAL'] = False

        # RSI Indicator
        if 'RSI' in kwargs:
            rsi_window, rsi_lower_threshold = kwargs['RSI']
            rsi_1 = ta.momentum.RSIIndicator(data.Close, rsi_window)
            data['rsi'] = rsi_1.rsi()
            data['BUY_SIGNAL'] |= (data['rsi'] < rsi_lower_threshold)
            data['SELL_SIGNAL'] |= (data['rsi'] > 80)

        # Bollinger Bands Indicator
        if 'Bollinger' in kwargs:
            bollinger_window, bollinger_std = kwargs['Bollinger']
            bollinger = ta.volatility.BollingerBands(data.Close, bollinger_window, bollinger_std)
            data['bollinger_hband_indicator'] = bollinger.bollinger_hband_indicator()
            data['bollinger_lband_indicator'] = bollinger.bollinger_lband_indicator()
            data['BUY_SIGNAL'] |= data['bollinger_lband_indicator'].astype(bool)
            data['SELL_SIGNAL'] |= data['bollinger_hband_indicator'].astype(bool)

        # MACD Indicator
        if 'MACD' in kwargs:
            macd_fastperiod, macd_slowperiod, macd_signalperiod = kwargs['MACD']
            macd = ta.trend.MACD(data.Close, macd_fastperiod, macd_slowperiod, macd_signalperiod)
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['BUY_SIGNAL'] |= (data['macd'] > data['macd_signal'])
            data['SELL_SIGNAL'] |= (data['macd'] < data['macd_signal'])

        # Stochastic Indicator
        if 'Stochastic' in kwargs:
            stochastic_k, stochastic_d = kwargs['Stochastic'][:2]
            stochastic = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, stochastic_k, stochastic_d)
            data['stochastic'] = stochastic.stoch()
            data['BUY_SIGNAL'] |= (data['stochastic'] < 20)
            data['SELL_SIGNAL'] |= (data['stochastic'] > 80)

        # SMA Indicator
        if 'SMA' in kwargs:
            sma_period = kwargs['SMA'][0]
            sma = ta.trend.SMAIndicator(data.Close, sma_period)
            data['sma'] = sma.sma_indicator()
            data['BUY_SIGNAL'] |= (data['Close'] > data['sma'])
            data['SELL_SIGNAL'] |= (data['Close'] < data['sma'])

        return data.dropna()

    def profit(self, trial, kwargs):
        capital = 1_000_000
        n_shares = trial.suggest_int("n_shares", 50, 150)
        stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
        take_profit = trial.suggest_float("take_profit", 0.05, 0.15)
        max_active_operations = 1000
        COM = 0.125 / 100

        active_positions = []
        portfolio_value = [capital]

        technical_data = self.create_signals(self.data, **kwargs)

        for i, row in technical_data.iterrows():
            active_pos_copy = active_positions.copy()
            for pos in active_pos_copy:
                if pos["type"] == "LONG":
                    if row.Close < pos["stop_loss"]:
                        capital += row.Close * pos["n_shares"] * (1 - COM)
                        active_positions.remove(pos)
                    elif row.Close > pos["take_profit"]:
                        capital += row.Close * pos["n_shares"] * (1 - COM)
                        active_positions.remove(pos)
                elif pos["type"] == "SHORT":
                    if row.Close > pos["stop_loss"]:
                        capital += (pos["sold_at"] - row.Close) * pos["n_shares"]
                        active_positions.remove(pos)
                    elif row.Close < pos["take_profit"]:
                        capital += (pos["sold_at"] - row.Close) * pos["n_shares"]
                        active_positions.remove(pos)

            if row.BUY_SIGNAL and len(active_positions) < max_active_operations:
                if capital > row.Close * (1 + COM) * n_shares:
                    capital -= row.Close * (1 + COM) * n_shares
                    active_positions.append({
                        "type": "LONG",
                        "bought_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 - stop_loss),
                        "take_profit": row.Close * (1 + take_profit)
                    })

            if row.SELL_SIGNAL and len(active_positions) < max_active_operations:
                if capital > row.Close * COM * n_shares:
                    capital -= row.Close * COM * n_shares
                    active_positions.append({
                        "type": "SHORT",
                        "sold_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 + stop_loss),
                        "take_profit": row.Close * (1 - take_profit)
                    })

            positions_value = sum((pos["n_shares"] * row.Close if pos["type"] == "LONG"
                                  else pos["n_shares"] * (pos["sold_at"] - row.Close)) for pos in active_positions)
            portfolio_value.append(capital + positions_value)

        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            if pos["type"] == "LONG":
                capital += row.Close * pos["n_shares"] * (1 - COM)
            elif pos["type"] == "SHORT":
                capital += (pos["sold_at"] - row.Close) * pos["n_shares"]
            active_positions.remove(pos)

        portfolio_value.append(capital)

        return portfolio_value[-1]

    def optimize_strategy(self, indicators, n_trials_per_strategy=2):
        best_params = {}
        study = optuna.create_study(direction='maximize')

        def objective(trial, indicators):
            strategy_indicators = list(indicators)
            kwargs = {}
            for indicator in strategy_indicators:
                if indicator == 'RSI':
                    rsi_window = 14
                    rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
                    kwargs['RSI'] = (rsi_window, rsi_lower_threshold)
                elif indicator == 'Bollinger Bands':
                    bollinger_window = trial.suggest_int("bollinger_window", 5, 50)
                    kwargs['Bollinger'] = (bollinger_window, 2)  # Assuming std is 2
                elif indicator == 'MACD':
                    macd_fastperiod = trial.suggest_int("macd_fastperiod", 10, 20)
                    macd_slowperiod = trial.suggest_int("macd_slowperiod", 21, 40)
                    macd_signalperiod = trial.suggest_int("macd_signalperiod", 5, 15)
                    kwargs['MACD'] = (macd_fastperiod, macd_slowperiod, macd_signalperiod)
                elif indicator == 'Stochastic':
                    stochastic_k = trial.suggest_int("stochastic_k", 5, 21)
                    stochastic_d = trial.suggest_int("stochastic_d", 3, 14)
                    kwargs['Stochastic'] = (stochastic_k, stochastic_d)
                elif indicator == 'SMA':
                    sma_period = trial.suggest_int("sma_period", 5, 50)
                    kwargs['SMA'] = (sma_period,)

            return self.profit(trial, kwargs)

        for i in range(1, len(indicators) + 1):
            for subset in combinations(indicators, i):
                study.optimize(lambda trial: objective(trial, subset), n_trials=n_trials_per_strategy)
                best_params[subset] = study.best_params

        self.best_params = best_params
        return best_params

    def backtest_strategy(self, test_file):
        test_data = pd.read_csv(test_file).dropna()
        best_strategy = max(self.best_params, key=lambda k: self.profit_dummy(test_data, self.best_params[k]))

        best_indicators = list(best_strategy)
        best_params = self.best_params[best_strategy]
        print(f"Best Strategy: {best_indicators} with params {best_params}")

        backtest_data = self.create_signals(test_data, **best_params)
        backtest_data['strategy_returns'] = backtest_data['Close'].pct_change().shift(-1)
        backtest_data['strategy_cumulative_returns'] = (1 + backtest_data['strategy_returns']).cumprod()

        backtest_data['buy_hold_returns'] = backtest_data['Close'].pct_change().shift(-1)
        backtest_data['buy_hold_cumulative_returns'] = (1 + backtest_data['buy_hold_returns']).cumprod()

        plt.figure(figsize=(14, 7))
        plt.plot(backtest_data.index, backtest_data['strategy_cumulative_returns'], label='Strategy Returns')
        plt.plot(backtest_data.index, backtest_data['buy_hold_cumulative_returns'], label='Buy and Hold Returns')
        plt.title('Strategy vs Buy and Hold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.show()

# Ejemplo de uso:
strategy = TradingStrategy('A5')  # Cambia el archivo según tu necesidad
indicadores = ['RSI', 'Bollinger Bands', 'MACD', 'Stochastic', 'SMA']  # Indicadores disponibles
best_params = strategy.optimize_strategy(indicadores)
print(best_params)
strategy.backtest_strategy('data/aapl_project_1m_test.csv')
