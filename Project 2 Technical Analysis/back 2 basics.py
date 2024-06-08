import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna


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
            "B5": "data/btc_project_train.csv",
        }
        file_name = file_mapping.get(self.file)
        if not file_name:
            raise ValueError("File not found.")
        data = pd.read_csv(file_name).dropna()
        return data

    def generate_strategies(self):
        # Generate all possible combinations of indicators
        indicators = ['RSI', 'Bollinger', 'MACD', 'Stochastic', 'SMA']
        strat_list = []
        for r in range(1, len(indicators) + 1):
            combinations_list = combinations(indicators, r)
            for combo in combinations_list:
                strat_list.append(list(combo))
        return strat_list

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
            data['SELL_SIGNAL'] |= (data['rsi'] > 70)

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

    def profit(self, trial):
        capital = 1_000_000
        n_shares = trial.suggest_int("n_shares", 50, 150)
        stop_loss = trial.suggest_float("stop_loss", 0.05, 0.15)
        take_profit = trial.suggest_float("take_profit", 0.05, 0.15)
        max_active_operations = 1000
        COM = 0.125 / 100

        active_positions = []
        portfolio_value = [capital]

        # Iterate over all combinations of strategies
        strat_list = self.generate_strategies()
        for strategy in strat_list:
            kwargs = {}
            for indicator in strategy:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30)
                    rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
                    kwargs['RSI'] = (rsi_window, rsi_lower_threshold)
                elif indicator == 'Bollinger':
                    bollinger_window = trial.suggest_int('bollinger_window', 10, 50)
                    bollinger_std = 2  # Assuming std is 2
                    kwargs['Bollinger'] = (bollinger_window, bollinger_std)
                elif indicator == 'MACD':
                    macd_fastperiod = trial.suggest_int('macd_fast', 10, 20)
                    macd_slowperiod = trial.suggest_int('macd_slow', 21, 40)
                    macd_signalperiod = trial.suggest_int('macd_sign', 5, 15)
                    kwargs['MACD'] = (macd_fastperiod, macd_slowperiod, macd_signalperiod)
                elif indicator == 'Stochastic':
                    stochastic_k = trial.suggest_int('stoch_k_window', 5, 21)
                    stochastic_d = trial.suggest_int('stoch_d_window', 3, 14)
                    kwargs['Stochastic'] = (stochastic_k, stochastic_d, 0)  # Assuming smoothing is 0
                elif indicator == 'SMA':
                    short_ma_window = trial.suggest_int('short_ma_window', 5, 20)
                    long_ma_window = trial.suggest_int('long_ma_window', 21, 50)
                    kwargs['SMA'] = (short_ma_window, long_ma_window)

            # Generate technical data with signals for current combination
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
                            capital += row.Close * pos["n_shares"] * (1 - COM)
                            active_positions.remove(pos)
                        elif row.Close < pos["take_profit"]:
                            capital += row.Close * pos["n_shares"] * (1 - COM)
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
                    if capital > row.Close * (1 + COM) * n_shares:
                        capital -= row.Close * (1 + COM) * n_shares
                        active_positions.append({
                            "type": "SHORT",
                            "sold_at": row.Close,
                            "n_shares": n_shares,
                            "stop_loss": row.Close * (1 + stop_loss),
                            "take_profit": row.Close * (1 - take_profit)
                        })

                positions_value = len(active_positions) * n_shares * row.Close
                portfolio_value.append(capital + positions_value)

            # Clean up remaining active positions at the end of each strategy combination
            active_pos_copy = active_positions.copy()
            for pos in active_pos_copy:
                capital += row.Close * pos["n_shares"] * (1 - COM)
                active_positions.remove(pos)

            portfolio_value.append(capital)

        return portfolio_value[-1]

    def run_optimization(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.profit, n_trials=10)
        self.best_params = study.best_params
        return self.best_params

    def backtest_strategy(self, best_params):
        capital = 1_000_000
        n_shares = best_params['n_shares']
        stop_loss = best_params['stop_loss']
        take_profit = best_params['take_profit']
        max_active_operations = 1000
        COM = 0.125 / 100

        active_positions = []
        portfolio_value_passive = [capital]
        portfolio_value_active = [capital]

        # Passive strategy (buy and hold)
        for i, row in self.data.iterrows():
            positions_value = len(active_positions) * n_shares * row.Close
            portfolio_value_passive.append(capital + positions_value)

        # Active strategy (using optimized parameters)
        kwargs = {}
        for indicator, value in best_params.items():
            if indicator == 'RSI':
                kwargs['RSI'] = value
            elif indicator == 'Bollinger':
                kwargs['Bollinger'] = value
            elif indicator == 'MACD':
                kwargs['MACD'] = value
            elif indicator == 'Stochastic':
                kwargs['Stochastic'] = value
            elif indicator == 'SMA':
                kwargs['SMA'] = value

        technical_data = self.create_signals(self.data, **kwargs)

        for i, row in technical_data.iterrows():
            active_pos_copy = active_positions.copy()
            for pos in active_pos_copy:
                if row.Close < pos["stop_loss"]:
                    capital += row.Close * pos["n_shares"] * (1 - COM)
                    active_positions.remove(pos)
                if row.Close > pos["take_profit"]:
                    capital += row.Close * pos["n_shares"] * (1 - COM)
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
                if capital > row.Close * (1 + COM) * n_shares:
                    capital -= row.Close * (1 + COM) * n_shares
                    active_positions.append({
                        "type": "SHORT",
                        "sold_at": row.Close,
                        "n_shares": n_shares,
                        "stop_loss": row.Close * (1 + stop_loss),
                        "take_profit": row.Close * (1 - take_profit)
                    })

            positions_value = len(active_positions) * n_shares * row.Close
            portfolio_value_active.append(capital + positions_value)

        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

        portfolio_value_active.append(capital)

        # Plot results
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_value_passive, label='Passive Strategy')
        plt.plot(portfolio_value_active, label='Active Strategy')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()


# Ejemplo de uso:
strategy = TradingStrategy('A1')  # Aquí se puede especificar el archivo que deseas cargar
best_params = strategy.run_optimization()
strategy.backtest_strategy(best_params)

