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

        # RSI Indicator
        if 'RSI' in kwargs:
            rsi_window = kwargs['RSI'][0]
            rsi_1 = ta.momentum.RSIIndicator(data.Close, rsi_window)
            data['rsi'] = rsi_1.rsi()

        # Bollinger Bands Indicator
        if 'Bollinger' in kwargs:
            bollinger_window, bollinger_std = kwargs['Bollinger']
            bollinger = ta.volatility.BollingerBands(data.Close, bollinger_window, bollinger_std)
            data['BUY_SIGNAL'] = (data['rsi'] < kwargs.get('rsi_lower_threshold', 30))
            data['BUY_SIGNAL'] = data['BUY_SIGNAL'] & bollinger.bollinger_lband_indicator().astype(bool)

        # MACD Indicator
        if 'MACD' in kwargs:
            macd_fastperiod, macd_slowperiod, macd_signalperiod = kwargs['MACD']
            macd = ta.trend.MACD(data.Close, macd_fastperiod, macd_slowperiod, macd_signalperiod)
            data['macd'] = macd.macd()
            data['macd_signal'] = macd.macd_signal()
            data['BUY_SIGNAL'] = data['BUY_SIGNAL'] & (data['macd'] > data['macd_signal'])

        # Stochastic Indicator
        if 'Stochastic' in kwargs:
            stochastic_k, stochastic_d = kwargs['Stochastic'][:2]
            stochastic = ta.momentum.StochasticOscillator(data.High, data.Low, data.Close, stochastic_k, stochastic_d)
            data['stochastic'] = stochastic.stoch()
            data['BUY_SIGNAL'] = data['BUY_SIGNAL'] & (data['stochastic'] < 20)

        # SMA Indicator
        if 'SMA' in kwargs:
            sma_period = kwargs['SMA'][0]
            sma = ta.trend.SMAIndicator(data.Close, sma_period)
            data['sma'] = sma.sma_indicator()
            data['BUY_SIGNAL'] = data['BUY_SIGNAL'] & (data.Close > data['sma'])

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

        technical_data = self.create_signals(self.data,
            RSI=self.best_params.get('RSI', (14,)) if self.best_params else (14,),
            Bollinger=self.best_params.get('Bollinger', (20, 2)) if self.best_params else (20, 2),
            MACD=self.best_params.get('MACD', (12, 26, 9)) if self.best_params else (12, 26, 9),
            Stochastic=self.best_params.get('Stochastic', (14, 3, 0)) if self.best_params else (14, 3, 0),
            SMA=self.best_params.get('SMA', (30,)) if self.best_params else (30,)
        )

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

            positions_value = len(active_positions) * n_shares * row.Close
            portfolio_value.append(capital + positions_value)

        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

        portfolio_value.append(capital)
        return portfolio_value[-1]

    def run_optimization(self, strat_list):
        study = optuna.create_study(direction='maximize')
        study.optimize(func=self.profit, n_trials=50)
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
        technical_data = self.create_signals(self.data,
                                             RSI=self.best_params.get('RSI', (14,)),
                                             Bollinger=self.best_params.get('Bollinger', (20, 2)),
                                             MACD=self.best_params.get('MACD', (12, 26, 9)),
                                             Stochastic=self.best_params.get('Stochastic', (14, 3, 0)),
                                             SMA=self.best_params.get('SMA', (30,)))

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

            positions_value = len(active_positions) * n_shares * row.Close
            portfolio_value_active.append(capital + positions_value)

        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

        portfolio_value_active.append(capital)

        # Plotting results
        plt.figure(figsize=(12, 8))
        plt.plot(self.data.Close, label='Price')
        plt.plot(portfolio_value_passive, label='Passive Strategy')
        plt.plot(portfolio_value_active, label='Active Strategy')
        plt.title('Backtest Results')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()

    def show_indicators(self):
        # Function to plot indicators (like RSI, Bollinger Bands, etc.) with price
        fig, axs = plt.subplots(len(self.best_params), 1, figsize=(12, 8), sharex=True)

        for i, (indicator, params) in enumerate(self.best_params.items()):
            if indicator == 'RSI':
                rsi_window, rsi_lower_threshold = params
                rsi_1 = ta.momentum.RSIIndicator(self.data.Close, rsi_window)
                self.data['rsi'] = rsi_1.rsi()
                axs[i].plot(self.data.Close[:214])
                axs[i].set_title('Closing Prices')
                axs[i].plot(self.data.rsi[:214])
                axs[i].plot([0, 214], [70, 70], label="Upper Threshold")
                axs[i].plot([0, 214], [30, 30], label="Lower Threshold")
                axs[i].set_title('RSI')
                axs[i].legend()

            elif indicator == 'Bollinger':
                bollinger_window, bollinger_std = params
                bollinger = ta.volatility.BollingerBands(self.data.Close, bollinger_window, bollinger_std)
                axs[i].plot(self.data.Close[:214])
                axs[i].set_title('Closing Prices')
                axs[i].plot(bollinger.bollinger_mavg()[:214], label='Bollinger Mavg')
                axs[i].fill_between(self.data.index[:214], bollinger.bollinger_hband()[:214],
                                    bollinger.bollinger_lband()[:214], alpha=0.1, color='gray')
                axs[i].set_title('Bollinger Bands')
                axs[i].legend()

            elif indicator == 'MACD':
                macd_fastperiod, macd_slowperiod, macd_signalperiod = params
                macd = ta.trend.MACD(self.data.Close, macd_fastperiod, macd_slowperiod, macd_signalperiod)
                axs[i].plot(self.data.Close[:214])
                axs[i].set_title('Closing Prices')
                axs[i].plot(macd.macd()[:214], label='MACD')
                axs[i].plot(macd.macd_signal()[:214], label='MACD Signal')
                axs[i].set_title('MACD')
                axs[i].legend()

            elif indicator == 'Stochastic':
                stochastic_k, stochastic_d = params[:2]
                stochastic = ta.momentum.StochasticOscillator(self.data.High, self.data.Low, self.data.Close,
                                                              stochastic_k, stochastic_d)
                axs[i].plot(self.data.Close[:214])
                axs[i].set_title('Closing Prices')
                axs[i].plot(stochastic.stoch()[:214], label='Stochastic Oscillator')
                axs[i].plot([0, 214], [80, 80], label="Overbought (80)")
                axs[i].plot([0, 214], [20, 20], label="Oversold (20)")
                axs[i].set_title('Stochastic Oscillator')
                axs[i].legend()

            elif indicator == 'SMA':
                sma_period = params[0]
                sma = ta.trend.SMAIndicator(self.data.Close, sma_period)
                axs[i].plot(self.data.Close[:214])
                axs[i].set_title('Closing Prices')
                axs[i].plot(sma.sma_indicator()[:214], label='SMA')
                axs[i].set_title('SMA')
                axs[i].legend()

        plt.tight_layout()
        plt.show()

    def test(self):
        # Function to load test data and test the optimized strategy
        test_file_mapping = {
            "A1T": "data/aapl_project_1m_test.csv",
            "A5T": "data/aapl_project_test.csv",
            "B1T": "data/btc_project_1m_test.csv",
            "B5T": "data/btc_project_test.csv"
        }
        test_data = pd.read_csv(test_file_mapping[self.file[:-1] + 'T']).dropna()

        capital = 1_000_000
        n_shares = self.best_params['n_shares']
        stop_loss = self.best_params['stop_loss']
        take_profit = self.best_params['take_profit']

        rsi_window = self.best_params.get('RSI', (14,))[0]
        rsi_lower_threshold = self.best_params.get('RSI', (14,))[1] if 'RSI' in self.best_params else 30
        bollinger_window, bollinger_std = self.best_params.get('Bollinger', (20, 2))
        macd_fastperiod, macd_slowperiod, macd_signalperiod = self.best_params.get('MACD', (12, 26, 9))
        stochastic_k, stochastic_d = self.best_params.get('Stochastic', (14, 3, 0))[:2]
        sma_period = self.best_params.get('SMA', (30,))[0]

        COM = 0.125 / 100
        max_active_operations = 1000

        active_positions = []
        strategy_value_passive = [capital]
        strategy_value_active = [capital]

        # Passive strategy (buy and hold)
        for i, row in test_data.iterrows():
            positions_value = len(active_positions) * n_shares * row.Close
            strategy_value_passive.append(capital + positions_value)

        # Active strategy (using optimized parameters)
        technical_data = self.create_signals(test_data,
                                             RSI=(rsi_window, rsi_lower_threshold),
                                             Bollinger=(bollinger_window, bollinger_std),
                                             MACD=(macd_fastperiod, macd_slowperiod, macd_signalperiod),
                                             Stochastic=(stochastic_k, stochastic_d, 0),
                                             SMA=(sma_period,))

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

            positions_value = len(active_positions) * n_shares * row.Close
            strategy_value_active.append(capital + positions_value)

        active_pos_copy = active_positions.copy()
        for pos in active_pos_copy:
            capital += row.Close * pos["n_shares"] * (1 - COM)
            active_positions.remove(pos)

        strategy_value_active.append(capital)

        # Plotting results
        plt.figure(figsize=(12, 8))
        plt.plot(test_data.Close, label='Price')
        plt.plot(strategy_value_passive, label='Passive Strategy')
        plt.plot(strategy_value_active, label='Active Strategy')
        plt.title('Test Results')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    file_type = "A1"  # Example, can be A1, A5, B1, B5 according to desired file
    strategy = TradingStrategy(file_type)

    # Generate strategies
    strat_list = strategy.generate_strategies()

    # Optimize parameters
    best_params = strategy.run_optimization(strat_list)

    # Backtest strategy
    strategy.backtest_strategy(best_params)

    # Show indicators
    strategy.show_indicators()

    # Test strategy
    strategy.test()
