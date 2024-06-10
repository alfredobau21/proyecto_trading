import pandas as pd
import matplotlib.pyplot as plt
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
