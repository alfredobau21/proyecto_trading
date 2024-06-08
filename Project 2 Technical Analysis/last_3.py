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
        # self.sold_at = None
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

        self.data['SAR'] = ta.trend.PSARIndicator(high=self.data['High'], low=self.data['Low'],
                                                  close=self.data['Close']).psar()

        stoch_indicator = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],
                                                           close=self.data['Close'], window=14, smooth_window=3)
        self.data['stoch_%K'] = stoch_indicator.stoch()
        self.data['stoch_%D'] = stoch_indicator.stoch_signal()

        self.data.dropna(inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def define_buy_sell_signals(self):
        self.indicators = {
            'RSI': {'buy': self.rsi_buy_signal, 'sell': self.rsi_sell_signal},
            'SMA': {'buy': self.sma_buy_signal, 'sell': self.sma_sell_signal},
            'MACD': {'buy': self.macd_buy_signal, 'sell': self.macd_sell_signal},
            'SAR': {'buy': self.sar_buy_signal, 'sell': self.sar_sell_signal},
            'Stoch': {'buy': self.stoch_buy_signal, 'sell': self.stoch_sell_signal}
        }

    def activate_indicator(self, indicator_name):
        if indicator_name in self.indicators:
            self.active_indicators.append(indicator_name)

    def stoch_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] < prev_row['stoch_%D'] and row['stoch_%K'] > row[
            'stoch_%D'] and row['stoch_%K'] < 20

    def stoch_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['stoch_%K'] > prev_row['stoch_%D'] and row['stoch_%K'] < row[
            'stoch_%D'] and row['stoch_%K'] > 80

    def rsi_buy_signal(self, row, prev_row=None):
        return row.RSI < 30

    def rsi_sell_signal(self, row, prev_row=None):
        return row.RSI > 70

    def sma_buy_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] > prev_row['SHORT_SMA'] and row['LONG_SMA'] < row[
            'SHORT_SMA']

    def sma_sell_signal(self, row, prev_row=None):
        return prev_row is not None and prev_row['LONG_SMA'] < prev_row['SHORT_SMA'] and row['LONG_SMA'] > row[
            'SHORT_SMA']

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

    def run_signals(self):

        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data.apply(lambda row: self.indicators[indicator]['buy'](row,
                                                                                                                 self.data.iloc[
                                                                                                                     row.name - 1] if row.name > 0 else None),
                                                                   axis=1)
            self.data[indicator + '_sell_signal'] = self.data.apply(lambda row: self.indicators[indicator]['sell'](row,
                                                                                                                   self.data.iloc[
                                                                                                                       row.name - 1] if row.name > 0 else None),
                                                                    axis=1)

        # Asegurarse de que las señales de compra y venta se conviertan a valores numéricos (1 para True, 0 para False)
        for indicator in list(self.indicators.keys()):
            self.data[indicator + '_buy_signal'] = self.data[indicator + '_buy_signal'].astype(int)
            self.data[indicator + '_sell_signal'] = self.data[indicator + '_sell_signal'].astype(int)

    def execute_trades(self, best=False):

        if best == True:
            for indicator in self.best_combination:
                self.data['total_buy_signals'] = self.data[
                    [indicator + '_buy_signal' for indicator in self.best_combination]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[
                    [indicator + '_sell_signal' for indicator in self.best_combination]].sum(axis=1)
                total_active_indicators = len(self.best_combination)


        else:  # False
            for indicator in self.active_indicators:
                self.data['total_buy_signals'] = self.data[
                    [indicator + '_buy_signal' for indicator in self.active_indicators]].sum(axis=1)
                self.data['total_sell_signals'] = self.data[
                    [indicator + '_sell_signal' for indicator in self.active_indicators]].sum(axis=1)
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

            # Verifica y cierra operaciones basadas en stop_loss o take_profit
            self.check_close_operations(row)

            # Actualiza el valor de la estrategia en cada iteración
            total_value = self.cash + sum(
                self.calculate_operation_value(op, row['Close']) for op in self.operations if not op.closed)
            # print(f"Fila: {i}, Valor de la estrategia: {total_value}")
            self.strategy_value.append(total_value)

    def _open_operation(self, operation_type, row):
        if operation_type == 'long':
            stop_loss = row['Close'] * 0.95
            take_profit = row['Close'] * 1.05
        else:  # 'short'
            stop_loss = row['Close'] * 1.05
            take_profit = row['Close'] * 0.95

        self.operations.append(Operation(operation_type, row['Close'], row.name, self.n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * self.n_shares * (1 + self.com)
        else:  # 'short'
            self.cash += row['Close'] * self.n_shares * (
                        1 - self.com)  # Incrementa el efectivo al abrir la venta en corto

        # print(f"Operación {operation_type} iniciada en {row.name}, Precio: {row['Close']}, Cash restante: {self.cash}")

    def check_close_operations(self, row):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (
                    row['Close'] >= op.take_profit or row['Close'] <= op.stop_loss)) or
                                  (op.operation_type == 'short' and (
                                          row['Close'] <= op.take_profit or row['Close'] >= op.stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * op.n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * op.n_shares * (
                                1 + self.com)  # Decrementa el efectivo al cerrar la venta en corto, basado en el nuevo precio

                op.closed = True
                # print(f"Operación {op.operation_type} cerrada en {row.name}, Precio: {row['Close']}, Cash resultante: {self.cash}")

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
                print(f"Ejecutando con combinación de indicadores: {self.active_indicators}")
                self.execute_trades()

                final_value = self.strategy_value[-1]
                if final_value > self.best_value:
                    self.best_value = final_value
                    self.best_combination = self.active_indicators.copy()
                self.reset_strategy()

        print(
            f"Mejor combinación de indicadores: {self.best_combination} con un valor de estrategia de: {self.best_value}")

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]

    def optimize_parameters(self):
        def objective(trial):
            self.reset_strategy()
            # Configura los parámetros para cada indicador activo en la mejor combinación
            for indicator in self.best_combination:
                if indicator == 'RSI':
                    rsi_window = trial.suggest_int('rsi_window', 5, 30)
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

            # Ejecutar la estrategia con la mejor combinación y los nuevos parámetros
            self.run_signals()
            self.execute_trades(best=True)
            # print(len(self.strategy_value))

            return self.strategy_value[-1]

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)  # Ajusta el número de pruebas según sea necesario

        # Imprimir y aplicar los mejores parámetros encontrados para cada indicador
        print(f"Mejores parámetros encontrados: {study.best_params}")
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
            "A1T": "data/aapl_project_1m_test.csv",
            "A5T": "data/aapl_project_test.csv",
            "B1T": "data/btc_project_1m_test.csv",
            "B5T": "data/btc_project_test.csv"
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

    def show_RSI(self):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        axs[0].plot(self.data.Close[:214])
        axs[0].set_title('Closing Prices')

        axs[1].plot(self.data.RSI[:214])
        axs[1].plot([0, 214], [70, 70], label="Upper Threshold")
        axs[1].plot([0, 214], [30, 30], label="Lower Threshold")
        axs[1].set_title('RSI')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    def show_SMAs(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.Close[:250], label='price')
        plt.plot(self.data.SHORT_SMA[:250], label='SMA(5)')
        plt.plot(self.data.LONG_SMA[:250], label='SMA(21)')
        plt.legend()
        plt.show()

    def show_MACD(self):
        plt.figure(figsize=(12, 8))

        # Plot the MACD
        plt.plot(self.data.index[:214], self.data['MACD'][:214], label='MACD', color='blue')

        # Plot the signal line
        plt.plot(self.data.index[:214], self.data['Signal line'][:214], label='Signal Line', color='red')

        # Add title and legend
        plt.title('MACD and Signal Line')
        plt.legend()

        plt.show()

        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['Signal line']

        plt.figure(figsize=(12, 6))

        # Plot the MACD and the signal line
        plt.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data.index, self.data['Signal_Line'], label='Signal Line', color='red')

        # Fill the histogram between MACD and the signal line
        # We will use a different color depending on whether the histogram is positive or negative
        plt.bar(self.data.index, self.data['MACD_Histogram'], label='MACD Histogram',
                color=['green' if val >= 0 else 'red' for val in self.data['MACD_Histogram']])

        # Add title and legend
        plt.title('MACD, Signal Line, and Histogram')
        plt.legend()
        plt.xlim(100, 200)
        plt.ylim(-1, 1)
        plt.show()

    def show_SAR(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.data.Close.iloc[:214], label='Close Price')

        legend_added_buy = False
        legend_added_sell = False

        for i in range(214):
            if self.data.SAR.iloc[i] < self.data.Close.iloc[i]:
                color = 'green'
                label = 'Buy Order' if not legend_added_buy else None
                legend_added_buy = True
            else:
                color = 'red'
                label = 'Sell Order' if not legend_added_sell else None
                legend_added_sell = True

            plt.scatter(self.data.index[i], self.data.SAR.iloc[i], color=color, s=20, label=label)

        if not legend_added_buy:
            plt.scatter([], [], color='green', s=20, label='Buy Order')
        if not legend_added_sell:
            plt.scatter([], [], color='red', s=20, label='Sell Order')

        plt.title('SAR Indicator with Closing Prices')
        plt.legend()
        plt.show()

    def plot_stochastic_signals(self):

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[:250], self.data['Close'][:250], label='Close Price')

        # Señales de compra
        plt.scatter(self.data.index[:250][self.data['Buy_Signal'][:250] == 1],
                    self.data['Close'][:250][self.data['Buy_Signal'][:250] == 1], color='green', marker='^',
                    label='Buy Signal')
        # Señales de venta
        plt.scatter(self.data.index[:250][self.data['Sell_Signal'][:250] == 1],
                    self.data['Close'][:250][self.data['Sell_Signal'][:250] == 1], color='red', marker='v',
                    label='Sell Signal')

        plt.title('Stochastic Buy/Sell Signals')
        plt.xlabel('Index')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def rendimiento(data_path, cash=1000000):
        # Leer el archivo CSV
        data = pd.read_csv(data_path)

        # Convertir la columna 'Date' a tipo datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Obtener el precio de cierre del primer y último dato
        primer_cierre = data.iloc[0]['Close']
        ultimo_cierre = data.iloc[-1]['Close']

        # Calcular el rendimiento del activo
        rend_pasivo = (ultimo_cierre - primer_cierre) / primer_cierre
        print("The passive asset return from the first close to the last close is: {:.2%}".format(rend_pasivo))

        # Comparativa con la estrategia utilizada
        cashfinal = 1107153.05  # Modifica este valor según lo que obtengas
        rend_estrategia = (cashfinal - cash) / cash
        print("The strategy return from the first close to the last close is: {:.2%}".format(rend_estrategia))

        # Ordenar los datos por fecha si no están ordenados
        data = data.sort_values(by='Date')

        # Graficar el precio de cierre del activo
        plt.figure(figsize=(12, 8))
        plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        plt.title('Close Price of the Asset')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Calcular el rendimiento acumulado
        data['Returns'] = data['Close'].pct_change().fillna(0)

        # Calcular el valor acumulado
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
