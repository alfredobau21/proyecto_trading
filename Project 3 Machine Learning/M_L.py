import pandas as pd
import matplotlib.pyplot as plt
import ta
from itertools import combinations
import optuna
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
        self.active_indicators = []
        self.calculate_variables()
        self.best_combination = None
        self.best_value = 0
        self.stop_loss = 0.85
        self.take_profit = 1.15
        self.n_shares = 10
        self.best_buylog_params = None
        self.best_selllog_params = None

    def load_data(self, time_frame):
        file_name = self.file_mapping.get(time_frame)
        if not file_name:
            raise ValueError("Unsupported time frame.")
        self.data = pd.read_csv(file_name)
        self.data.dropna(inplace=True)

    def calculate_variables(self):
        # 20 Lagged close values
        for i in range(1, 21):
            self.data[f"Close_t{i}"] = self.data["Close"].shift(i)

        # Some of the best indicators we got on our Technical Analysis
        # Stochastic K and D
        stoch_indicator_16 = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],close=self.data['Close'], window=16, smooth_window=3)
        self.data['Stoch_K_16'] = stoch_indicator_16.stoch()
        self.data['Stoch_D_16'] = stoch_indicator_16.stoch_signal()

        stoch_indicator_18 = ta.momentum.StochasticOscillator(high=self.data['High'], low=self.data['Low'],close=self.data['Close'], window=18, smooth_window=3)
        self.data['Stoch_K_18'] = stoch_indicator_18.stoch()
        self.data['Stoch_D_18'] = stoch_indicator_18.stoch_signal()

        # RSI
        rsi_indicator_10 = ta.momentum.RSIIndicator(close=self.data['Close'], window=10)
        self.data['RSI_10'] = rsi_indicator_10.rsi()

        rsi_indicator_11 = ta.momentum.RSIIndicator(close=self.data['Close'], window=11)
        self.data['RSI_11'] = rsi_indicator_11.rsi()

        rsi_indicator_15 = ta.momentum.RSIIndicator(close=self.data['Close'], window=15)
        self.data['RSI_15'] = rsi_indicator_15.rsi()

        # Volatility & Returns
        self.data['Returns'] = self.data["Close"].pct_change()
        self.data['Volatility'] = self.data["Returns"].rolling(window=10).std()

        # Ewma
        self.data['EMA_20'] = ta.trend.ema_indicator(self.data["Close"], window=20)

        # Signals
        self.data['buy_s'] = (self.data.Close < self.data.Close.shift(-20)).astype(int)
        self.data['sell_s'] = (self.data.Close > self.data.Close.shift(-20)).astype(int)

        self.data.dropna(inplace=True)

        self.data.reset_index(drop=True, inplace=True)

    def prepare_data(self, train_size=0.75):

        features = ['Close_t1', 'Close_t2', 'Close_t3', 'Close_t4', 'Close_t5',
                     'Close_t6', 'Close_t7', 'Close_t8', 'Close_t9', 'Close_t10',
                     'Close_t11', 'Close_t12', 'Close_t13', 'Close_t14', 'Close_t15',
                     'Close_t16', 'Close_t17', 'Close_t18', 'Close_t19', 'Close_t20',
                     'Stoch_K_16', 'Stoch_K_18', 'Stoch_D_16', 'Stoch_D_18', 'RSI_10',
                     'RSI_11', 'RSI_15', 'Returns', 'Volatility', 'EMA_20', 'buy_s', 'sell_s']

        self.X = self.data[features]

        cut = int(len(self.X) * (train_size))

        # Create a single DataFrame for the training set including both features and targets
        self.train_df = self.X.iloc[:cut]
        self.X_train_xgb = self.train_df.drop(['buy_s', 'sell_s'], errors='ignore', axis=1)
        self.Y_train_xgb_buy = self.train_df['buy_s']
        self.Y_train_xgb_sell = self.train_df['sell_s']

        # Create a single DataFrame for the test set including both features and targets
        self.test_df = self.X.iloc[cut:]
        self.X_test_xgb = self.test_df.drop(['buy_s', 'sell_s'], errors='ignore', axis=1)
        self.Y_test_xgb_buy = self.test_df['buy_s']
        self.Y_test_xgb_sell = self.test_df['sell_s']

    def xgb_model(self, direction='buy'):
        # Check if is a buy or a sell
        X_train = self.X_train_xgb
        y_train = self.Y_train_xgb_buy if direction == 'buy' else self.Y_train_xgb_sell
        X_val = self.X_test_xgb
        y_val = self.Y_test_xgb_buy if direction == 'buy' else self.Y_test_xgb_sell

        def objective_xgb(trial):

            booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])

            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.3),
                'booster': booster,
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            }

            if booster != 'gblinear':
                param['max_depth'] = trial.suggest_int('max_depth', 5, 20)
                param['max_leaves'] = trial.suggest_int('max_leaves', 5, 60)
                param['gamma'] = trial.suggest_float('gamma', 0.0, 5.0)

            model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')
            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_xgb, n_trials=25)

        # Store the best parameters
        if direction == 'buy':
            self.best_xgbuy_params = study.best_params
        elif direction == 'sell':
            self.best_xgsell_params = study.best_params

        # Train on the whole training dataset
        best_params = study.best_params
        best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        best_model.fit(X_train, y_train)

        # Predictions
        X_total = self.X.drop(['buy_s', 'sell_s'], axis=1, errors='ignore')
        predictions = best_model.predict(X_total)

        # Add predictions back to the dataset
        if direction == 'buy':
            self.data['XGBoost_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['XGBoost_sell_signal'] = predictions

    def svm_model(self, direction='buy'):
        # Check if is a buy or a sell
        X_train = self.X_train_xgb
        y_train = self.Y_train_xgb_buy if direction == 'buy' else self.Y_train_xgb_sell
        X_val = self.X_test_xgb
        y_val = self.Y_test_xgb_buy if direction == 'buy' else self.Y_test_xgb_sell

        def objective_svm(trial):

            C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            gamma = 'scale' if kernel == 'linear' else trial.suggest_float('gamma', 1e-6, 1e+1, log=True)
            model = SVC(C=C, kernel=kernel, gamma=gamma, max_iter=10_000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_svm, n_trials=20)

        # Store the best params
        if direction == 'buy':
            self.best_svmbuy_params = study.best_params
        elif direction == 'sell':
            self.best_svmsell_params = study.best_params

        # Train on the whole training dataset
        best_params = study.best_params
        best_model = SVC(**best_params, max_iter=10_000)
        best_model.fit(X_train, y_train)

        # Predictions
        X_total = self.X.drop(['buy_s', 'sell_s'], axis=1, errors='ignore')
        predictions = best_model.predict(X_total)

        # Add predictions back to the dataset
        if direction == 'buy':
            self.data['SVM_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['SVM_sell_signal'] = predictions


    def lr_model(self, direction='buy'):
        # Check if is a buy or a sell
        X_train = self.X_train_xgb
        y_train = self.Y_train_xgb_buy if direction == 'buy' else self.Y_train_xgb_sell
        X_val = self.X_test_xgb
        y_val = self.Y_test_xgb_buy if direction == 'buy' else self.Y_test_xgb_sell

        def objective(trial):
            C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0, 0.999)
            fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])

            model = LogisticRegression(C=C, fit_intercept=fit_intercept, penalty='elasticnet', l1_ratio=l1_ratio,
                                       solver='saga', max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='binary')

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25)

        # Store the best params
        if direction == 'buy':
            self.best_buylog_params = study.best_params
        elif direction == 'sell':
            self.best_selllog_params = study.best_params

        # Train on the whole training dataset
        best_log_params = study.best_params
        best_model = LogisticRegression(**best_log_params, penalty='elasticnet', solver='saga', max_iter=10_000)
        best_model.fit(X_train, y_train)

        # Predictions
        X_total = self.X.drop(['buy_s', 'sell_s'], axis=1, errors='ignore')
        predictions = best_model.predict(X_total)

        if direction == 'buy':
            self.data['Logistic_buy_signal'] = predictions
        elif direction == 'sell':
            self.data['Logistic_sell_signal'] = predictions

    def optimize_and_fit_models(self):
        self.prepare_data()

        self.lr_model(direction='buy')
        self.lr_model(direction='sell')

        self.xgb_model(direction='buy')
        self.xgb_model(direction='sell')

        self.svm_model(direction='buy')
        self.svm_model(direction='sell')

    def execute_trades(self, best=False, stop_loss=None, take_profit=None, n_shares=None):

        stop_loss = stop_loss or self.stop_loss
        take_profit = take_profit or self.take_profit
        n_shares = n_shares or self.n_shares

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
                    self._open_operation('long', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
                elif self.data.total_sell_signals.iloc[i] == total_active_indicators:
                    self._open_operation('short', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
            else:
                if self.data.total_buy_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('long', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)
                elif self.data.total_sell_signals.iloc[i] > (total_active_indicators / 2):
                    self._open_operation('short', row, stop_loss=stop_loss, take_profit=take_profit, n_shares=n_shares)

            self.check_close_operations(row, stop_loss, take_profit, n_shares)

            total_value = self.cash + sum(self.calculate_operation_value(op, row['Close'], n_shares) for op in self.operations if not op.closed)
            self.strategy_value.append(total_value)

    def _open_operation(self, operation_type, row, stop_loss, take_profit, n_shares):
        if operation_type == 'long':
            stop_loss = row['Close'] * stop_loss
            take_profit = row['Close'] * take_profit
        else:  # 'short'
            stop_loss = row['Close'] * take_profit
            take_profit = row['Close'] * stop_loss

        self.operations.append(Operation(operation_type, row['Close'], row.name, n_shares, stop_loss, take_profit))
        if operation_type == 'long':
            self.cash -= row['Close'] * n_shares * (1 + self.com)
        else:  # 'short'
            self.cash += row['Close'] * n_shares * (1 - self.com)

    def check_close_operations(self, row, stop_loss, take_profit, n_shares):
        for op in self.operations:
            if not op.closed and ((op.operation_type == 'long' and (row['Close'] >= take_profit or row['Close'] <= stop_loss)) or
                    (op.operation_type == 'short' and (row['Close'] <= take_profit or row['Close'] >= stop_loss))):
                if op.operation_type == 'long':
                    self.cash += row['Close'] * n_shares * (1 - self.com)
                else:  # 'short'
                    self.cash -= row['Close'] * n_shares * (1 + self.com)

                op.closed = True

    def calculate_operation_value(self, op, current_price, n_shares):
        if op.operation_type == 'long':
            return (current_price - op.bought_at) * n_shares if not op.closed else 0
        else:  # 'short'
            return (op.bought_at - current_price) * n_shares if not op.closed else 0

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
        all_indicators = ['Logistic', 'XGBoost', 'SVM']
        combinations_to_test = [['Logistic'], ['XGBoost'], ['SVM'], all_indicators]

        for combination in combinations_to_test:
            self.active_indicators = combination
            print(f"Using: {self.active_indicators} for ML")
            self.execute_trades()

            final_value = self.strategy_value[-1]
            if final_value > self.best_value:
                self.best_value = final_value
                self.best_combination = self.active_indicators.copy()
            self.reset_strategy()

        print(f"The best one was: {self.best_combination} with a value of: {self.best_value}")

    def reset_strategy(self):
        self.operations.clear()
        self.cash = 1_000_000
        self.strategy_value = [1_000_000]

    def optimize_trade_parameters(self):
        def objective(trial):
            stop_loss_pct = trial.suggest_float('stop_loss_pct', 0.90, 0.99)
            take_profit_pct = trial.suggest_float('take_profit_pct', 1.01, 1.10)
            n_shares = trial.suggest_int('n_shares', 1, 100)

            self.reset_strategy()
            self.execute_trades(best=True, stop_loss=stop_loss_pct, take_profit=take_profit_pct, n_shares=n_shares)
            final_strategy_value = self.strategy_value[-1]

            return final_strategy_value

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        # Mejores parámetros encontrados
        best_params = study.best_params
        print(f"Best params: {best_params}")

        # Aplicar los mejores parámetros a la estrategia
        self.stop_loss_pct = best_params['stop_loss_pct']
        self.take_profit_pct = best_params['take_profit_pct']
        self.n_shares = best_params['n_shares']

    def rend(self, file):
        if file == 'A1':
            data = pd.read_csv('data/aapl_project_1m_test.csv')
        elif file == 'A5':
            data = pd.read_csv('data/aapl_project_test.csv')
        elif file == 'B1':
            data = pd.read_csv('data/btc_project_1m_test.csv')
        elif file == 'B5':
            data = pd.read_csv('data/btc_project_test.csv')

        # Turn date into datetime
        data['Datetime'] = pd.to_datetime(data['Datetime'])

        # Get first and last close data
        primer_cierre = data.iloc[0]['Close']
        ultimo_cierre = data.iloc[-1]['Close']

        # Get asset yeild
        rend_pasivo = (ultimo_cierre - primer_cierre) / primer_cierre
        print("The passive asset return is: {:.2%}".format(rend_pasivo))

        # Compare with used strategy
        cash = 1000000
        cashfinal = 987377.4577192613 # Change this
        rend_estrategia = (cashfinal - cash) / cash
        print("The trading strategy return is: {:.2%}".format(rend_estrategia))

        # Sort data
        data = data.sort_values(by='Datetime')

        # Rend
        data['Returns'] = data['Close'].pct_change().fillna(0)

        # See the value passive
        initial_investment = cash
        data['Investment_Value'] = (1 + data['Returns']).cumprod() * initial_investment

        # Graficar el rendimiento de la inversión
        plt.figure(figsize=(12, 8))
        plt.plot(data['Datetime'], data['Investment_Value'], label='Investment Value', color='blue')
        plt.title('Passive strategy')
        plt.xlabel('Date')
        plt.ylabel('Investment Value')
        plt.legend()
        plt.grid(True)
        plt.show()

        valor_final = data['Investment_Value'].iloc[-1]
        print("The final value of the investment: ${:,.2f}".format(valor_final))

