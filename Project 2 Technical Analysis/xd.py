import pandas as pd
import optuna
from trading_functions import create_signals, backtest_strategy

# Definir los nombres de los archivos de entrenamiento y prueba
datasets = {
    "A1": "data/aapl_project_1m_train.csv",
    "A5": "data/aapl_project_train.csv",
    "B1": "data/btc_project_1m_train.csv",
    "B5": "data/btc_project_train.csv"
}

# Cargar los datos de entrenamiento
data = {}
for key, path in datasets.items():
    data[key] = pd.read_csv(path)

# Definir los nombres de los archivos de prueba
test_datasets = {
    "A1T": "data/aapl_project_1m_test.csv",
    "A5T": "data/aapl_project_test.csv",
    "B1T": "data/btc_project_1m_test.csv",
    "B5T": "data/btc_project_test.csv"
}


def objective(trial):
    # Definir los rangos de hiperparámetros a probar
    rsi_lower_threshold = trial.suggest_int("rsi_lower_threshold", 10, 30)
    bollinger_window = trial.suggest_int("bollinger_window", 5, 50)
    # Añadir más hiperparámetros según sea necesario para otros indicadores


    # Crear señales usando los hiperparámetros sugeridos
    technical_data = create_signals(data[key],
                                    rsi_window=14,  # Definir el valor base para RSI u otro indicador
                                    rsi_lower_threshold=rsi_lower_threshold,
                                    bollinger_window=bollinger_window,
                                    bollinger_std=2)

    # Ejecutar el backtesting y obtener el rendimiento del portafolio
    portfolio_value = backtest_strategy(technical_data, key, trial)

    # Retornar el valor negativo de rendimiento ya que optuna maximiza
    return -portfolio_value


def optimize_strategy(data, key):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Obtener los mejores hiperparámetros
    best_params = study.best_params

    # Realizar el backtesting con los mejores hiperparámetros
    technical_data = create_signals(data[key], **best_params)
    best_portfolio_value = backtest_strategy(technical_data, key, best_params)

    return best_params, best_portfolio_value

