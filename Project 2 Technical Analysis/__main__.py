import pandas as pd
import matplotlib.pyplot as plt
import ta
import optuna


# Let's try it!
def main():
    file = "A1" # can be change to other file
    data = load_data(file)
    data = calculate_indicators(data)
    indicators = define_buy_sell_signals()
    data = run_signals(data, indicators)

    # Combinaciones de indicadores
    best_combination, best_result = run_combinations(data, 3, indicators, 1_000_000)
    print(f"Mejor combinación de indicadores: {best_combination} con resultado {best_result}")

    # Optimización de parámetros con Optuna
    best_indicators, best_value = optimize_parameters(data, indicators)
    print(f"Mejores indicadores optimizados: {best_indicators} con valor {best_value}")

    # Prueba de estrategia
    strategy_value = test_strategy(data, indicators, best_combination, 1_000_000)
    plot_results(strategy_value, data)


if __name__ == "__main__":
    main()


