import pandas as pd
from matplotlib import pyplot as plt

from TA import Operation, TradingStrategy
strategy = TradingStrategy('B5')
strategy.run_combinations()

strategy.plot_results(best = True)

strategy.test()

a= strategy.strategy_value[-1]
print(a) # best value of Test

strategy.optimize_parameters()

strategy.plot_results(best = True)

b = strategy.strategy_value[-1]
print(b) # best value of backtest w optimized

strategy.test()

x = strategy.strategy_value[-1]
print(x) # optimized params for test

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
cash=1000000
cashfinal = x
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