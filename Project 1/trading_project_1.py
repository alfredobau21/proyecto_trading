from sympy import integrate, init_printing as sp
from sympy.abc import x as xsp
import scipy.stats as st
import scipy.optimize as opt
import tqdm
from scipy.integrate import quad
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

# Grafica de la distribucion Weibull
lambda_param = 50
k_param = 10

x = np.linspace(0, 100, 1000)
pdf = (k_param / lambda_param) * (x / lambda_param)**(k_param - 1) * np.exp(-(x / lambda_param)**(k_param))

# Plot del Weibull PDF
plt.figure(figsize=(8, 6))
plt.plot(x, pdf, 'r-', lw=2, label='Weibull PDF')
plt.title('Weibull Probability Density Function')
plt.xlabel('Price')
plt.ylabel('Probability Density')
plt.legend(loc = 'best')
plt.show()

# Probabilidad de que el trader este informado y el spot price
pi_I = 0.4
S_0 = 51

# Funcion de distribucion
def weibull_pdf(x):
    return (k_param / lambda_param) * (x / lambda_param)**(k_param - 1) * np.exp(-(x / lambda_param)**(k_param))

# Funciones para el beneficio de liquidez, ajustadas para el rango [0, 0.5]
def pi_LB(K_A):
    return max(0, min(0.5, 0.5 - 0.08 * (K_A - S_0)))

def pi_LS(K_B):
    return max(0, min(0.5, 0.5 - 0.08 * (S_0 - K_B)))

def objective_function(K):
    K_A, K_B = K  #
    income = (1 - pi_I) * (pi_LB(K_A) * (K_A - S_0) + pi_LS(K_B) * (S_0 - K_B))

    integral_above_K_A, _ = quad(lambda S: (S - K_A) * weibull_pdf(S), K_A, np.inf)
    integral_below_K_B, _ = quad(lambda S: (K_B - S) * weibull_pdf(S), 0, K_B)
    
    cost = pi_I * (integral_above_K_A + integral_below_K_B)

    
    return -(income - cost)

K_A_values = np.linspace(S_0, S_0 + 10*k_param, 100)
K_B_values = np.linspace(S_0 - 10*k_param, S_0, 100)

initial_guess = [S_0 + 2, S_0 - 2]
bounds = [(S_0, None), (S_0-10*k_param , S_0)]

result = minimize(objective_function, initial_guess, bounds=bounds)
bid = result.x[1]  
ask = result.x[0]
print(f'Resultado de la optimización: revenue esperado de {round(-result.fun, 4)} con un bid de ${round(bid, 2)} y un ask de ${round(ask,2)}')

plt.figure(figsize = (10, 5))
plt.plot(x, pdf, 'k-', lw=2, label='Weibull PDF')
plt.axvline(bid, color='green', linestyle='--', label=f'Optimal Bid: {bid}')
plt.axvline(ask, color='red', linestyle='--', label=f'Optimal Ask: {ask}')
plt.axvline(S_0, color='blue', linestyle='--', label=f'Spot Price: {S_0}')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Density')
plt.legend(loc = 'best')

# Genera los spreads
spreads = np.linspace(0, ask - bid, 1000)

# Calcula el expected revenues para los diferentes escenarios
liq_trades = spreads
inf_trades = spreads * (1 - pi_I)
trades3 = list(map(lambda x: max(0,(0.5 - 0.08*x) * (1 - pi_I) * x), spreads))

# Grafica de los escenarios
plt.figure(figsize=(10, 5))

plt.plot(spreads, liq_trades, '-',  color='blue', label=' Scenario 1: $ \Pi_{LB} = \Pi_{LS} = 0.5 $')
plt.plot(spreads, inf_trades, '-', color='green', label=' Scenario 2 $ \Pi_{I} = 40 \% , \Pi_{LB} = \Pi_{LS} = 0.5 $')
plt.plot(spreads, trades3,'-', color='red', label=' Scenario 3: $ \Pi_{I} = 0.4 , \Pi_{LB}(S) = \Pi_{LS}(S) = 0.5 - 0.08 S $')
plt.legend(loc='best')

plt.title('Expected Revenues')
plt.xlabel('Spread')
plt.ylabel('Revenue')
plt.grid(True)

