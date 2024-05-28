import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sympy import integrate, init_printing as sp
from sympy.abc import x as xsp
import scipy.stats as st
import scipy.optimize as opt
import tqdm
from scipy.integrate import quad
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm


# Plot the Price Distribution
lambda_param = 50
k_param = 10

x = np.linspace(0, 100, 1000)
pdf = (k_param / lambda_param) * (x / lambda_param)**(k_param - 1) * np.exp(-(x / lambda_param)**(k_param))

# Plot the PDF
plt.figure(figsize=(8, 6))
plt.plot(x, pdf, 'r-', lw=2, label='Weibull PDF')
plt.title('Weibull Probability Density Function')
plt.xlabel('Price')
plt.ylabel('Probability Density')
plt.legend(loc = 'best')
plt.show()