# Machine Learning Project

In this project we applied `Machine Learning (ML)` models such as: 
`XGBoost`, `Logistic Regression` and `Support Vecto Machine (SVM)`
we create variables to predict if we will have a buy or sell signal,
the methodology used was a prediction with each of them and a stack of
the three of them. 

For these predictions we used distinct variables, such as: 
* price with a 20 day lag, 
* best indicators from the past project (`Stoch`,`ADX`,`ATR`,`RSI`,`EMA`,`MACD`)
* returns,
* volatility,
* engulfing/doji to note any trend on the candle chart.

We applied these models in the in AAPL (5min, 1min) and BTC (5min, 1min) datasets,
for each one there's a `Train` and a `Test` dataset. `Train` datasets were split in
train and test with an 80/20 ratio. Then hyperparameters were suggested for the 80% train split
to predict values in the 20% test, after that optuna has been used to maximize f1 scores
and best hyperparameters that predict buy and sell signals, the best model found will train the hole
`Train` dataset and predict the `Test` dataset, to make a backtesting afterward.

The parameters each model received were: 

* Logistic Reg
  * C
  * fit_intercept
  * l1_ratio
* SVC
  * C
  * kernel
  * gamma
* XGBoost (xgboost.XGBClassifier)
  * n_estimators
  * max_depth
  * max_leaves
  * learning_rate
  * booster
  * gamma
  * reg_alpha
  * reg_lambda

# Deliverables:
* Follow a Python Project structure.
* Work with a training and validation dataset to optimize & test your trading strategies using the datasets provided in the introduction section.
* The ML models that we'll be using are Logistic Regression, Support Vector Machine, XGBoost and Stack all of them.
* Define the independent and dependent variables to train the models, remember that you can add any technical indicator to your dataset.
* Split the `train` datasets into train/test.
* Our dependent variable should be a category that we want to predict, i.e. "Buy" and "Not buy", or "Sell" and "Not sell" for the short models, we can construct it if the next k price is over / under a certain threshold.
* For each model, fine tune all hyperparameters worth moving, then you can easily generate the True / False signals to backtest.
* Be careful when selecting a metric to fine-tune.
* For each dataset train/test pair of BTC-USD & AAPL (5m, 1m):
* Use the buy/sell signals from the predictions.
* Backtest the strategies while keeping track of the operations and cash/portfolio value time series, remember that we'll be opening long & short positions.
* You can optimize them, or just set them manually, no shame in the game.
* Select the optimal strategy and describe it thoroughly (X, y variables used, a brief description of the ML models, results).
* Present only the results & conclusions in a Jupyter notebook (without unnecessary code, only plot-related things), include the list of the operations, candlestick charts, indicators, trading signals, cash through time, portfolio value through time, and any other chart that you consider important.
* Github Link of the repository!

# Running the code instructions
* Python 3.11 or higher
* Libraries listed in `requirements.txt`
* First call: 
> from `M_L.py` import Operation, TradingStrategy

Specify which dataset we are going to use for example:

AAPL 5m = `A5`

APPL 1m = `A1`

BTC 1m = `B1`

BTC 5m = `B5`

> strategy = TradingStrategy('A5')
 
Then, call.
>strategy.optimize_and_fit_models()
>
> strategy.run_combinations()

so you can see which model performs best and which hyperparam is using.

Then you optimize the `take_profit`, `stop_loss` and `n_shares`
>strategy.optimize_trade_parameters()

After you get the results you will need to change manually the 
`ML_Test.py` you can see commented in the code, were you will need to change
with `#Change`.

Then depending on the dataset you are using you will need to change `train path` and `test path`, the run this code:
>from ML_Test import Operation, TradingStrategy
>
> strategy = TradingStrategy(train_path="data/aapl_project_train.csv", 
> test_path="data/aapl_project_test.csv")
> 
> strategy.train_model()
> 
>strategy.plot_results()


`And that´s a wrap!`




# Authors
* Manuel Reyna aka `Manureymon`
* Eduardo Gutierrez aka `Wachiloco`
* Jose Carlos Lopez aka `Liandroboy`
* Alejandro Frizard aka `Blizzard`
* Alfredo Bautista aka `Bolinha`
