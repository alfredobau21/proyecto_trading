# Technical Analysis Trading Project

This project consists in simulation and modeling of trading
strategies with technical analysis for 2 tickers with 2 time
frames for each: 
* **APPLE** train for 1 and 5 mins
*  **BITCOIN** train for 1 and 5 mins
* and their respective test data set for both.

### Tecnhical Indicators
> The techincal indicators used for this project where:
> * RSI
> * SMA
> * MACD
> * Stochastic Oscilator
> * SAR
> 
> for a total of 31 possible strategies.

With this TIs the team created **BUY** and **SELL** signals.

Then the code checked which indicators were used in the best combination,
 also it reviewed if there was enough capital to create a position, if not the position 
was not executed until there was sufficient capital.

The goal for this project is to obtain the best combination of technical indicators
and get the best parameters and put them into test.

To obtain the best result the team made a for loop, to get al the possible combinations
then the code pass through an optimization process, where the best params and the best combination
are obtained. 

With this result, we try the params in the test dataset and compare the results in a plot
with a **Benchmark** which is a portfolio with the maximum amount of AAPL/BTC stocks bought with
the starting capital. This is made to have a visual of which strategy performed better.


# Instructions for __ main __.py
In 
>__ main __.py

The user will choose which dataset
will be executed. 

* A1 = aaple 1 min window
* A5 = apple 5 min window
* B1 = Bitcoin 1 min window
* B5 = Bitcoin 5 min window

in 

__strategy = TradingStrategy('...')__

the user should choose either (A1, A5, B1 or B5)
and write it inside the ('...').

Once the strategy has been chosen the user should hit 
> Run |>




