# Information Effects on the Bid-Ask Spread

This project explores the information effects on the bid-ask spread. Building on the contributions of Bagehot ("The Only Game in Town") and the model by Copeland & Galai ("Information Effects on the Bid-Ask Spread"), we demonstrate how a spread can exist even in the absence of transaction costs. By analyzing these models, we aim to maximize the market maker's profit function with respect to the bid and ask prices. This research is crucial as it helps us understand the underlying factors influencing the bid-ask spread and provides insights into the behavior of market makers in financial markets.

## Instructions

1. **Stock Price Follows a Weibull Distribution:**
   - Distribution: $P \sim \text{Weibull}(\lambda = 50, k = 10)$
   - Stock Price: $51$

2. **Probability of an Informed Trade:**
   - $\Pi_I = 0.4$
   - This represents the probability that a trade is informed, set at $40\%$.

3. **Buy Limit Probability ($\Pi_{LB}$) as a Function of $S$:**
   - $\Pi_{LB}(S) = 0.5 - 0.08S$
   - Constraints: $\Pi_{LB} \in [0, 0.5]$
   - Description: Defines $\Pi_{LB}$, the probability of a buy limit, as a linear function of $S$, where $S = (A - S_0)$, $A$ is the proposed stock price, and $S_0$ is the initial stock price.

4. **Sell Limit Probability ($\Pi_{LS}$) as a Function of $S$:**
   - $\Pi_{LS}(S) = 0.5 - 0.08S$
   - Constraints: $\Pi_{LS} \in [0, 0.5]$
   - Description: Defines $\Pi_{LS}$, the probability of a sell limit, as a linear function of $S$, where $S = (S_0 - B)$, $S_0$ is the initial stock price, and $B$ is the bid price.
