# Pairs-Trading
This project explores pairs trading, i.e. exploits the spread between two assets which mean revert.

The bulk of the code consists of the PairsTrading() class, which allows us to analyse a pair of assets' potential for pairs trading.

Functionalities:
- Test for cointegration of assets (via EG/ADF test)
- Compute hedge ratio for spread (using simple moving average)
- Estimate half-life by fitting Ornstein-Uhlenbeck Process
- Evalute trading strategy based on bollinger bands entry criteria
