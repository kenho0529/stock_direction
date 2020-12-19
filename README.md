# Stock Price Direction Prediction
This is for predicting the direction of a stock in the next trading day. The direction is defined as difference between the closing price in the next trading day and today trading day. If the closing price in the next trading day is higher than today, it will be assigned as 1. In this project, I have applied Random forest to predict the stock price direction. I have transformed a regression problem to a classification problem, which I believe it can improve the accuracy. However, the drawback of this approach is the uncertainy of magnitude of each direction. Rising 10% and rising 0.1% are the same in this approach but it is so different in the stock market.




## language Used
⋅⋅* Python

## Library Used

MarkUp : - Numpy
         * Pandas
         * matplotlib
         * datetime
         * yfinance
         * sklearn
