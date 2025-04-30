# py-factor-screening-a
A quantitative work to screen listed A share stock, with both technical and fundamental factor, and backtest
data from tushare, require valid license

<get_china_stock.py>
### step 1: get universe
A share listed company on Shanghai and Shenzhen exchang
apply filters in market cap / trading volume

<china_stock_strategy.py>
### step 2: technical factor
here I use MACD, RSI and bollinger band

### step 3: fundamental factor
PE ratio, PB ratio, debt to equity, ROE, revenue growth

### step 4: get trade recommendation, and back test for performance
based on above factor, sort for best 10 stock, perform month end rebalnce, and back test

<ML_strategy.py>
### step 5: machine learning for trade recommendation
use histoical indicators and share price to trade model(random forest), use trained model to make trade recommendation, and calculate performance

<portfolio_analysis.py>
### step 6: portfolio analysis
return and risk mertrics, plots
