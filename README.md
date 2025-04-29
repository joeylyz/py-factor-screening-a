# py-factor-screening-a
A quantitative work to screen listed A share stock, with both technical and fundamental factor, and backtest
data from tushare, require valid license

### step 1: get universe
A share listed company on Shanghai and Shenzhen exchang
apply filters in market cap / trading volume

### step 2: technical factor
here I use MACD, RSI and bollinger band

### step 3: fundamental factor
PE ratio, PB ratio, debt to equity, ROE, revenue growth

### step 4: get trade recommendation, and back test for performance
based on above factor, sort for best 10 stock, perform month end rebalnce, and back test

### next step (underconstruction)
for simplicity, all the factor are weighted as 1 for now
next time will be use machine learning to examine the real impact of each factor, and suggest the best weight,then run back test

