import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
from datetime import datetime, timedelta
import os
import time

def get_stock_universe():
    """
    Get stock universe from saved CSV and apply filters
    """
    print("Loading stock universe...")
    
    try:
        # Load stocks from CSV
        stocks = pd.read_csv('china_stocks.csv')
        
        # Filter out Beijing stocks
        stocks = stocks[~stocks['ts_code'].str.endswith('.BJ')]
        print(f"After removing Beijing stocks: {len(stocks)} stocks")
        
        # Get current date
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Get daily data for volume
        daily_data = pro.daily(trade_date=current_date, 
                             fields='ts_code,vol')
        
        # Get daily basic data for market cap
        daily_basic_data = pro.daily_basic(trade_date=current_date,
                                         fields='ts_code,total_mv')
        
        if daily_data is not None and not daily_data.empty:
            # Merge with daily data for volume
            stocks = stocks.merge(daily_data, on='ts_code', how='left')
            
            # Filter by volume (lower requirement)
            min_volume = 1000000  # 1 million shares
            stocks = stocks[stocks['vol'] >= min_volume]
            print(f"After volume filtering: {len(stocks)} stocks")
            
            # Merge with daily basic data for market cap
            if daily_basic_data is not None and not daily_basic_data.empty:
                stocks = stocks.merge(daily_basic_data, on='ts_code', how='left')
                # Sort by market cap
                stocks = stocks.sort_values('total_mv', ascending=False)
            
        return stocks['ts_code'].tolist()
        
    except Exception as e:
        print(f"Error in get_stock_universe: {str(e)}")
        return []

def get_technical_data(ts_code, start_date, end_date):
    """Get and calculate technical indicators"""
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if df is None or df.empty:
            return None
            
        # Calculate technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        upper, middle, lower = talib.BBANDS(
            df['close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        return df
        
    except Exception as e:
        print(f"Error getting technical data for {ts_code}: {str(e)}")
        return None

def get_fundamental_data(ts_code, period):
    """Get fundamental indicators"""
    try:
        # Get daily basic data for price and market cap
        daily_basic = pro.daily_basic(ts_code=ts_code, trade_date=period)
        if daily_basic is None or daily_basic.empty:
            return None
            
        # Get income statement for earnings
        income = pro.income(ts_code=ts_code, period=period)
        if income is None or income.empty:
            income = pro.income(ts_code=ts_code, period='20231231')
            if income is None or income.empty:
                return None
            
        # Get balance sheet for book value
        balance = pro.balancesheet(ts_code=ts_code, period=period)
        if balance is None or balance.empty:
            balance = pro.balancesheet(ts_code=ts_code, period='20231231')
            if balance is None or balance.empty:
                return None
            
        # Calculate fundamental ratios
        try:
            price = daily_basic['close'].iloc[0]
            total_shares = daily_basic['total_share'].iloc[0] * 1000  # Convert from thousands to actual
            
            if total_shares <= 0:
                return None
            
            # Calculate per-share values
            net_income = income['n_income'].iloc[0]
            total_equity = balance['total_hldr_eqy_exc_min_int'].iloc[0]
            eps = net_income / total_shares
            bvps = total_equity / total_shares
            
            # Get revenue growth
            prev_income = pro.income(ts_code=ts_code, period='20230630')
            if prev_income is not None and not prev_income.empty:
                current_revenue = income['total_revenue'].iloc[0]
                prev_revenue = prev_income['total_revenue'].iloc[0]
                revenue_growth = (current_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else 0
            else:
                revenue_growth = 0
            
            return {
                'pe_ratio': price / eps if eps != 0 else float('inf'),
                'pb_ratio': price / bvps if bvps != 0 else float('inf'),
                'debt_to_equity': balance['total_liab'].iloc[0] / total_equity if total_equity != 0 else float('inf'),
                'roe': net_income / total_equity if total_equity != 0 else 0,
                'revenue_growth': revenue_growth
            }
            
        except Exception as e:
            return None
            
    except Exception as e:
        return None

def calculate_scores(technical_data, fundamental_data):
    """Calculate combined score based on technical and fundamental indicators"""
    if technical_data is None or fundamental_data is None:
        return None
        
    # Technical score components
    tech_score = 0
    if technical_data['rsi'].iloc[-1] < 30:
        tech_score += 1
    if technical_data['macd_hist'].iloc[-1] > 0:
        tech_score += 1
    if technical_data['close'].iloc[-1] < technical_data['bb_lower'].iloc[-1]:
        tech_score += 1
        
    # Fundamental score components
    fund_score = 0
    if fundamental_data['pe_ratio'] < 50:
        fund_score += 1
    if fundamental_data['pb_ratio'] < 5:
        fund_score += 1
    if fundamental_data['debt_to_equity'] < 2:
        fund_score += 1
    if fundamental_data['roe'] > 0.1:
        fund_score += 1
    if fundamental_data['revenue_growth'] > 0.1:
        fund_score += 1
        
    # Combined score (equal weight)
    total_score = (tech_score + fund_score) / 2
    return total_score

def run_strategy(start_date, end_date):
    """Run the strategy for the given date range"""
    print("\nRunning strategy...")
    # Get stock universe
    stock_list = get_stock_universe()
    
    if not stock_list:
        print("No stocks available for the strategy. Exiting.")
        return []
    
    # Generate rebalancing dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    
    portfolio_history = []
    
    for date in rebalance_dates:
        print(f"\nRebalancing on {date.strftime('%Y-%m-%d')}")
        
        # Get scores for all stocks
        stock_scores = {}
        processed_stocks = 0
        valid_stocks = 0
        
        for ts_code in stock_list:
            # Get technical data for the past 3 months
            tech_start = (date - timedelta(days=90)).strftime('%Y%m%d')
            tech_end = date.strftime('%Y%m%d')
            
            technical_data = get_technical_data(ts_code, tech_start, tech_end)
            if technical_data is None:
                continue
                
            # Use the most recent available data for fundamental analysis
            current_date = date.strftime('%Y%m%d')
            fundamental_data = get_fundamental_data(ts_code, current_date)
            
            # If no data for current date, try previous month
            if fundamental_data is None:
                prev_month = (date - timedelta(days=30)).strftime('%Y%m%d')
                fundamental_data = get_fundamental_data(ts_code, prev_month)
            
            # If still no data, try December 2023
            if fundamental_data is None:
                fundamental_data = get_fundamental_data(ts_code, '20231231')
            
            if fundamental_data is None:
                continue
            
            score = calculate_scores(technical_data, fundamental_data)
            if score is not None:
                stock_scores[ts_code] = score
                valid_stocks += 1
            
            processed_stocks += 1
            if processed_stocks % 60 == 0:
                print(f"\nProcessed {processed_stocks} stocks...")
                print(f"Valid stocks so far: {valid_stocks}")
                time.sleep(1)
        
        if not stock_scores:
            print(f"No stocks with valid scores on {date.strftime('%Y-%m-%d')}")
            continue
            
        # Sort stocks by score and select top N
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = sorted_stocks[:10]  # Top 10 stocks
        
        # Record portfolio composition
        portfolio_history.append({
            'date': date,
            'portfolio': {stock: score for stock, score in selected_stocks}
        })
        
        print(f"\nSelected stocks: {list(portfolio_history[-1]['portfolio'].keys())}")
        print(f"Number of valid stocks: {valid_stocks}")
        
    return portfolio_history

def calculate_returns(portfolio_history):
    """Calculate strategy returns"""
    print("\nCalculating returns...")
    returns = []
    dates = []
    
    for i in range(1, len(portfolio_history)):
        prev_date = portfolio_history[i-1]['date']
        curr_date = portfolio_history[i]['date']
        dates.append(curr_date)
        
        # Calculate returns for each stock in the portfolio
        period_returns = []
        for ts_code in portfolio_history[i-1]['portfolio'].keys():
            # Get price data for the period
            start_date = prev_date.strftime('%Y%m%d')
            end_date = curr_date.strftime('%Y%m%d')
            price_data = get_technical_data(ts_code, start_date, end_date)
            
            if price_data is not None and not price_data.empty:
                period_return = (price_data['close'].iloc[-1] / price_data['close'].iloc[0]) - 1
                period_returns.append(period_return)
        
        # Calculate equal-weighted portfolio return
        if period_returns:
            portfolio_return = np.mean(period_returns)
            returns.append(portfolio_return)
    
    return dates, returns

def get_csi300_returns(start_date, end_date):
    """Get CSI300 index returns for comparison"""
    try:
        # Get CSI300 daily data
        csi300 = pro.index_daily(ts_code='000300.SH', 
                                start_date=start_date.strftime('%Y%m%d'),
                                end_date=end_date.strftime('%Y%m%d'))
        
        if csi300 is None or csi300.empty:
            return None, None
            
        # Convert to monthly returns
        csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
        csi300.set_index('trade_date', inplace=True)
        monthly_returns = csi300['close'].resample('M').last().pct_change()
        
        return monthly_returns.index, monthly_returns.values
        
    except Exception as e:
        print(f"Error getting CSI300 returns: {str(e)}")
        return None, None

def plot_results(dates, returns):
    """Plot strategy performance and compare with CSI300"""
    print("\nPlotting results...")
    
    # Get CSI300 returns
    csi300_dates, csi300_returns = get_csi300_returns(dates[0], dates[-1])
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + np.array(returns)).cumprod()
    if csi300_returns is not None:
        csi300_cumulative = (1 + np.array(csi300_returns)).cumprod()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(dates, strategy_cumulative, label='Strategy', linewidth=2)
    if csi300_returns is not None:
        plt.plot(csi300_dates, csi300_cumulative, label='CSI300', linewidth=2, linestyle='--')
    plt.title('Strategy vs CSI300 Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('strategy_vs_csi300.png')
    plt.close()
    
    # Plot monthly returns
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(dates))
    plt.bar(x - width/2, returns, width, label='Strategy')
    if csi300_returns is not None:
        plt.bar(x + width/2, csi300_returns, width, label='CSI300')
    plt.title('Strategy vs CSI300 Monthly Returns')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.xticks(x, [d.strftime('%Y-%m') for d in dates], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('monthly_returns_comparison.png')
    plt.close()
    
    # Print performance comparison
    if csi300_returns is not None:
        print("\nPerformance Comparison:")
        print(f"Strategy Total Return: {((1 + np.array(returns)).prod() - 1) * 100:.2f}%")
        print(f"CSI300 Total Return: {((1 + np.array(csi300_returns)).prod() - 1) * 100:.2f}%")
        print(f"Strategy Annualized Return: {((1 + np.array(returns)).prod() ** (12/len(returns)) - 1) * 100:.2f}%")
        print(f"CSI300 Annualized Return: {((1 + np.array(csi300_returns)).prod() ** (12/len(csi300_returns)) - 1) * 100:.2f}%")
        print(f"Strategy Sharpe Ratio: {np.mean(returns) / np.std(returns) * np.sqrt(12):.2f}")
        print(f"CSI300 Sharpe Ratio: {np.mean(csi300_returns) / np.std(csi300_returns) * np.sqrt(12):.2f}")

def save_trading_history(portfolio_history):
    """Save trading history to CSV"""
    print("\nSaving trading history...")
    trading_history = []
    for record in portfolio_history:
        date = record['date']
        for stock, score in record['portfolio'].items():
            trading_history.append({
                'Date': date,
                'Stock': stock,
                'Score': score
            })
    
    df = pd.DataFrame(trading_history)
    df.to_csv('trading_history.csv', index=False)
    print("Trading history saved to trading_history.csv")

def get_stock_name(ts_code):
    """Get stock name from Tushare"""
    try:
        stock_info = pro.stock_basic(ts_code=ts_code)
        if stock_info is not None and not stock_info.empty:
            return stock_info['name'].iloc[0]
        return ts_code
    except:
        return ts_code

def print_trading_table(portfolio_history):
    """Print a table of selected stocks for each month"""
    print("\nMonthly Trading Table:")
    print("-" * 100)
    print(f"{'Date':<12} {'Stock Code':<12} {'Stock Name':<30} {'Score':<10}")
    print("-" * 100)
    
    for record in portfolio_history:
        date = record['date'].strftime('%Y-%m')
        for stock, score in record['portfolio'].items():
            stock_name = get_stock_name(stock)
            print(f"{date:<12} {stock:<12} {stock_name:<30} {score:.2f}")
        print("-" * 100)

def main():
    """Main function to run the strategy"""
    # Initialize Tushare
    ts.set_token('30dd89a41c53facb136d77f8ba652b5963fe96919483ac2a4a111a4f')
    global pro
    pro = ts.pro_api()
    
    # Set date range for backtest (from January 2023 to December 2024)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print(f"Running backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Run strategy
    portfolio_history = run_strategy(start_date, end_date)
    
    if not portfolio_history:
        print("No portfolio history generated. Exiting.")
        return
    
    # Calculate returns
    dates, returns = calculate_returns(portfolio_history)
    
    # Print results
    print("\nStrategy Performance:")
    print(f"Total Return: {((1 + np.array(returns)).prod() - 1) * 100:.2f}%")
    print(f"Annualized Return: {((1 + np.array(returns)).prod() ** (12/len(returns)) - 1) * 100:.2f}%")
    print(f"Average Monthly Return: {np.mean(returns) * 100:.2f}%")
    print(f"Monthly Return Std Dev: {np.std(returns) * 100:.2f}%")
    
    # Plot results and compare with CSI300
    plot_results(dates, returns)
    
    # Print trading table
    print_trading_table(portfolio_history)
    
    # Save trading history
    save_trading_history(portfolio_history)

if __name__ == "__main__":
    main() 