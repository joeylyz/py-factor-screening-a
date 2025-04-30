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
    try:
        # Load the filtered stock universe
        stocks = pd.read_csv('china_stocks.csv')
        return stocks['ts_code'].tolist()
    except Exception as e:
        print(f"Error in get_stock_universe: {str(e)}")
        return []

def get_technical_data(ts_code, start_date, end_date):
    """Get and calculate technical indicators from the single all_price_data.csv file, or from a local technical file if available."""
    all_file = 'stock_data/all_price_data.csv'
    local_file = f'stock_data/{ts_code}_technical_{start_date}_{end_date}.csv'
    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
        if not df.empty:
            return df
    if not os.path.exists(all_file):
        return None
    df = pd.read_csv(all_file)
    df = df[(df['ts_code'] == ts_code) & (df['trade_date'] >= int(start_date)) & (df['trade_date'] <= int(end_date))]
    if df.empty:
        return None
    df = df.sort_values('trade_date')
    if 'rsi' not in df.columns:
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
        # Save technical data to local file for future use
        df.to_csv(local_file, index=False)
    return df

def get_fundamental_data(ts_code, period):
    """Get fundamental indicators from a local file if available."""
    local_file = f'stock_data/{ts_code}_fundamental_{period}.csv'
    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
        if not df.empty:
            return df.to_dict('records')[0]
    # If not available, fallback to original fetching logic
    try:
        daily_basic = pro.daily_basic(ts_code=ts_code, trade_date=period)
        if daily_basic is None or daily_basic.empty:
            return None
        income = pro.income(ts_code=ts_code, period=period)
        if income is None or income.empty:
            income = pro.income(ts_code=ts_code, period='20231231')
            if income is None or income.empty:
                return None
        balance = pro.balancesheet(ts_code=ts_code, period=period)
        if balance is None or balance.empty:
            balance = pro.balancesheet(ts_code=ts_code, period='20231231')
            if balance is None or balance.empty:
                return None
        price = daily_basic['close'].iloc[0]
        total_shares = daily_basic['total_share'].iloc[0] * 1000
        if total_shares <= 0:
            return None
        net_income = income['n_income'].iloc[0]
        total_equity = balance['total_hldr_eqy_exc_min_int'].iloc[0]
        eps = net_income / total_shares
        bvps = total_equity / total_shares
        prev_income = pro.income(ts_code=ts_code, period='20230630')
        if prev_income is not None and not prev_income.empty:
            current_revenue = income['total_revenue'].iloc[0]
            prev_revenue = prev_income['total_revenue'].iloc[0]
            revenue_growth = (current_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else 0
        else:
            revenue_growth = 0
        ratios = {
            'pe_ratio': price / eps if eps != 0 else float('inf'),
            'pb_ratio': price / bvps if bvps != 0 else float('inf'),
            'debt_to_equity': balance['total_liab'].iloc[0] / total_equity if total_equity != 0 else float('inf'),
            'roe': net_income / total_equity if total_equity != 0 else 0,
            'revenue_growth': revenue_growth
        }
        os.makedirs('stock_data', exist_ok=True)
        pd.DataFrame([ratios]).to_csv(local_file, index=False)
        return ratios
    except Exception as e:
        return None

def calculate_scores(technical_data, fundamental_data):
    """Calculate combined score based on technical and fundamental indicators"""
    if technical_data is None or fundamental_data is None:
        return None
    # Technical score components
    tech_score = 0
    if technical_data['rsi'].iloc[-1] < 30:
        tech_score += 0.5
    if technical_data['macd_hist'].iloc[-1] > 0:
        tech_score += 1
    if technical_data['close'].iloc[-1] < technical_data['bb_lower'].iloc[-1]:
        tech_score += 1
    # Fundamental score components
    fund_score = 0
    if 10 < fundamental_data['pe_ratio'] < 30:
        fund_score += 1.5
    if 0.3 < fundamental_data['pb_ratio'] < 1:
        fund_score += 1.5
    if fundamental_data['debt_to_equity'] < 2:
        fund_score += 1
    if fundamental_data['roe'] > 0.1:
        fund_score += 1
    if fundamental_data['revenue_growth'] > 0.1:
        fund_score += 1
    # Combined score (weighted)
    total_score = tech_score * 0.66 + fund_score * 0.34
    return total_score

def save_stock_data(stock_data, date):
    """Save technical and fundamental data for all stocks"""
    try:
        # Create directory if it doesn't exist
        os.makedirs('stock_data', exist_ok=True)
        
        # Save data to CSV
        filename = f"stock_data/stock_metrics_{date.strftime('%Y%m')}.csv"
        stock_data.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error saving stock data: {str(e)}")

def run_strategy(start_date, end_date):
    """Run the strategy for the given date range"""
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
        
        # Try to load precomputed metrics for this month
        metrics_file = f"stock_data/stock_metrics_{date.strftime('%Y%m')}.csv"
        stock_scores = {}
        stock_metrics = []
        valid_stocks = 0
        if os.path.exists(metrics_file):
            print(f"Loading all stock metrics from {metrics_file}")
            metrics_df = pd.read_csv(metrics_file)
            # Recalculate score for each row using the latest logic
            for idx, row in metrics_df.iterrows():
                technical_data = {
                    'rsi': pd.Series([row['rsi']]),
                    'macd_hist': pd.Series([row['macd_hist']]),
                    'close': pd.Series([row['bb_position']]),  # Use close if available
                    'bb_lower': pd.Series([row['bb_position'] - 1e-6]),  # Dummy, adjust as needed
                }
                fundamental_data = {
                    'pe_ratio': row['pe_ratio'],
                    'pb_ratio': row['pb_ratio'],
                    'debt_to_equity': row['debt_to_equity'],
                    'roe': row['roe'],
                    'revenue_growth': row['revenue_growth'],
                }
                score = calculate_scores(technical_data, fundamental_data)
                metrics_df.at[idx, 'score'] = score
                if score is not None:
                    stock_scores[row['ts_code']] = score
                    valid_stocks += 1
                    stock_metrics.append({
                        'ts_code': row['ts_code'],
                        'date': date.strftime('%Y-%m-%d'),
                        'rsi': row['rsi'],
                        'macd_hist': row['macd_hist'],
                        'bb_position': row['bb_position'],
                        'pe_ratio': row['pe_ratio'],
                        'pb_ratio': row['pb_ratio'],
                        'debt_to_equity': row['debt_to_equity'],
                        'roe': row['roe'],
                        'revenue_growth': row['revenue_growth'],
                        'score': score
                    })
            # Save metrics for this month with updated scores
            metrics_df.to_csv(metrics_file, index=False)
        else:
            processed_stocks = 0
            batch_size = 50
            for i in range(0, len(stock_list), batch_size):
                batch = stock_list[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(stock_list)-1)//batch_size + 1}")
                for ts_code in batch:
                    try:
                        # Get technical data for the past 3 months
                        tech_start = (date - timedelta(days=90)).strftime('%Y%m%d')
                        tech_end = date.strftime('%Y%m%d')
                        technical_data = get_technical_data(ts_code, tech_start, tech_end)
                        if technical_data is None:
                            continue
                        # Use the most recent available data for fundamental analysis
                        current_date = date.strftime('%Y%m%d')
                        fundamental_data = get_fundamental_data(ts_code, current_date)
                        if fundamental_data is None:
                            prev_month = (date - timedelta(days=30)).strftime('%Y%m%d')
                            fundamental_data = get_fundamental_data(ts_code, prev_month)
                        if fundamental_data is None:
                            fundamental_data = get_fundamental_data(ts_code, '20231231')
                        if fundamental_data is None:
                            continue
                        score = calculate_scores(technical_data, fundamental_data)
                        if score is not None:
                            stock_scores[ts_code] = score
                            valid_stocks += 1
                            stock_metrics.append({
                                'ts_code': ts_code,
                                'date': date.strftime('%Y-%m-%d'),
                                'rsi': technical_data['rsi'].iloc[-1],
                                'macd_hist': technical_data['macd_hist'].iloc[-1],
                                'bb_position': (technical_data['close'].iloc[-1] - technical_data['bb_lower'].iloc[-1]) / \
                                             (technical_data['bb_upper'].iloc[-1] - technical_data['bb_lower'].iloc[-1]),
                                'pe_ratio': fundamental_data['pe_ratio'],
                                'pb_ratio': fundamental_data['pb_ratio'],
                                'debt_to_equity': fundamental_data['debt_to_equity'],
                                'roe': fundamental_data['roe'],
                                'revenue_growth': fundamental_data['revenue_growth'],
                                'score': score
                            })
                        processed_stocks += 1
                    except Exception as e:
                        print(f"Error processing {ts_code}: {str(e)}")
                        continue
                # Save metrics for this month
                if stock_metrics:
                    metrics_df = pd.DataFrame(stock_metrics)
                    metrics_df.to_csv(metrics_file, index=False)
        if not stock_scores:
            continue
        # Sort stocks by score and select top N
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        selected_stocks = sorted_stocks[:10]  # Top 10 stocks
        # Record portfolio composition
        portfolio_history.append({
            'date': date,
            'portfolio': {stock: score for stock, score in selected_stocks}
        })
    return portfolio_history

def calculate_returns(portfolio_history):
    """Calculate strategy returns"""
    returns = []
    dates = []
    portfolio_values = [1.0]  # Start with portfolio value of 1
    
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
            # Update portfolio value
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
    
    return dates, returns, portfolio_values

def get_csi300_returns(start_date, end_date):
    """Get CSI300 index returns for comparison"""
    try:
        # First check if data exists locally
        local_file = f'stock_data/csi300_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        if os.path.exists(local_file):
            try:
                csi300 = pd.read_csv(local_file)
                csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
                csi300.set_index('trade_date', inplace=True)
                return csi300.index, csi300['pct_chg'].values / 100  # Convert percentage to decimal
            except:
                pass  # If local file is corrupted, continue to download

        # Get CSI300 daily data
        csi300 = pro.index_daily(ts_code='000300.SH', 
                                start_date=start_date.strftime('%Y%m%d'),
                                end_date=end_date.strftime('%Y%m%d'))
        
        if csi300 is None or csi300.empty:
            return None, None
            
        # Convert to monthly returns
        csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
        csi300.set_index('trade_date', inplace=True)
        csi300 = csi300.resample('M').last()
        monthly_returns = csi300['pct_chg'].values / 100  # Convert percentage to decimal
        
        # Save to local file
        os.makedirs('stock_data', exist_ok=True)
        csi300.to_csv(local_file)
        
        return csi300.index, monthly_returns
        
    except Exception as e:
        return None, None

def plot_results(dates, returns, portfolio_values):
    """Plot strategy performance and compare with CSI300"""
    # Get CSI300 returns
    csi300_dates, csi300_returns = get_csi300_returns(dates[0], dates[-1])
    
    if csi300_returns is None:
        # Plot only strategy returns
        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values[1:], label='Strategy', linewidth=2)
        plt.title('Strategy Portfolio Values')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Starting from 1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_vs_csi300.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Calculate CSI300 portfolio values
        csi300_values = [1.0]
        for ret in csi300_returns:
            csi300_values.append(csi300_values[-1] * (1 + ret))
        
        # Plot portfolio values
        plt.figure(figsize=(12, 6))
        plt.plot(dates, portfolio_values[1:], label='Strategy', linewidth=2)
        plt.plot(csi300_dates, csi300_values[1:], label='CSI300', linewidth=2, linestyle='--')
        plt.title('Strategy vs CSI300 Portfolio Values')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Starting from 1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('strategy_vs_csi300.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot monthly returns
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(len(dates))
        plt.bar(x - width/2, returns, width, label='Strategy')
        plt.bar(x + width/2, csi300_returns, width, label='CSI300')
        plt.title('Strategy vs CSI300 Monthly Returns')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.xticks(x, [d.strftime('%Y-%m') for d in dates], rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('monthly_returns_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print performance comparison
        print("\nPerformance Comparison:")
        print(f"Strategy Total Return: {(portfolio_values[-1] - 1) * 100:.2f}%")
        print(f"CSI300 Total Return: {(csi300_values[-1] - 1) * 100:.2f}%")
        print(f"Strategy Annualized Return: {(portfolio_values[-1] ** (12/len(returns)) - 1) * 100:.2f}%")
        print(f"CSI300 Annualized Return: {(csi300_values[-1] ** (12/len(csi300_returns)) - 1) * 100:.2f}%")
        print(f"Strategy Sharpe Ratio: {np.mean(returns) / np.std(returns) * np.sqrt(12):.2f}")
        print(f"CSI300 Sharpe Ratio: {np.mean(csi300_returns) / np.std(csi300_returns) * np.sqrt(12):.2f}")

def save_trading_history(portfolio_history):
    """Save trading history to CSV"""
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

def fetch_and_store_all_price_data(stock_list, start_date, end_date):
    """Fetch and store all historical daily price data for all stocks in a single CSV file."""
    all_data = []
    all_file = 'stock_data/all_price_data.csv'
    if os.path.exists(all_file):
        all_df = pd.read_csv(all_file)
        existing = set(zip(all_df['ts_code'], all_df['trade_date']))
    else:
        existing = set()
    for ts_code in stock_list:
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date.strftime('%Y%m%d'), end_date=end_date.strftime('%Y%m%d'))
            if df is not None and not df.empty:
                df = df[~df.apply(lambda row: (row['ts_code'], row['trade_date']) in existing, axis=1)]
                if not df.empty:
                    all_data.append(df)
        except Exception as e:
            print(f"Error fetching price data for {ts_code}: {str(e)}")
    if all_data:
        os.makedirs('stock_data', exist_ok=True)
        all_df = pd.concat([pd.read_csv(all_file)] + all_data) if os.path.exists(all_file) else pd.concat(all_data)
        all_df.to_csv(all_file, index=False)

def main():
    """Main function to run the strategy"""
    # Initialize Tushare
    ts.set_token('30dd89a41c53facb136d77f8ba652b5963fe96919483ac2a4a111a4f')
    global pro
    pro = ts.pro_api()
    
    # Set date range for backtest (from January 2023 to December 2024)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Get stock universe
    stock_list = get_stock_universe()
    if not stock_list:
        print("No stocks available for the strategy. Exiting.")
        return
    
    # Prefetch all price data for the universe and time range, only if not already present
    if not os.path.exists('stock_data/all_price_data.csv'):
        fetch_and_store_all_price_data(stock_list, start_date, end_date)
    
    # Run strategy
    portfolio_history = run_strategy(start_date, end_date)
    
    if not portfolio_history:
        print("No portfolio history generated. Exiting.")
        return
    
    # Calculate returns
    dates, returns, portfolio_values = calculate_returns(portfolio_history)
    
    # Save actual monthly returns for analysis
    pd.DataFrame({'Date': dates, 'Return': returns}).to_csv('portfolio_monthly_returns.csv', index=False)
    
    # Print results
    print("\nStrategy Performance:")
    print(f"Total Return: {(portfolio_values[-1] - 1) * 100:.2f}%")
    print(f"Annualized Return: {(portfolio_values[-1] ** (12/len(returns)) - 1) * 100:.2f}%")
    print(f"Average Monthly Return: {np.mean(returns) * 100:.2f}%")
    print(f"Monthly Return Std Dev: {np.std(returns) * 100:.2f}%")
    
    # Plot results and compare with CSI300
    plot_results(dates, returns, portfolio_values)
    
    # Print trading table
    print_trading_table(portfolio_history)
    
    # Save trading history
    save_trading_history(portfolio_history)

if __name__ == "__main__":
    main() 
