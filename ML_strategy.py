import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime, timedelta
import tushare as ts

def load_stock_data():
    """Load all saved stock metrics data"""
    print("Loading stock data files...")
    all_data = []
    stock_data_files = [f for f in os.listdir('stock_data') 
                       if f.startswith('stock_metrics_') and f.endswith('.csv')]
    
    if not stock_data_files:
        print("Error: No stock metrics files found in stock_data directory")
        return None
    
    print(f"Found {len(stock_data_files)} stock metrics files")
    
    for i, filename in enumerate(stock_data_files, 1):
        try:
            file_path = os.path.join('stock_data', filename)
            print(f"Loading file {i}/{len(stock_data_files)}: {filename}")
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {filename}: {str(e)}")
            continue
    
    if not all_data:
        print("Error: No data was successfully loaded")
        return None
    
    print("Concatenating all data...")
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Total records loaded: {len(combined_data)}")
    return combined_data

def fetch_and_save_historical_prices(stock_list, start_date, end_date):
    """Fetch and save historical prices for all stocks"""
    print(f"\nFetching historical prices for {len(stock_list)} stocks...")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    all_prices = {}
    
    for i, ts_code in enumerate(stock_list, 1):
        if i % 10 == 0:  # Print progress every 10 stocks
            print(f"Processing stock {i}/{len(stock_list)}")
        
        try:
            df = pro.daily(ts_code=ts_code, 
                          start_date=start_date.strftime('%Y%m%d'),
                          end_date=end_date.strftime('%Y%m%d'))
            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                all_prices[ts_code] = df['close']
                print(f"  {ts_code}: Found {len(df)} price points")
            else:
                print(f"  {ts_code}: No data available")
        except Exception as e:
            print(f"Error fetching data for {ts_code}: {str(e)}")
            continue
    
    # Save prices to CSV
    print("\nSaving historical prices...")
    prices_df = pd.DataFrame(all_prices)
    prices_df.to_csv('historical_prices.csv')
    print(f"Saved prices for {len(prices_df.columns)} stocks")
    
    return prices_df

def load_historical_prices():
    """Load historical prices from saved file"""
    try:
        print("Loading historical prices...")
        prices_df = pd.read_csv('historical_prices.csv', index_col=0, parse_dates=True)
        print(f"Loaded prices for {len(prices_df.columns)} stocks")
        return prices_df
    except FileNotFoundError:
        print("Historical prices file not found")
        return None

def calculate_monthly_returns_from_saved(ts_code, prices_df):
    """Calculate monthly returns using saved price data"""
    if ts_code not in prices_df.columns:
        return None
    monthly_returns = prices_df[ts_code].resample('ME').last().pct_change(fill_method=None)
    return monthly_returns

def prepare_training_data():
    """Prepare data for machine learning model"""
    print("\nLoading and preparing data...")
    # Load all stock metrics
    stock_data = load_stock_data()
    if stock_data is None:
        return None, None, None
    
    # Get unique stocks and date range
    unique_stocks = stock_data['ts_code'].unique()
    start_date = stock_data['date'].min()
    end_date = stock_data['date'].max()
    
    # Load or fetch historical prices
    prices_df = load_historical_prices()
    if prices_df is None:
        prices_df = fetch_and_save_historical_prices(unique_stocks, 
                                                   datetime.strptime(start_date, '%Y-%m-%d'),
                                                   datetime.strptime(end_date, '%Y-%m-%d'))
    
    print("\nCalculating returns for each stock...")
    stock_data['next_month_return'] = np.nan
    print(f"Total unique stocks: {len(unique_stocks)}")
    
    for i, stock in enumerate(unique_stocks, 1):
        if i % 10 == 0:  # Print progress every 10 stocks
            print(f"Processing stock {i}/{len(unique_stocks)}")
        
        stock_df = stock_data[stock_data['ts_code'] == stock]
        for idx, row in stock_df.iterrows():
            current_date = datetime.strptime(row['date'], '%Y-%m-%d')
            returns = calculate_monthly_returns_from_saved(stock, prices_df)
            if returns is not None and not returns.empty:
                # Find the next month's return
                next_month = current_date + timedelta(days=30)
                if next_month in returns.index:
                    stock_data.loc[idx, 'next_month_return'] = returns[next_month]
    
    # Drop rows with missing returns
    initial_count = len(stock_data)
    stock_data = stock_data.dropna(subset=['next_month_return'])
    dropped_count = initial_count - len(stock_data)
    print(f"\nDropped {dropped_count} rows with missing returns")
    print(f"Remaining records: {len(stock_data)}")
    
    if len(stock_data) == 0:
        print("Error: No valid data remaining after processing")
        return None, None, None
    
    # Prepare features and target
    features = ['rsi', 'macd_hist', 'bb_position', 'pe_ratio', 'pb_ratio', 
                'debt_to_equity', 'roe', 'revenue_growth']
    
    # Check for missing features
    missing_features = [f for f in features if f not in stock_data.columns]
    if missing_features:
        print(f"Error: Missing required features: {missing_features}")
        return None, None, None
    
    X = stock_data[features]
    y = stock_data['next_month_return']
    
    print("\nData preparation complete")
    print(f"Final dataset size: {len(X)} records")
    print(f"Features: {features}")
    
    return X, y, stock_data

def train_model(X, y):
    """Train machine learning model"""
    print("Training model...")
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"Model Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def get_stock_name(ts_code):
    """Get stock name from Tushare"""
    try:
        stock_info = pro.stock_basic(ts_code=ts_code)
        if stock_info is not None and not stock_info.empty:
            return stock_info['name'].iloc[0]
        return ts_code
    except:
        return ts_code

def predict_monthly_stocks(model, stock_data, start_date, end_date):
    """Predict returns for stocks each month"""
    print("\nGenerating monthly stock recommendations...")
    features = ['rsi', 'macd_hist', 'bb_position', 'pe_ratio', 'pb_ratio', 
                'debt_to_equity', 'roe', 'revenue_growth']
    
    # Generate monthly dates
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    monthly_recommendations = []
    
    for date in dates:
        # Get latest data before prediction date
        latest_data = stock_data[stock_data['date'] <= date.strftime('%Y-%m-%d')]
        if latest_data.empty:
            continue
            
        # Get predictions
        predictions = model.predict(latest_data[features])
        
        # Create DataFrame with predictions
        results = pd.DataFrame({
            'ts_code': latest_data['ts_code'],
            'predicted_return': predictions
        })
        
        # Sort by predicted return and get top 10
        top_stocks = results.sort_values('predicted_return', ascending=False).head(10)
        
        # Add stock names
        top_stocks['stock_name'] = top_stocks['ts_code'].apply(get_stock_name)
        
        # Store recommendations
        monthly_recommendations.append({
            'date': date,
            'recommendations': top_stocks
        })
        
        # Print recommendations for this month
        print(f"\nRecommendations for {date.strftime('%Y-%m')}:")
        print("-" * 100)
        print(f"{'Stock Code':<12} {'Stock Name':<30} {'Predicted Return':<15}")
        print("-" * 100)
        for _, row in top_stocks.iterrows():
            print(f"{row['ts_code']:<12} {row['stock_name']:<30} {row['predicted_return']:.2%}")
    
    # Save all recommendations to CSV
    all_recommendations = []
    for rec in monthly_recommendations:
        rec_df = rec['recommendations'].copy()
        rec_df['date'] = rec['date']
        all_recommendations.append(rec_df)
    
    if all_recommendations:
        final_recommendations = pd.concat(all_recommendations, ignore_index=True)
        final_recommendations = final_recommendations[['date', 'ts_code', 'stock_name', 'predicted_return']]
        final_recommendations.to_csv('monthly_recommendations.csv', index=False)
        print("\nSaved all recommendations to monthly_recommendations.csv")
    
    return monthly_recommendations

def plot_performance_comparison(monthly_recommendations, prices_df):
    """Plot cumulative returns comparison"""
    # Initialize arrays for daily returns and portfolio values
    all_dates = []
    strategy_daily_returns = []
    csi300_daily_returns = []
    strategy_portfolio_values = [1.0]
    csi300_portfolio_values = [1.0]
    
    # Sort recommendations by date
    monthly_recommendations.sort(key=lambda x: x['date'])
    
    # Get CSI300 data for the entire period
    start_date = monthly_recommendations[0]['date']
    end_date = datetime(2025, 4, 26)
    
    csi300 = pro.index_daily(ts_code='000300.SH', 
                            start_date=start_date.strftime('%Y%m%d'),
                            end_date=end_date.strftime('%Y%m%d'))
    
    if csi300 is not None and not csi300.empty:
        csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
        csi300 = csi300.sort_values('trade_date')
        csi300['daily_return'] = csi300['close'].pct_change()
        
        # Initialize strategy returns array
        strategy_daily_returns = [0] * len(csi300)
        
        # Process each month's recommendations
        for rec in monthly_recommendations:
            date = rec['date']
            next_month = date + timedelta(days=30)
            
            if next_month > datetime(2025, 4, 26):
                continue
            
            # Get recommended stocks for this month
            top_stocks = rec['recommendations']['ts_code'].tolist()
            stock_daily_data = {}
            
            # Fetch daily data for all recommended stocks
            for stock in top_stocks:
                try:
                    stock_data = pro.daily(ts_code=stock,
                                         start_date=date.strftime('%Y%m%d'),
                                         end_date=next_month.strftime('%Y%m%d'))
                    if stock_data is not None and not stock_data.empty:
                        stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'])
                        stock_data.set_index('trade_date', inplace=True)
                        stock_daily_data[stock] = stock_data['close']
                except:
                    continue
            
            if not stock_daily_data:
                continue
            
            # Calculate daily returns for each date in this month
            for j in range(len(csi300)):
                trade_date = csi300['trade_date'].iloc[j]
                if trade_date < date or trade_date > next_month:
                    continue
                
                # Calculate strategy daily return (average of this month's stocks)
                stock_returns = []
                for stock, prices in stock_daily_data.items():
                    if trade_date in prices.index:
                        today_price = prices[trade_date]
                        yesterday_price = prices[prices.index < trade_date].iloc[-1] if len(prices[prices.index < trade_date]) > 0 else today_price
                        if yesterday_price != 0:
                            stock_return = (today_price / yesterday_price) - 1
                            stock_returns.append(stock_return)
                
                if stock_returns:
                    strategy_daily_returns[j] = np.mean(stock_returns)
        
        # Calculate portfolio values
        current_month = None
        month_start_value = 1.0
        
        for i in range(1, len(csi300)):
            trade_date = csi300['trade_date'].iloc[i]
            all_dates.append(trade_date)
            csi300_daily_returns.append(csi300['daily_return'].iloc[i])
            
            # Update CSI300 portfolio value
            csi300_portfolio_values.append(csi300_portfolio_values[-1] * (1 + csi300['daily_return'].iloc[i]))
            
            # Check if we're starting a new month
            if current_month != trade_date.month:
                current_month = trade_date.month
                month_start_value = 1.0
            
            # For strategy, only update if we have a valid return
            if strategy_daily_returns[i] != 0:
                new_value = month_start_value * (1 + strategy_daily_returns[i])
                strategy_portfolio_values.append(new_value)
            else:
                strategy_portfolio_values.append(strategy_portfolio_values[-1])
    
    if not all_dates:
        return
    
    # Convert to numpy arrays
    strategy_daily_returns = np.array(strategy_daily_returns[1:])
    csi300_daily_returns = np.array(csi300_daily_returns)
    strategy_portfolio_values = np.array(strategy_portfolio_values[1:])
    csi300_portfolio_values = np.array(csi300_portfolio_values[1:])
    
    # Calculate monthly returns for performance metrics
    monthly_dates = []
    monthly_strategy_returns = []
    monthly_csi300_returns = []
    
    current_month = None
    month_start_idx = 0
    
    for i, date in enumerate(all_dates):
        if current_month != date.month:
            if current_month is not None:
                month_end_idx = i
                monthly_strategy_returns.append(strategy_portfolio_values[month_end_idx-1] / strategy_portfolio_values[month_start_idx] - 1)
                monthly_csi300_returns.append(csi300_portfolio_values[month_end_idx-1] / csi300_portfolio_values[month_start_idx] - 1)
                monthly_dates.append(all_dates[month_start_idx])
            current_month = date.month
            month_start_idx = i
    
    # Calculate Sharpe ratio (annualized)
    monthly_strategy_returns = np.array(monthly_strategy_returns)
    monthly_csi300_returns = np.array(monthly_csi300_returns)
    
    strategy_sharpe = np.mean(monthly_strategy_returns) / np.std(monthly_strategy_returns) * np.sqrt(12) if np.std(monthly_strategy_returns) != 0 else 0
    csi300_sharpe = np.mean(monthly_csi300_returns) / np.std(monthly_csi300_returns) * np.sqrt(12) if np.std(monthly_csi300_returns) != 0 else 0
    
    # Find the index for March 28, 2025
    plot_end_date = datetime(2025, 3, 28)
    plot_end_idx = np.searchsorted(all_dates, plot_end_date)
    
    # Plot portfolio values up to March 28
    plt.figure(figsize=(12, 6))
    plt.plot(all_dates[:plot_end_idx], strategy_portfolio_values[:plot_end_idx], 
             label='ML Strategy', linewidth=2)
    plt.plot(all_dates[:plot_end_idx], csi300_portfolio_values[:plot_end_idx], 
             label='CSI300', linewidth=2, linestyle='--')
    plt.title('ML Strategy vs CSI300 Portfolio Values (Jan-Mar 2025)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Starting from 1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ml_strategy_vs_csi300.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print performance metrics
    print("\nPerformance Metrics (Jan-Apr 2025):")
    print(f"Strategy Total Return: {(strategy_portfolio_values[-1] - 1) * 100:.2f}%")
    print(f"CSI300 Total Return: {(csi300_portfolio_values[-1] - 1) * 100:.2f}%")
    print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
    print(f"CSI300 Sharpe Ratio: {csi300_sharpe:.2f}")
    
    # Save performance metrics
    performance_df = pd.DataFrame({
        'Date': all_dates,
        'Strategy_Daily_Return': strategy_daily_returns,
        'CSI300_Daily_Return': csi300_daily_returns,
        'Strategy_Portfolio_Value': strategy_portfolio_values,
        'CSI300_Portfolio_Value': csi300_portfolio_values
    })
    performance_df.to_csv('performance_metrics_2025.csv', index=False)

def debug_csi300_calculation():
    """Debug function to verify CSI300 returns calculation"""
    print("\nDebugging CSI300 returns calculation...")
    
    # Initialize Tushare
    ts.set_token('30dd89a41c53facb136d77f8ba652b5963fe96919483ac2a4a111a4f')
    pro = ts.pro_api()
    
    # Get CSI300 data from Jan 1 to Apr 29, 2025
    start_date = '20250101'
    end_date = '20250429'
    
    print(f"Fetching CSI300 data from {start_date} to {end_date}...")
    csi300 = pro.index_daily(ts_code='000300.SH', 
                            start_date=start_date,
                            end_date=end_date)
    
    if csi300 is None or csi300.empty:
        print("Error: No CSI300 data found")
        return
    
    # Sort by date
    csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
    csi300 = csi300.sort_values('trade_date')
    
    # Calculate daily returns
    csi300['daily_return'] = csi300['close'].pct_change()
    
    # Calculate cumulative portfolio value
    portfolio_value = 1.0
    portfolio_values = [portfolio_value]
    
    for i in range(1, len(csi300)):
        portfolio_value *= (1 + csi300['daily_return'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    # Print detailed information
    print("\nFirst few days of data:")
    print(csi300[['trade_date', 'close', 'daily_return']].head())
    
    print("\nLast few days of data:")
    print(csi300[['trade_date', 'close', 'daily_return']].tail())
    
    print(f"\nTotal return: {(portfolio_values[-1] - 1) * 100:.2f}%")
    print(f"Starting price: {csi300['close'].iloc[0]:.2f}")
    print(f"Ending price: {csi300['close'].iloc[-1]:.2f}")
    print(f"Simple return: {(csi300['close'].iloc[-1] / csi300['close'].iloc[0] - 1) * 100:.2f}%")

def main():
    # Initialize Tushare
    print("Initializing Tushare...")
    ts.set_token('YOUR TOKEN HERE')
    global pro
    pro = ts.pro_api()
    
    # Debug CSI300 calculation first
    debug_csi300_calculation()
    
    # Prepare and train model
    X, y, stock_data = prepare_training_data()
    if X is None or y is None or stock_data is None:
        print("Error: Failed to prepare training data. Exiting.")
        return
    
    print("\nTraining model...")
    model = train_model(X, y)
    
    # Generate monthly recommendations for 2025, using 2024 data for January predictions
    print("\nGenerating recommendations for 2025...")
    start_date = datetime(2025, 1, 1)  # Start from January 2025
    end_date = datetime(2025, 4, 26)   # End date is April 26, 2025
    monthly_recommendations = predict_monthly_stocks(model, stock_data, start_date, end_date)
    
    # Plot performance comparison
    print("\nGenerating performance plots...")
    plot_performance_comparison(monthly_recommendations, None)  # We don't need prices_df anymore
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 
