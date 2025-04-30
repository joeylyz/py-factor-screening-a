import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime
import tushare as ts

def load_trading_history():
    """Load trading history and calculate portfolio returns"""
    # Load trading history
    trading_history = pd.read_csv('trading_history.csv')
    trading_history['Date'] = pd.to_datetime(trading_history['Date'])
    
    # Load stock metrics
    stock_data = []
    for filename in os.listdir('stock_data'):
        if filename.startswith('stock_metrics_') and filename.endswith('.csv'):
            file_path = os.path.join('stock_data', filename)
            df = pd.read_csv(file_path)
            stock_data.append(df)
    stock_data = pd.concat(stock_data, ignore_index=True)
    
    return trading_history, stock_data

def load_portfolio_returns():
    """Load actual monthly portfolio returns from CSV file, grouped by month."""
    df = pd.read_csv('portfolio_monthly_returns.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    # Group by month and take the mean if there are duplicate months
    df['YearMonth'] = df['Date'].dt.to_period('M')
    df = df.groupby('YearMonth')['Return'].mean().reset_index()
    df['Date'] = df['YearMonth'].dt.to_timestamp()
    df.set_index('Date', inplace=True)
    print("Portfolio returns index covers:", df.index.min(), "to", df.index.max())
    return df['Return']

def calculate_portfolio_returns():
    """Calculate portfolio returns and statistics using actual monthly returns."""
    portfolio_returns = load_portfolio_returns()
    stats_dict = {
        'Total Return': (1 + portfolio_returns).prod() - 1,
        'Annualized Return': (1 + portfolio_returns).prod() ** (12/len(portfolio_returns)) - 1,
        'Annualized Volatility': portfolio_returns.std() * np.sqrt(12),
        'Sharpe Ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12),
        'Max Drawdown': (1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()) - 1,
        'Win Rate': (portfolio_returns > 0).mean(),
        'Average Win': portfolio_returns[portfolio_returns > 0].mean(),
        'Average Loss': portfolio_returns[portfolio_returns < 0].mean(),
        'Skewness': stats.skew(portfolio_returns),
        'Kurtosis': stats.kurtosis(portfolio_returns)
    }
    return portfolio_returns, stats_dict

def calculate_risk_metrics(portfolio_returns):
    """Calculate various risk metrics"""
    # Calculate Value at Risk (VaR)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)
    
    # Calculate Expected Shortfall (ES)
    es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
    
    # Calculate Beta (using CSI300 as market)
    csi300 = pro.index_daily(ts_code='000300.SH', 
                            start_date=portfolio_returns.index[0].strftime('%Y%m%d'),
                            end_date=portfolio_returns.index[-1].strftime('%Y%m%d'))
    csi300['trade_date'] = pd.to_datetime(csi300['trade_date'])
    csi300.set_index('trade_date', inplace=True)
    csi300_returns = csi300['close'].pct_change()
    
    # Align dates
    common_dates = portfolio_returns.index.intersection(csi300_returns.index)
    portfolio_returns_aligned = portfolio_returns[common_dates]
    csi300_returns_aligned = csi300_returns[common_dates]
    
    # Calculate Beta
    covariance = np.cov(portfolio_returns_aligned, csi300_returns_aligned)[0, 1]
    market_variance = np.var(csi300_returns_aligned)
    beta = covariance / market_variance
    
    return {
        'VaR_95': var_95,
        'VaR_99': var_99,
        'ES_95': es_95,
        'ES_99': es_99,
        'Beta': beta
    }

def plot_performance_analysis(portfolio_returns, stats, risk_metrics):
    """Create various performance analysis plots"""
    plt.style.use('seaborn-v0_8')

    # 1. Cumulative Returns Plot
    plt.figure(figsize=(12, 6))
    portfolio_returns = portfolio_returns.sort_index()
    cum_returns = (1 + portfolio_returns).cumprod()
    cum_returns.plot(label='Portfolio')
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.legend()
    plt.xlim(cum_returns.index.min(), cum_returns.index.max())
    plt.xticks(cum_returns.index, [d.strftime('%b %Y') for d in cum_returns.index], rotation=45)
    plt.tight_layout()
    plt.savefig('cumulative_returns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Drawdown Plot
    plt.figure(figsize=(12, 6))
    drawdown = stats['Max Drawdown']
    drawdown.index = pd.to_datetime(drawdown.index)
    drawdown = drawdown.sort_index()
    drawdown.plot(label='Drawdown')
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.legend()
    plt.xlim(drawdown.index.min(), drawdown.index.max())
    plt.xticks(drawdown.index, [d.strftime('%b %Y') for d in drawdown.index], rotation=45)
    plt.tight_layout()
    plt.savefig('drawdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Returns Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(portfolio_returns, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('returns_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Rolling Statistics: Only rolling std, rolling Sharpe, rolling drawdown
    rolling_window = 12
    plt.figure(figsize=(12, 6))
    rolling_std = portfolio_returns.rolling(window=rolling_window).std()
    rolling_sharpe = portfolio_returns.rolling(window=rolling_window).mean() / rolling_std * np.sqrt(12)
    rolling_drawdown = (1 + portfolio_returns).cumprod().rolling(window=rolling_window).min() / (1 + portfolio_returns).cumprod() - 1
    rolling_std.plot(label='Rolling Std (Volatility)')
    rolling_sharpe.plot(label='Rolling Sharpe Ratio')
    rolling_drawdown.plot(label='Rolling Min Drawdown')
    plt.title('Rolling Statistics')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.xlim(portfolio_returns.index.min(), portfolio_returns.index.max())
    plt.xticks(portfolio_returns.index, [d.strftime('%b %Y') for d in portfolio_returns.index], rotation=45)
    plt.tight_layout()
    plt.savefig('rolling_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_performance_report(stats, risk_metrics):
    """Print comprehensive performance report"""
    print("\nPortfolio Performance Report")
    print("=" * 50)
    
    print("\nReturn Metrics:")
    print(f"Total Return: {stats['Total Return']:.2%}")
    print(f"Annualized Return: {stats['Annualized Return']:.2%}")
    print(f"Annualized Volatility: {stats['Annualized Volatility']:.2%}")
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {stats['Win Rate']:.2%}")
    print(f"Average Win: {stats['Average Win']:.2%}")
    print(f"Average Loss: {stats['Average Loss']:.2%}")
    
    print("\nRisk Metrics:")
    print(f"Maximum Drawdown: {stats['Max Drawdown'].min():.2%}")
    print(f"VaR (95%): {risk_metrics['VaR_95']:.2%}")
    print(f"VaR (99%): {risk_metrics['VaR_99']:.2%}")
    print(f"Expected Shortfall (95%): {risk_metrics['ES_95']:.2%}")
    print(f"Expected Shortfall (99%): {risk_metrics['ES_99']:.2%}")
    print(f"Beta: {risk_metrics['Beta']:.2f}")
    
    print("\nDistribution Statistics:")
    print(f"Skewness: {stats['Skewness']:.2f}")
    print(f"Kurtosis: {stats['Kurtosis']:.2f}")

def print_top_pnl_contributors():
    """Print a table of top 3 PnL contributors for each month."""
    trading_history = pd.read_csv('trading_history.csv')
    trading_history['Date'] = pd.to_datetime(trading_history['Date'])
    price_data = pd.read_csv('stock_data/all_price_data.csv')
    price_data['trade_date'] = pd.to_datetime(price_data['trade_date'], format='%Y%m%d')
    trading_history['YearMonth'] = trading_history['Date'].dt.to_period('M')
    months = trading_history['YearMonth'].unique()
    print(f"{'Month':<10} {'Stock':<12} {'PnL':>10}")
    for ym in months:
        month_trades = trading_history[trading_history['YearMonth'] == ym]
        pnl_dict = {}
        for _, row in month_trades.iterrows():
            ts_code = row['Stock']
            month_start = row['Date'].replace(day=1)
            next_month = (month_start + pd.offsets.MonthEnd(1)).replace(day=1)
            price_start = price_data[(price_data['ts_code'] == ts_code) & (price_data['trade_date'] >= month_start)].sort_values('trade_date')['close'].head(1)
            price_end = price_data[(price_data['ts_code'] == ts_code) & (price_data['trade_date'] < next_month)].sort_values('trade_date')['close'].tail(1)
            if not price_start.empty and not price_end.empty:
                pnl = price_end.values[0] - price_start.values[0]
                pnl_dict[ts_code] = pnl
        top3 = sorted(pnl_dict.items(), key=lambda x: x[1], reverse=True)[:3]
        for stock, pnl in top3:
            print(f"{str(ym):<10} {stock:<12} {pnl:>10.2f}")

def main():
    # Initialize Tushare
    ts.set_token('YOUR TOKEN HERE')
    global pro
    pro = ts.pro_api()
    
    # Load actual monthly returns
    portfolio_returns, stats = calculate_portfolio_returns()
    
    # Calculate risk metrics
    risk_metrics = calculate_risk_metrics(portfolio_returns)
    
    # Generate plots
    plot_performance_analysis(portfolio_returns, stats, risk_metrics)
    
    # Print performance report
    print_performance_report(stats, risk_metrics)

    # Print top PnL contributors
    print_top_pnl_contributors()

if __name__ == "__main__":
    main() 
