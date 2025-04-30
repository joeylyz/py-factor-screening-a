import tushare as ts
import pandas as pd
from datetime import datetime

def get_china_stocks():
    """
    Get all listed stocks in China and save to CSV
    """
    # Initialize Tushare
    ts.set_token('YOUT TOKEN HERE')
    pro = ts.pro_api()
    
    print("Getting all listed stocks in China...")
    
    try:
        # Get all listed stocks
        stocks = pro.query('stock_basic', 
                         exchange='', 
                         list_status='L', 
                         fields='ts_code,symbol,name,area,industry,list_date')
        
        if stocks is None or stocks.empty:
            print("Error: Could not get stock list")
            return
        
        print(f"Found {len(stocks)} total stocks")
        
        # Filter out Beijing stocks
        stocks = stocks[~stocks['ts_code'].str.endswith('.BJ')]
        print(f"After removing Beijing stocks: {len(stocks)} stocks")

        # Get current date for latest volume and market cap
        current_date = datetime.now().strftime('%Y%m%d')
        # Get daily data for volume
        daily_data = pro.daily(trade_date=current_date, fields='ts_code,vol')
        # Get daily basic data for market cap
        daily_basic_data = pro.daily_basic(trade_date=current_date, fields='ts_code,total_mv')

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

        # Save to CSV (filtered)
        output_file = 'china_stocks.csv'
        stocks.to_csv(output_file, index=False)
        print(f"Successfully saved {len(stocks)} stocks to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    get_china_stocks() 
