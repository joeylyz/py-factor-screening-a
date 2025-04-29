import tushare as ts
import pandas as pd
from datetime import datetime

def get_china_stocks():
    """
    Get all listed stocks in China and save to CSV
    """
    # Initialize Tushare
    ts.set_token('INPUT YOUR LICENSE HERE')
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
        
        # Save to CSV
        output_file = 'china_stocks.csv'
        stocks.to_csv(output_file, index=False)
        print(f"Successfully saved {len(stocks)} stocks to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    get_china_stocks() 
