import pandas as pd
import os
from datetime import datetime, timedelta
import json

class DataManager:
    def __init__(self, data_file="mental_health_data.csv"):
        self.data_file = data_file
        self.ensure_data_file_exists()
    
    def ensure_data_file_exists(self):
        """Create data file if it doesn't exist"""
        if not os.path.exists(self.data_file):
            # Create empty DataFrame with all required columns
            columns = ['date'] + [f'phq9_{i}' for i in range(9)] + [f'gad7_{i}' for i in range(7)]
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(self.data_file, index=False)
    
    def save_daily_entry(self, date, responses):
        """Save or update a daily entry"""
        # Load existing data
        df = pd.read_csv(self.data_file)
        
        # Convert date to string for consistency
        date_str = date.isoformat()
        
        # Remove existing entry for this date if it exists
        df = df[df['date'] != date_str]
        
        # Create new entry
        new_entry = {'date': date_str}
        
        # Add PHQ-9 responses
        for i in range(9):
            key = f'phq9_{i}'
            new_entry[key] = responses.get(key, False)
        
        # Add GAD-7 responses
        for i in range(7):
            key = f'gad7_{i}'
            new_entry[key] = responses.get(key, False)
        
        # Add new entry to dataframe
        new_df = pd.DataFrame([new_entry])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # Save to file
        df.to_csv(self.data_file, index=False)
    
    def get_data_for_date(self, date):
        """Get data for a specific date"""
        df = pd.read_csv(self.data_file)
        date_str = date.isoformat()
        
        matching_rows = df[df['date'] == date_str]
        if matching_rows.empty:
            return None
        
        return matching_rows.iloc[0].to_dict()
    
    def get_historical_data(self, start_date, end_date):
        """Get historical data for a date range"""
        df = pd.read_csv(self.data_file)
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter by date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        return filtered_df.sort_values('date')
    
    def get_rolling_data(self, end_date, days=14):
        """Get data for rolling window calculation"""
        df = pd.read_csv(self.data_file)
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        end_date = pd.to_datetime(end_date)
        start_date = end_date - timedelta(days=days-1)
        
        # Filter by date range
        filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        return filtered_df.sort_values('date')
    
    def get_all_data(self):
        """Get all data"""
        df = pd.read_csv(self.data_file)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
        return df
