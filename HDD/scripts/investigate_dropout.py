import polars as pl
import glob
from datetime import datetime, timedelta

def investigate():
    q4_path = "Data/HardDrive/data_Q4_2025/data_Q4_2025/"
    csv_files = glob.glob(q4_path + "*.csv")
    csv_files.sort()
    
    # Pass 1: Find all failures
    failed_drives = {} 
    for file in csv_files:
        df = pl.read_csv(file, columns=['date', 'serial_number', 'failure', 'model'])
        fails = df.filter(pl.col('failure') == 1)
        if len(fails) > 0:
            for row in fails.iter_rows(named=True):
                failed_drives[row['serial_number']] = {
                    'date': datetime.strptime(row['date'], '%Y-%m-%d'),
                    'model': row['model']
                }
                
    # Pass 2: Re-scan to see who gets dropped and also check smart_18_raw null rate
    LEAD_DAYS = 7
    target_dates_range_for_failed = {
        serial: [
            (data['date'] - timedelta(days=LEAD_DAYS+1)).strftime('%Y-%m-%d'),
            (data['date'] - timedelta(days=LEAD_DAYS)).strftime('%Y-%m-%d'),
            (data['date'] - timedelta(days=LEAD_DAYS-1)).strftime('%Y-%m-%d')
        ]
        for serial, data in failed_drives.items()
    }
    
    found_precursors = set()
    
    # We also want to check smart_18_raw across all drives in a few files just to see sparsity
    smart_18_stats = []
    
    for file in csv_files:
        # Load safely
        df = pl.read_csv(file, ignore_errors=True)
        if len(df) == 0:
            continue
            
        file_date = df['date'][0]
        
        # Check targets
        serials_needed_today = [
            s for s, dates in target_dates_range_for_failed.items() 
            if file_date in dates and s not in found_precursors
        ]
        if serials_needed_today:
            todays_targets = df.filter(pl.col('serial_number').is_in(serials_needed_today))
            if len(todays_targets) > 0:
                found_precursors.update(todays_targets['serial_number'].to_list())
                
        # Sample smart_18_raw stats for the first few files
        if 'smart_18_raw' in df.columns and len(smart_18_stats) < 5:
            # aggregate null rate by model
            df = df.with_columns(pl.col('smart_18_raw').is_null().alias('is_null_18'))
            agg = df.group_by('model').agg([
                pl.len().alias('count'),
                pl.col('is_null_18').sum().alias('null_count')
            ])
            smart_18_stats.append(agg)
            
    # Analyze dropouts
    dropped = []
    for serial, data in failed_drives.items():
        if serial not in found_precursors:
            dropped.append(data['date'])
            
    print(f"Total failures: {len(failed_drives)}")
    print(f"Dropped failures: {len(dropped)}")
    
    week_counts = {}
    for d in dropped:
        week = d.isocalendar()[1]
        week_counts[week] = week_counts.get(week, 0) + 1
        
    print("\nDropped failures by week of year:")
    for week in sorted(week_counts.keys()):
        print(f"Week {week}: {week_counts[week]} drops")
        
    if smart_18_stats:
        print("\nsmart_18_raw null rate by model (sample from a few days):")
        combined = pl.concat(smart_18_stats).group_by('model').agg([
            pl.col('count').sum(),
            pl.col('null_count').sum()
        ])
        combined = combined.with_columns(
            (pl.col('null_count') / pl.col('count') * 100).alias('null_pct')
        )
        combined = combined.sort('count', descending=True)
        # only show models with more than 1000 drives
        major_models = combined.filter(pl.col('count') > 1000)
        for row in major_models.iter_rows(named=True):
            print(f"{row['model'].ljust(25)}: {row['null_pct']:.1f}% null ({row['count']} drives)")

if __name__ == "__main__":
    investigate()
