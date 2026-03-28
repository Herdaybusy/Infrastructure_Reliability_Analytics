import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir  = os.path.join(BASE_DIR, 'data', 'raw')
proc_dir = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(proc_dir, exist_ok=True)


# --- environmental data ---

env = pd.read_csv(os.path.join(raw_dir, 'environmental_data.csv'))
env = env.drop_duplicates()

# keeping the columns that are actually useful for the model
cols = ['datetime', 'tempmax', 'tempmin', 'temp', 'precip',
        'humidity', 'windgust', 'windspeed', 'winddir', 'visibility', 'cloudcover']
env = env[cols].copy()

env = env.rename(columns={
    'tempmax'   : 'max_temp',
    'tempmin'   : 'min_temp',
    'windspeed' : 'wind_speed',
    'windgust'  : 'wind_gust',
    'cloudcover': 'cloud_cover',
    'winddir'   : 'wind_dir'
})

numeric = ['max_temp', 'min_temp', 'temp', 'humidity',
           'wind_gust', 'wind_speed', 'wind_dir', 'visibility', 'cloud_cover']

env[numeric] = env[numeric].apply(pd.to_numeric, errors='coerce')
env['datetime'] = pd.to_datetime(env['datetime'], errors='coerce')

# filling any gaps with the column mean rather than dropping rows
for col in numeric:
    env[col] = env[col].fillna(env[col].mean())

env = env.dropna(subset=['datetime'])

# resample to monthly — the delay data is quarterly so we need to aggregate
# doing it at monthly first gives us more flexibility later
env = env.set_index('datetime')
monthly = env.resample('ME').mean().reset_index()

num_cols = monthly.select_dtypes(include=['float64', 'int64']).columns
monthly[num_cols] = monthly[num_cols].round(2)

env_out = os.path.join(proc_dir, 'cleaned_environmental_data.csv')
monthly.to_csv(env_out, index=False)
logger.info(f'environmental data saved — {monthly.shape[0]} monthly records')


# --- train delay data ---

delays = pd.read_excel(os.path.join(raw_dir, 'Train_delay.xlsx'))

# the original column names had newlines and extra spaces in them
delays.columns = (
    delays.columns
    .str.strip()
    .str.replace('\n', ' ')
    .str.replace('  ', ' ')
)

delays = delays.drop_duplicates()
delays = delays.dropna()
delays = delays.round(2)

delays = delays.rename(columns={
    'Time period'                                                                        : 'time_period',
    'National or Operator'                                                               : 'operator',
    'Number of trains planned'                                                           : 'trains_planned',
    'Number of trains part cancelled'                                                    : 'part_cancellations',
    'Number of trains full cancelled'                                                    : 'full_cancellations',
    'Cancellations score'                                                                : 'cancellation_score',
    'Cancellations score by responsibility, infrastructure and network management'       : 'infra_network_score',
    'Cancellations score by responsibility, infrastructure owner external event'         : 'infra_external_score',
    'Cancellations score by responsibility, train operator fault'                        : 'operator_fault_score',
    'Cancellations score by responsibility, operator external event'                     : 'operator_external_score',
    'Quarterly cancellations score (percentage)'                                         : 'quarterly_cancel_pct',
    'Moving annual average cancellations score (percentage)'                             : 'annual_cancel_avg'
})

# parse the date from the time_period string e.g. "Apr to Jun 2014"
start_month = delays['time_period'].str.split(' to ').str[0]
year = delays['time_period'].str[-4:]
delays['date'] = pd.to_datetime(start_month + ' ' + year, format='%b %Y', errors='coerce')
delay = delays.dropna(subset=['date'])
delays['quarter'] = delays['date'].dt.to_period('Q').astype(str)
delays['total_cancellations'] = delays['part_cancellations'] + delays['full_cancellations']

keep = ['time_period', 'trains_planned', 'cancellation_score',
        'infra_network_score', 'infra_external_score', 'operator_fault_score',
        'operator_external_score', 'date', 'quarter', 'total_cancellations']

delays = delays[keep]

delay_out = os.path.join(proc_dir, 'cleaned_delay_data.csv')
delays.to_csv(delay_out, index=False)
logger.info(f'delay data saved — {delays.shape[0]} quarterly records')
