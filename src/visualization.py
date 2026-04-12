import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_dir    = os.path.join(BASE_DIR, 'data', 'processed')
outputs_dir = os.path.join(BASE_DIR, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150

# load data
delay_df = pd.read_csv(os.path.join(proc_dir, 'cleaned_delay_data.csv'))
env_df   = pd.read_csv(os.path.join(proc_dir, 'cleaned_environmental_data.csv'))

delay_df['date']   = pd.to_datetime(delay_df['date'],   errors='coerce')
env_df['datetime'] = pd.to_datetime(env_df['datetime'], errors='coerce')

env_df['year']    = env_df['datetime'].dt.year
env_df['month']   = env_df['datetime'].dt.month
env_df['quarter'] = env_df['datetime'].dt.to_period('Q').astype(str)

# merge for any plots that need both datasets together
merged = pd.merge(env_df, delay_df, on='quarter', how='inner')

# save the merged dataset in case it's useful for further analysis
merged_out = os.path.join(outputs_dir, 'merged_data.csv')
merged.to_csv(merged_out, index=False)
logger.info(f'merged dataset saved — {merged.shape[0]} rows')

# correlation heatmap
# wanted to see which weather variables actually relate to cancellations

numeric_cols = merged.select_dtypes(include='number')

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix: Environmental Factors vs Train Disruptions',
          fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'correlation_heatmap.png'), bbox_inches='tight')
plt.close()
logger.info('saved: correlation_heatmap.png')


# time series of the three main weather variables

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(env_df['datetime'], env_df['temp'], color='tomato', linewidth=1.5)
axes[0].set_ylabel('Avg Temp (°F)')
axes[0].set_title('Temperature Over Time')

axes[1].plot(env_df['datetime'], env_df['precip'], color='steelblue', linewidth=1.5)
axes[1].set_ylabel('Precipitation')
axes[1].set_title('Precipitation Over Time')

axes[2].plot(env_df['datetime'], env_df['wind_gust'], color='seagreen', linewidth=1.5)
axes[2].set_ylabel('Wind Gust (mph)')
axes[2].set_title('Wind Gust Over Time')
axes[2].set_xlabel('Date')

plt.suptitle('Environmental Conditions — Scotland Railway Study Period',
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'environmental_timeseries.png'), bbox_inches='tight')
plt.close()
logger.info('saved: environmental_timeseries.png')


# cancellations by quarter bar chart
# sorting by date first so the x-axis actually reads chronologically

plt.figure(figsize=(14, 5))
sns.barplot(data=delay_df.sort_values('date'),
            x='quarter', y='total_cancellations',
            hue='quarter', palette='Reds_r', legend=False)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Quarter')
plt.ylabel('Total Cancellations')
plt.title('Total Train Cancellations by Quarter')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'cancellations_by_quarter.png'), bbox_inches='tight')
plt.close()
logger.info('saved: cancellations_by_quarter.png')


# wind gust vs cancellation score scatter
# colouring by month to see if seasonal clustering is visible

plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged, x='wind_gust', y='cancellation_score',
                hue='month', palette='coolwarm', alpha=0.8)
plt.xlabel('Wind Gust (mph)')
plt.ylabel('Cancellation Score')
plt.title('Wind Gust vs Cancellation Score (coloured by Month)')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'wind_vs_cancellations.png'), bbox_inches='tight')
plt.close()
logger.info('saved: wind_vs_cancellations.png')


# seasonal boxplot

plt.figure(figsize=(12, 5))
sns.boxplot(data=merged, x='month', y='cancellation_score',
            hue='month', palette='Blues', legend=False)
plt.xlabel('Month')
plt.ylabel('Cancellation Score')
plt.title('Seasonal Distribution of Train Cancellations')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'seasonal_cancellations_boxplot.png'), bbox_inches='tight')
plt.close()
logger.info('saved: seasonal_cancellations_boxplot.png')

logger.info('all charts done')
