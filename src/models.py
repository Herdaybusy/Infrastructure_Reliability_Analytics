import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# logger 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger import get_logger

logger = get_logger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc_dir    = os.path.join(BASE_DIR, 'data', 'processed')
outputs_dir = os.path.join(BASE_DIR, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150


# load and merge

delay_df = pd.read_csv(os.path.join(proc_dir, 'cleaned_delay_data.csv'))
env_df   = pd.read_csv(os.path.join(proc_dir, 'cleaned_environmental_data.csv'))

delay_df['date']   = pd.to_datetime(delay_df['date'],   errors='coerce')
env_df['datetime'] = pd.to_datetime(env_df['datetime'], errors='coerce')
env_df['quarter']  = env_df['datetime'].dt.to_period('Q').astype(str)

# env data is monthly, delay data is quarterly
# aggregating to quarterly means 
env_q = env_df.groupby('quarter').agg({
    'max_temp'   : 'mean',
    'min_temp'   : 'mean',
    'temp'       : 'mean',
    'precip'     : 'mean',
    'humidity'   : 'mean',
    'wind_gust'  : 'mean',
    'wind_speed' : 'mean',
    'visibility' : 'mean',
    'cloud_cover': 'mean'
}).reset_index()

merged = pd.merge(delay_df, env_q, on='quarter', how='inner')
logger.info(f'merged shape: {merged.shape}')


# features and target

features = ['temp', 'max_temp', 'min_temp', 'precip', 'humidity',
            'wind_gust', 'wind_speed', 'visibility', 'cloud_cover']

X = merged[features]
y = merged['cancellation_score']

logger.info(f'X shape: {X.shape}')
logger.info(f'missing values:\n{X.isnull().sum()}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling for KNN and the regularised models
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

logger.info(f'train: {X_train.shape}  test: {X_test.shape}')


# training all six models
# lasso needs max_iter bumped up — default 1000 wasn't enough and threw a ConvergenceWarning

models = {
    'Linear Regression'  : LinearRegression(),
    'Lasso Regression'   : Lasso(alpha=0.1, max_iter=10000),
    'Ridge Regression'   : Ridge(alpha=1.0),
    'Decision Tree'      : DecisionTreeRegressor(random_state=42),
    'Random Forest'      : RandomForestRegressor(n_estimators=100, random_state=42),
    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'model' : model,
        'y_pred': y_pred,
        'MAE'   : mean_absolute_error(y_test, y_pred),
        'MSE'   : mean_squared_error(y_test, y_pred),
        'RMSE'  : np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2'    : r2_score(y_test, y_pred)
    }
    logger.info(f'{name} done')


# metrics summary

metrics_df = pd.DataFrame([
    {'Model': name, 'MAE': round(v['MAE'], 2), 'MSE': round(v['MSE'], 2),
     'RMSE': round(v['RMSE'], 2), 'R²': round(v['R2'], 4)}
    for name, v in results.items()
]).sort_values('RMSE').reset_index(drop=True)

logger.info(f'\n{metrics_df.to_string(index=False)}')

metrics_df.to_csv(os.path.join(outputs_dir, 'model_evaluation_metrics.csv'), index=False)


# RMSE bar chart

plt.figure(figsize=(10, 5))
sns.barplot(data=metrics_df, x='Model', y='RMSE',
            hue='Model', palette='Blues_r', legend=False)
plt.xticks(rotation=20, ha='right')
plt.ylabel('RMSE')
plt.title('Model Comparison — Root Mean Squared Error (lower is better)')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'model_rmse_comparison.png'), bbox_inches='tight')
plt.close()


# R² comparison 
# tomato for anything negative so it's obvious at a glance which models struggled

plt.figure(figsize=(10, 5))
colors = ['tomato' if v < 0 else 'steelblue' for v in metrics_df['R²']]
plt.bar(metrics_df['Model'], metrics_df['R²'], color=colors)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.xticks(rotation=20, ha='right')
plt.ylabel('R² Score')
plt.title('Model Comparison — R² Score (higher is better, negative = poor fit)')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'model_r2_comparison.png'), bbox_inches='tight')
plt.close()


# actual vs predicted for all six 

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, (name, v) in enumerate(results.items()):
    axes[i].scatter(y_test, v['y_pred'], alpha=0.7,
                    color='steelblue', edgecolors='white', linewidth=0.4)
    axes[i].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()], 'r--', linewidth=1.5)
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')
    axes[i].set_title(f'{name}\nRMSE: {v["RMSE"]:.2f} | R²: {v["R2"]:.4f}')

plt.suptitle('Actual vs Predicted — Cancellation Score', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'actual_vs_predicted.png'), bbox_inches='tight')
plt.close()


# random forest feature importance
# useful for understanding which weather variables are actually doing the work

rf = results['Random Forest']['model']
importance_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature',
            hue='Feature', palette='Greens_r', legend=False)
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'feature_importance.png'), bbox_inches='tight')
plt.close()

logger.info(f'\nfeature importance:\n{importance_df.to_string(index=False)}')
logger.info(f'best model by RMSE: {metrics_df.iloc[0]["Model"]}')
logger.info('all outputs saved to outputs/')
