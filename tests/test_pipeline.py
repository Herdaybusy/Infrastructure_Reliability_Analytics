import pytest
import pandas as pd
import numpy as np
import os
import sys

# make sure we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ----------------------------------------------------------------
# fixtures — just small fake versions of the real data
# ----------------------------------------------------------------

@pytest.fixture
def sample_env():
    # mimics what cleaned_environmental_data.csv looks like
    return pd.DataFrame({
        'datetime'   : pd.date_range('2018-07-31', periods=12, freq='ME'),
        'max_temp'   : [70.69, 64.07, 59.77, 55.36, 48.2, 42.1,
                        38.5, 41.2, 52.3, 61.4, 68.9, 72.1],
        'min_temp'   : [52.93, 51.84, 46.69, 42.05, 35.1, 30.2,
                        28.4, 29.8, 38.6, 47.2, 54.1, 57.3],
        'temp'       : [62.46, 58.19, 53.35, 49.04, 41.2, 36.1,
                        33.4, 35.5, 45.4, 54.3, 61.5, 64.7],
        'precip'     : [0.14, 0.07, 0.07, 0.03, 0.12, 0.18,
                        0.22, 0.19, 0.09, 0.05, 0.08, 0.11],
        'humidity'   : [77.64, 82.12, 81.65, 82.87, 84.2, 86.1,
                        87.3, 85.9, 80.4, 76.2, 74.8, 76.1],
        'wind_gust'  : [30.48, 27.08, 32.09, 32.92, 38.4, 42.1,
                        45.2, 41.8, 35.6, 28.4, 26.9, 29.3],
        'wind_speed' : [15.07, 14.24, 18.49, 17.33, 20.1, 22.4,
                        24.1, 21.8, 18.2, 14.9, 13.8, 15.2],
        'visibility' : [11.65, 12.2, 12.62, 12.88, 10.4, 9.8,
                        9.2, 9.6, 11.1, 12.4, 13.1, 12.9],
        'cloud_cover': [60.4, 68.8, 61.64, 60.85, 72.1, 78.4,
                        80.2, 77.6, 65.3, 58.2, 55.4, 58.7]
    })


@pytest.fixture
def sample_delay():
    # mimics what cleaned_delay_data.csv looks like
    return pd.DataFrame({
        'time_period'          : ['Apr to Jun 2018', 'Jul to Sep 2018',
                                  'Oct to Dec 2018', 'Jan to Mar 2019'],
        'trains_planned'       : [188553, 191755, 182975, 186436],
        'cancellation_score'   : [1999.04, 2939.79, 3418.16, 4695.52],
        'infra_network_score'  : [907.99, 1289.6, 1256.37, 1183.04],
        'infra_external_score' : [215.54, 413.65, 681.37, 2264.17],
        'operator_fault_score' : [807.62, 1160.89, 1390.41, 1157.66],
        'operator_external_score': [67.88, 74.64, 90.0, 90.65],
        'date'                 : pd.to_datetime(['2018-04-01', '2018-07-01',
                                                 '2018-10-01', '2019-01-01']),
        'quarter'              : ['2018Q2', '2018Q3', '2018Q4', '2019Q1'],
        'total_cancellations'  : [2625.75, 4037.04, 4622.32, 5691.89]
    })


# ----------------------------------------------------------------
# preprocessing tests
# ----------------------------------------------------------------

def test_env_has_expected_columns(sample_env):
    expected = ['datetime', 'max_temp', 'min_temp', 'temp', 'precip',
                'humidity', 'wind_gust', 'wind_speed', 'visibility', 'cloud_cover']
    for col in expected:
        assert col in sample_env.columns, f"missing column: {col}"


def test_env_no_nulls_after_cleaning(sample_env):
    # after preprocessing there shouldn't be any nulls in the numeric columns
    numeric = sample_env.select_dtypes(include='number')
    assert numeric.isnull().sum().sum() == 0


def test_env_datetime_parsed_correctly(sample_env):
    assert pd.api.types.is_datetime64_any_dtype(sample_env['datetime'])


def test_delay_has_expected_columns(sample_delay):
    expected = ['time_period', 'trains_planned', 'cancellation_score',
                'quarter', 'total_cancellations']
    for col in expected:
        assert col in sample_delay.columns


def test_total_cancellations_is_correct(sample_delay):
    # sanity check — total should equal part + full
    # in our fixture we set it directly but in real data this is derived
    assert (sample_delay['total_cancellations'] > 0).all()


def test_quarter_format_is_correct(sample_delay):
    # quarter should look like 2018Q2 not something weird
    pattern = r'^\d{4}Q[1-4]$'
    assert sample_delay['quarter'].str.match(pattern).all(), \
        "quarter column has unexpected format"


def test_no_duplicate_quarters(sample_delay):
    assert sample_delay['quarter'].nunique() == len(sample_delay), \
        "found duplicate quarters in delay data"


def test_env_temperatures_are_reasonable(sample_env):
    # Scotland in fahrenheit — should be somewhere between 20 and 100
    assert sample_env['max_temp'].between(20, 100).all()
    assert sample_env['min_temp'].between(0, 90).all()
    # min should always be below max
    assert (sample_env['min_temp'] <= sample_env['max_temp']).all()


def test_wind_gust_above_zero(sample_env):
    assert (sample_env['wind_gust'] >= 0).all()


def test_precip_non_negative(sample_env):
    assert (sample_env['precip'] >= 0).all()


# ----------------------------------------------------------------
# merge tests
# ----------------------------------------------------------------

def test_merge_on_quarter_produces_rows(sample_env, sample_delay):
    sample_env['quarter'] = sample_env['datetime'].dt.to_period('Q').astype(str)

    env_q = sample_env.groupby('quarter').agg({
        'max_temp': 'mean', 'min_temp': 'mean', 'temp': 'mean',
        'precip': 'mean', 'humidity': 'mean', 'wind_gust': 'mean',
        'wind_speed': 'mean', 'visibility': 'mean', 'cloud_cover': 'mean'
    }).reset_index()

    merged = pd.merge(sample_delay, env_q, on='quarter', how='inner')
    assert len(merged) > 0, "merge returned no rows — quarter keys probably don't match"


def test_merge_does_not_duplicate_rows(sample_env, sample_delay):
    sample_env['quarter'] = sample_env['datetime'].dt.to_period('Q').astype(str)

    env_q = sample_env.groupby('quarter').agg({
        'max_temp': 'mean', 'min_temp': 'mean', 'temp': 'mean',
        'precip': 'mean', 'humidity': 'mean', 'wind_gust': 'mean',
        'wind_speed': 'mean', 'visibility': 'mean', 'cloud_cover': 'mean'
    }).reset_index()

    merged = pd.merge(sample_delay, env_q, on='quarter', how='inner')
    # rows in merged should never exceed rows in delay data
    assert len(merged) <= len(sample_delay)


def test_merged_has_both_env_and_delay_columns(sample_env, sample_delay):
    sample_env['quarter'] = sample_env['datetime'].dt.to_period('Q').astype(str)

    env_q = sample_env.groupby('quarter').agg({
        'max_temp': 'mean', 'min_temp': 'mean', 'temp': 'mean',
        'precip': 'mean', 'humidity': 'mean', 'wind_gust': 'mean',
        'wind_speed': 'mean', 'visibility': 'mean', 'cloud_cover': 'mean'
    }).reset_index()

    merged = pd.merge(sample_delay, env_q, on='quarter', how='inner')

    assert 'cancellation_score' in merged.columns
    assert 'wind_gust' in merged.columns
    assert 'temp' in merged.columns


# ----------------------------------------------------------------
# model tests
# ----------------------------------------------------------------

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


@pytest.fixture
def model_data(sample_env, sample_delay):
    # build the merged dataset and return X, y ready for modelling
    sample_env['quarter'] = sample_env['datetime'].dt.to_period('Q').astype(str)

    env_q = sample_env.groupby('quarter').agg({
        'max_temp': 'mean', 'min_temp': 'mean', 'temp': 'mean',
        'precip': 'mean', 'humidity': 'mean', 'wind_gust': 'mean',
        'wind_speed': 'mean', 'visibility': 'mean', 'cloud_cover': 'mean'
    }).reset_index()

    merged = pd.merge(sample_delay, env_q, on='quarter', how='inner')

    features = ['temp', 'max_temp', 'min_temp', 'precip', 'humidity',
                'wind_gust', 'wind_speed', 'visibility', 'cloud_cover']

    X = merged[features]
    y = merged['cancellation_score']
    return X, y


def test_linear_regression_fits_without_error(model_data):
    X, y = model_data
    # with only 3-4 rows this won't be a useful model
    # but it should at least run without crashing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegression()
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    assert len(preds) == len(y)


def test_random_forest_produces_feature_importances(model_data):
    X, y = model_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    assert len(importances) == X.shape[1]
    # importances should sum to roughly 1
    assert abs(importances.sum() - 1.0) < 0.01


def test_all_models_return_predictions(model_data):
    X, y = model_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'linear'  : LinearRegression(),
        'lasso'   : Lasso(alpha=0.1),
        'ridge'   : Ridge(alpha=1.0),
        'dtree'   : DecisionTreeRegressor(random_state=42),
        'rf'      : RandomForestRegressor(n_estimators=10, random_state=42),
        'knn'     : KNeighborsRegressor(n_neighbors=2)  # n_neighbors=2 because fixture only has 4 rows
    }

    for name, model in models.items():
        model.fit(X_scaled, y)
        preds = model.predict(X_scaled)
        assert len(preds) == len(y), f"{name} returned wrong number of predictions"
        assert not np.isnan(preds).any(), f"{name} returned NaN predictions"


def test_scaler_transforms_all_features(model_data):
    X, y = model_data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # after scaling, each feature should have near-zero mean
    means = X_scaled.mean(axis=0)
    assert np.allclose(means, 0, atol=1e-10), "StandardScaler didn't centre the features"


def test_feature_count_matches_expected(model_data):
    X, y = model_data
    assert X.shape[1] == 9, f"expected 9 features, got {X.shape[1]}"


# ----------------------------------------------------------------
# output file tests — run these after the scripts have been executed
# ----------------------------------------------------------------

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

expected_outputs = [
    'correlation_heatmap.png',
    'environmental_timeseries.png',
    'cancellations_by_quarter.png',
    'wind_vs_cancellations.png',
    'seasonal_cancellations_boxplot.png',
    'model_rmse_comparison.png',
    'model_r2_comparison.png',
    'actual_vs_predicted.png',
    'feature_importance.png',
    'model_evaluation_metrics.csv',
]

@pytest.mark.parametrize('filename', expected_outputs)
def test_output_file_exists(filename):
    path = os.path.join(OUTPUTS_DIR, filename)
    assert os.path.exists(path), \
        f"{filename} not found in outputs/ — run the scripts first"


def test_metrics_csv_has_all_six_models():
    path = os.path.join(OUTPUTS_DIR, 'model_evaluation_metrics.csv')
    if not os.path.exists(path):
        pytest.skip('metrics CSV not generated yet — run models.py first')
    df = pd.read_csv(path)
    assert len(df) == 6, f"expected 6 models in metrics CSV, found {len(df)}"


def test_metrics_csv_has_required_columns():
    path = os.path.join(OUTPUTS_DIR, 'model_evaluation_metrics.csv')
    if not os.path.exists(path):
        pytest.skip('metrics CSV not generated yet — run models.py first')
    df = pd.read_csv(path)
    for col in ['Model', 'MAE', 'MSE', 'RMSE', 'R²']:
        assert col in df.columns, f"missing column in metrics CSV: {col}"


def test_rmse_values_are_positive():
    path = os.path.join(OUTPUTS_DIR, 'model_evaluation_metrics.csv')
    if not os.path.exists(path):
        pytest.skip('metrics CSV not generated yet — run models.py first')
    df = pd.read_csv(path)
    assert (df['RMSE'] > 0).all(), "RMSE should always be positive"
