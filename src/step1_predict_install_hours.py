"""
STEP 1: Predict Install Hours per Serial Number
Ridge Regression — no calendar information used here.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'device_lifecycle_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ridge_model.pkl')


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def build_features(df):
    """Select features — no calendar, no EOS, no procurement info."""
    feature_cols = [
        'device_type',
        'site_type',
        'vendor',
        'engineers_available',
        'available_window_days',
        'device_complexity_score',
    ]
    X = df[feature_cols].copy()
    y = df['install_hours_actual'].copy()
    return X, y


def build_pipeline(alpha=1.0):
    categorical = ['device_type', 'site_type', 'vendor']
    numerical = ['engineers_available', 'available_window_days', 'device_complexity_score']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Ridge(alpha=alpha))
    ])
    return pipeline


def train_and_evaluate(df=None):
    if df is None:
        df = load_data()

    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline(alpha=1.0)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    print(f"[Step 1] MAE: {mae:.2f} hours | R2: {r2:.3f} | CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"[Step 1] Model saved -> {MODEL_PATH}")

    return pipeline, mae, r2


def predict_all(df=None):
    """Predict install hours for every serial number."""
    if df is None:
        df = load_data()

    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    X, _ = build_features(df)
    df = df.copy()
    df['predicted_install_hours'] = pipeline.predict(X).clip(min=1)
    return df


if __name__ == '__main__':
    train_and_evaluate()
    df_with_predictions = predict_all()
    print(df_with_predictions[['serial_number', 'device_type', 'install_hours_actual', 'predicted_install_hours']].head(10))
