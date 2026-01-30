import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import RossmannPipeline
from src.core import setup_logger

logger = setup_logger(__name__)

def visualize_model_performance():
    """
    Evaluates the production model and generates visual reports.
    """
    train_csv = os.path.abspath("data/raw/train.csv")
    model_path = os.path.abspath("models/rossmann_production_model.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"Production model not found at {model_path}. Run production training first.")
        return

    logger.info("Initializing Visualization Pipeline...")
    pipeline = RossmannPipeline(train_csv)
    
    # Load model
    with open(model_path, 'rb') as f:
        pipeline.model = pickle.load(f)
        
    # 1. Prepare Validation Data (Final month of data)
    df_raw = pipeline.ingestor.ingest(train_csv)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    
    # Take latest 30 days for evaluation
    max_date = df_raw['Date'].max()
    eval_df = df_raw[df_raw['Date'] > (max_date - pd.Timedelta(days=30))]
    
    logger.info(f"Evaluating on {len(eval_df)} records from {eval_df['Date'].min().date()} to {max_date.date()}")
    df_feat = pipeline.run_feature_engineering(eval_df)
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['StoreType', 'Assortment']:
        if col in df_feat.columns:
            df_feat[col] = le.fit_transform(df_feat[col].astype(str))

    feature_cols = [
        'Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
        'Year', 'Month', 'Day', 'IsWeekend', 'DayOfMonth',
        'CompetitionDistance', 'CompetitionOpenTime', 'StoreType', 'Assortment'
    ] + [c for c in df_feat.columns if 'fourier' in c or 'easter' in c]
    
    X_eval = df_feat[feature_cols].fillna(0)
    y_eval = df_feat['target']
    
    # 2. Generate Predictions
    y_pred_log = pipeline.model.predict(X_eval)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_eval)
    
    metrics = pipeline.evaluate(X_eval, y_eval)
    logger.info(f"Evaluation Metrics: {metrics}")
    
    # --- VISUALIZATION ---
    os.makedirs('reports/figures', exist_ok=True)
    sns.set(style='whitegrid', palette='muted')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif'] # Fix for potential glyph issues
    plt.rcParams['axes.unicode_minus'] = False

    # Plot 1: Actual vs Predicted (Time Series Sample)
    plt.figure(figsize=(15, 6))
    # Aggregate by date for a cleaner plot
    eval_plot_df = pd.DataFrame({'Date': df_feat['Date'], 'Actual': y_true, 'Predicted': y_pred})
    ts_agg = eval_plot_df.groupby('Date').mean()
    
    plt.plot(ts_agg.index, ts_agg['Actual'], label='Actual Sales (Mean)', color='blue', alpha=0.7)
    plt.plot(ts_agg.index, ts_agg['Predicted'], label='Predicted Sales (Mean)', color='red', linestyle='--', alpha=0.9)
    plt.fill_between(ts_agg.index, ts_agg['Predicted']*0.9, ts_agg['Predicted']*1.1, color='red', alpha=0.1, label='10% error margin')
    
    plt.title('XGBoost Model: Daily Sales Forecast vs Actuals', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    plt.close()

    # Plot 2: Residual Distribution
    plt.figure(figsize=(10, 6))
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residual Distribution (Forecast Error)', fontsize=15)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.savefig('reports/figures/residuals.png')
    plt.close()

    # Plot 3: Feature Importance (Gains for XGBoost)
    plt.figure(figsize=(12, 8))
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': pipeline.model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
    plt.title('XGBoost Model Interpretability: Feature Importance (Gain)', fontsize=15)
    plt.savefig('reports/figures/feature_importance.png')
    plt.close()

    logger.info("Visualizations saved to reports/figures/")
    print(f"\nFinal SMAPE: {metrics['SMAPE']:.2f}%")
    print(f"Final MAE: {metrics['MAE']:.2f}")

if __name__ == "__main__":
    visualize_model_performance()
